"""
Orpheus TTS streaming HTTP server (Windows / llama.cpp backend)

Multiple OrpheusCpp engines, multiple worker threads, shared FIFO job queue.
Each worker owns one engine and processes jobs serially; jobs race to grab
work from the shared queue, giving parallelism at the sentence level.

Run:
  .venv\\Scripts\\python.exe tts_server_orpheus.py

Endpoints:
  POST /tts       JSON {"text": "...", "voice": "tara", "temperature": 0.8}
                  -> audio/mpeg (streaming MP3)
  GET  /health    JSON {"status": "ok", "queue_depth": N, "voices": [...]}
"""

import os
import sys
import time
import queue
import threading
import uuid

import orpheus_bootstrap  # noqa: F401  — patches PATH, DLL dirs, and llama_cpp

import numpy as np
import logging
from flask import Flask, Response, request, jsonify
from services.audio_service import make_mp3_encoder
import config
logging.getLogger("werkzeug").setLevel(logging.ERROR)
from orpheus_cpp import OrpheusCpp
import orpheus_cpp.model as _orph_model


# ── Model quality toggle ────────────────────────────────────────────────────
_MODEL_MAP = {
    "FASTEST": ("lex-au/Orpheus-3b-FT-Q2_K.gguf",   "Orpheus-3b-FT-Q2_K.gguf"),
    "AVERAGE": ("isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF", "orpheus-3b-0.1-ft-q4_k_m.gguf"),
    "BEST":    ("lex-au/Orpheus-3b-FT-Q8_0.gguf",   "Orpheus-3b-FT-Q8_0.gguf"),
}

_chosen = config.TTS_QUALITY.upper()
if _chosen not in _MODEL_MAP:
    _chosen = "AVERAGE"
_repo_id, _filename = _MODEL_MAP[_chosen]
_orph_model.OrpheusCpp.lang_to_model["en"] = _repo_id

_orig_hf_hub_download = _orph_model.hf_hub_download
def _patched_hf_hub_download(repo_id, filename=None, **kwargs):
    if repo_id == _repo_id and filename and filename.lower() == _filename.lower():
        filename = _filename
    return _orig_hf_hub_download(repo_id, filename=filename, **kwargs)
_orph_model.hf_hub_download = _patched_hf_hub_download

print(f"  [model] Quality: {_chosen} -> {_repo_id}")


# ── Settings ────────────────────────────────────────────────────────────────

HOST = "0.0.0.0"
PORT = 8080
SAMPLE_RATE = 24000
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

DEFAULT_VOICE = config.TTS_VOICE
DEFAULT_TEMP = 0.8
DEFAULT_TOP_P = config.TTS_TOP_P
DEFAULT_MAX_TOKENS = 2000
DEFAULT_MIN_P = config.TTS_MIN_P
DEFAULT_PRE_BUFFER = 0

MP3_BITRATE = config.MP3_BITRATE
MAX_QUEUE_DEPTH = 16
NUM_WORKERS = config.TTS_NUM_WORKERS


# ── Streaming MP3 encoder wrapper ───────────────────────────────────────────

class Mp3Streamer:
    """Wrap lameenc to encode PCM chunks into MP3 bytes incrementally."""
    def __init__(self, sample_rate: int = SAMPLE_RATE, bitrate: int = MP3_BITRATE):
        self.enc = make_mp3_encoder(sample_rate, bitrate)

    def feed(self, pcm_int16: np.ndarray) -> bytes:
        if pcm_int16.size == 0:
            return b""
        return bytes(self.enc.encode(pcm_int16.astype(np.int16).tobytes()))

    def flush(self) -> bytes:
        return bytes(self.enc.flush())


# ── Job + worker ────────────────────────────────────────────────────────────

class Job:
    __slots__ = ("id", "text", "voice", "temperature", "top_p", "max_tokens", "min_p",
                 "fmt", "out_q", "cancelled", "enqueued_at", "started_at")

    def __init__(self, text, voice, temperature, top_p, max_tokens, min_p, fmt):
        self.id = uuid.uuid4().hex[:8]
        self.text = text
        self.voice = voice
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.min_p = min_p
        self.fmt = fmt  # "mp3" or "pcm"
        # Bytes chunks flow here; sentinel None signals end of stream.
        self.out_q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=64)
        self.cancelled = False
        self.enqueued_at = time.perf_counter()
        self.started_at = None


job_queue: "queue.Queue[Job | None]" = queue.Queue()
engines: list[OrpheusCpp] = []
current_jobs: set[Job] = set()
_current_jobs_lock = threading.Lock()


def worker_loop(worker_id: int, engine: OrpheusCpp):
    """Pull jobs FIFO, stream Orpheus output, encode to MP3, push bytes to job out_q."""
    while True:
        job = job_queue.get()
        if job is None:  # shutdown sentinel
            return

        with _current_jobs_lock:
            current_jobs.add(job)

        job.started_at = time.perf_counter()
        wait_ms = (job.started_at - job.enqueued_at) * 1000

        print(f"TTS-Perf: [{job.id}] worker={worker_id} start | voice={job.voice} fmt={job.fmt} min_p={job.min_p} wait={wait_ms:.0f}ms \"{job.text[:60]}\"", flush=True)

        try:
            if job.cancelled:
                job.out_q.put(None)
                continue

            streamer = Mp3Streamer() if job.fmt == "mp3" else None
            options = {
                "voice_id": job.voice,
                "temperature": job.temperature,
                "top_p": job.top_p,
                "max_tokens": job.max_tokens,
                "min_p": job.min_p,
                "pre_buffer_size": DEFAULT_PRE_BUFFER,
            }

            t_gen_start = time.perf_counter()
            t_first = None
            total_pcm_samples = 0

            for sr, chunk in engine.stream_tts_sync(job.text, options=options):
                if job.cancelled:
                    break
                pcm = chunk.flatten().astype(np.int16)
                total_pcm_samples += pcm.size

                if streamer is None:
                    out_bytes = pcm.tobytes()  # raw little-endian int16 PCM
                else:
                    out_bytes = streamer.feed(pcm)

                if out_bytes:
                    if t_first is None:
                        t_first = time.perf_counter()
                    try:
                        job.out_q.put(out_bytes, timeout=30)
                    except queue.Full:
                        job.cancelled = True
                        break

            if streamer is not None:
                tail = streamer.flush()
                if tail and not job.cancelled:
                    try:
                        job.out_q.put(tail, timeout=5)
                    except queue.Full:
                        pass

            audio_s = total_pcm_samples / SAMPLE_RATE if total_pcm_samples else 0
            gen_s = time.perf_counter() - t_gen_start
            ttfa_ms = (t_first - t_gen_start) * 1000 if t_first else 0
            rtf = (gen_s / audio_s) if audio_s > 0 else 0
            status = "cancelled" if job.cancelled else "ok"
            print(f"TTS-Perf: [{job.id}] worker={worker_id} {status} | ttfa={ttfa_ms:.0f}ms audio={audio_s:.1f}s rtf={rtf:.2f}x", flush=True)

        except Exception as e:
            print(f"TTS-Perf: [{job.id}] worker={worker_id} ERROR: {e}", flush=True)
        finally:
            job.out_q.put(None)  # signal end
            with _current_jobs_lock:
                current_jobs.discard(job)


# ── HTTP API ────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/tts", methods=["POST", "GET"])
def tts():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
    else:
        data = request.args

    text = (data.get("text") or "").strip()
    voice = data.get("voice") or DEFAULT_VOICE
    fmt = (data.get("format") or "mp3").lower()
    if fmt not in ("mp3", "pcm"):
        fmt = "mp3"
    try:
        temperature = float(data.get("temperature", DEFAULT_TEMP))
        top_p = float(data.get("top_p", DEFAULT_TOP_P))
        max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
        min_p = float(data.get("min_p", DEFAULT_MIN_P))
    except (TypeError, ValueError):
        return jsonify({"error": "bad numeric params"}), 400
    min_p = max(0.0, min(1.0, min_p))

    if not text:
        return jsonify({"error": "empty text"}), 400
    if voice not in VOICES:
        voice = DEFAULT_VOICE

    if job_queue.qsize() >= MAX_QUEUE_DEPTH:
        return jsonify({"error": "queue full", "queue_depth": job_queue.qsize()}), 503

    job = Job(text, voice, temperature, top_p, max_tokens, min_p, fmt)
    job_queue.put(job)

    def stream():
        try:
            while True:
                chunk = job.out_q.get()  # blocks until worker produces
                if chunk is None:
                    break
                yield chunk
        except GeneratorExit:
            # client disconnected
            job.cancelled = True

    if fmt == "pcm":
        content_type = f"audio/L16;rate={SAMPLE_RATE};channels=1"
    else:
        content_type = "audio/mpeg"

    headers = {
        "Content-Type": content_type,
        "X-Job-Id": job.id,
        "X-Sample-Rate": str(SAMPLE_RATE),
        "Cache-Control": "no-store",
    }
    return Response(stream(), headers=headers)


@app.route("/abort", methods=["POST"])
def abort():
    drained = 0
    while not job_queue.empty():
        try:
            j = job_queue.get_nowait()
            if j is not None:
                j.cancelled = True
                j.out_q.put(None)
                drained += 1
        except queue.Empty:
            break

    with _current_jobs_lock:
        cancelled_count = len(current_jobs)
        for job in current_jobs:
            job.cancelled = True

    return jsonify({
        "status": "ok",
        "drained": drained,
        "current_cancelled": cancelled_count,
    })


@app.route("/health", methods=["GET"])
def health():
    with _current_jobs_lock:
        active = len(current_jobs)
    return jsonify({
        "status": "ok" if len(engines) == NUM_WORKERS else "loading",
        "workers": len(engines),
        "active_jobs": active,
        "queue_depth": job_queue.qsize(),
        "voices": VOICES,
        "sample_rate": SAMPLE_RATE,
        "mp3_bitrate": MP3_BITRATE,
    })


# ── Main ────────────────────────────────────────────────────────────────────

def load_engines_and_warm():
    global engines
    print(f"Loading up to {NUM_WORKERS} Orpheus engines (llama.cpp + CUDA)...")
    for i in range(NUM_WORKERS):
        t0 = time.perf_counter()
        try:
            engine = OrpheusCpp(n_gpu_layers=-1, verbose=False, lang="en")
        except Exception as e:
            print(f"  engine {i+1} FAILED: {e}")
            print(f"  stopping at {len(engines)} engine(s).")
            break
        print(f"  engine {i+1}/{NUM_WORKERS} loaded in {time.perf_counter() - t0:.1f}s")
        engines.append(engine)
        time.sleep(1)  # let llama.cpp / CUDA state settle before next load

    if not engines:
        raise RuntimeError("No Orpheus engines could be loaded. Check VRAM and that no other TTS server is running.")

    actual_workers = len(engines)
    print(f"Warming up {actual_workers} engine(s)...")
    for i, engine in enumerate(engines):
        for _ in engine.stream_tts_sync(
            "Hello.",
            options={"voice_id": DEFAULT_VOICE, "temperature": DEFAULT_TEMP,
                     "top_p": DEFAULT_TOP_P, "max_tokens": 200, "min_p": DEFAULT_MIN_P, "pre_buffer_size": 0},
        ):
            pass
        print(f"  engine {i+1}/{actual_workers} warm")
    print("Ready.\n")


if __name__ == "__main__":
    load_engines_and_warm()

    actual_workers = len(engines)
    for i in range(actual_workers):
        worker = threading.Thread(
            target=worker_loop,
            args=(i, engines[i]),
            daemon=True,
            name=f"orpheus-worker-{i}",
        )
        worker.start()

    print(f"Orpheus TTS server listening on http://{HOST}:{PORT}")
    print(f"Workers: {actual_workers} (requested {NUM_WORKERS})")
    print(f"Voices: {', '.join(VOICES)}")
    print(f"Max queue depth: {MAX_QUEUE_DEPTH}")
    print()
    # threaded=True so multiple clients can be served while workers run
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
