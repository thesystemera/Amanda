"""
Orpheus TTS


 - Single-file test & interactive demo

Backend options:
  A) Local llama.cpp + CUDA (orpheus-cpp, Windows native)
  B) vLLM server via HTTP (WSL2 or remote Linux box)

Voices:  tara, leah, jess, leo, dan, mia, zac, zoe
Emotions: <laugh> <chuckle> <sigh> <gasp> <cough> <sniffle> <groan> <yawn>
"""

import os
import sys
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import orpheus_bootstrap  # noqa: F401  — patches PATH, DLL dirs, and llama_cpp

import time
import wave
import json
import queue
import struct
import threading
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np

# ── Settings ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24000
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
EMOTIONS = ["laugh", "chuckle", "sigh", "gasp", "cough", "sniffle", "groan", "yawn"]

# GPU layers: -1 = offload everything to GPU
N_GPU_LAYERS = -1

DEFAULT_VOICE = "tara"
DEFAULT_TEMP = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 2000
DEFAULT_PRE_BUFFER = 0  # 0 = yield first chunk ASAP (lowest latency)

# vLLM server URL (WSL2 default, or change to remote Linux box IP)
VLLM_SERVER_URL = "http://localhost:8080"

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_output")


# ── Helpers ─────────────────────────────────────────────────────────────────

def save_wav(audio_int16: np.ndarray, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


# ── Backend: Local llama.cpp ────────────────────────────────────────────────

class LocalEngine:
    """orpheus-cpp (llama.cpp + CUDA) backend."""

    def __init__(self):
        from orpheus_cpp import OrpheusCpp
        print(f"Loading Orpheus (llama.cpp + CUDA, GPU layers: {N_GPU_LAYERS})...")
        t0 = time.perf_counter()
        self.engine = OrpheusCpp(n_gpu_layers=N_GPU_LAYERS, verbose=False, lang="en")
        print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    @property
    def name(self):
        return "llama.cpp + CUDA (local)"

    def generate(self, text, voice=DEFAULT_VOICE, temperature=DEFAULT_TEMP):
        """Generate speech, return (audio_int16, stats)."""
        max_tokens = len(text) * 20
        options = {
            "voice_id": voice, "temperature": temperature,
            "top_p": DEFAULT_TOP_P, "max_tokens": max_tokens,
            "pre_buffer_size": DEFAULT_PRE_BUFFER,
        }
        t_start = time.perf_counter()
        t_first = None
        chunks = []

        for sr, chunk in self.engine.stream_tts_sync(text, options=options):
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(chunk)

        if not chunks:
            return np.array([], dtype=np.int16), {"error": "no audio"}

        audio = np.concatenate([c.flatten() for c in chunks]).astype(np.int16)
        t_end = time.perf_counter()
        duration = len(audio) / SAMPLE_RATE
        ttfa = (t_first - t_start) * 1000 if t_first else 0
        total = (t_end - t_start) * 1000

        return audio, {
            "ttfa_ms": round(ttfa),
            "total_ms": round(total),
            "duration_s": round(duration, 2),
            "rtf": round((total / 1000) / duration, 3) if duration > 0 else 0,
        }

    def stream_chunks(self, text, voice=DEFAULT_VOICE, temperature=DEFAULT_TEMP):
        """Yield (audio_int16_chunk, is_first) for streaming playback."""
        max_tokens = len(text) * 20
        options = {
            "voice_id": voice, "temperature": temperature,
            "top_p": DEFAULT_TOP_P, "max_tokens": max_tokens,
            "pre_buffer_size": DEFAULT_PRE_BUFFER,
        }
        first = True
        for sr, chunk in self.engine.stream_tts_sync(text, options=options):
            flat = chunk.flatten().astype(np.int16)
            yield flat, first
            first = False


# ── Backend: vLLM Server (HTTP) ─────────────────────────────────────────────

class ServerEngine:
    """Connects to tts_server.py running in WSL2 or on a remote Linux box."""

    def __init__(self, base_url=VLLM_SERVER_URL):
        self.base_url = base_url.rstrip("/")
        print(f"Connecting to vLLM server at {self.base_url}...")
        try:
            resp = urlopen(f"{self.base_url}/health", timeout=5)
            info = json.loads(resp.read())
            print(f"Server OK — voices: {', '.join(info.get('voices', []))}")
        except Exception as e:
            raise ConnectionError(f"Cannot reach server at {self.base_url}: {e}")

    @property
    def name(self):
        return f"vLLM server ({self.base_url})"

    def _stream_raw(self, text, voice=DEFAULT_VOICE, temperature=DEFAULT_TEMP):
        """Hit /tts endpoint, yield raw bytes (skipping WAV header)."""
        max_tokens = len(text) * 20
        body = json.dumps({
            "text": text, "voice": voice, "temperature": temperature,
            "top_p": DEFAULT_TOP_P, "max_tokens": max_tokens,
        }).encode()
        req = Request(
            f"{self.base_url}/tts", data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=120)

        # First 44 bytes are the WAV header — skip them
        header = resp.read(44)
        if len(header) < 44:
            return

        # Read in chunks
        while True:
            data = resp.read(4096)
            if not data:
                break
            yield data

    def generate(self, text, voice=DEFAULT_VOICE, temperature=DEFAULT_TEMP):
        """Generate speech, return (audio_int16, stats)."""
        t_start = time.perf_counter()
        t_first = None
        raw_parts = []

        for data in self._stream_raw(text, voice, temperature):
            if t_first is None:
                t_first = time.perf_counter()
            raw_parts.append(data)

        if not raw_parts:
            return np.array([], dtype=np.int16), {"error": "no audio"}

        raw = b"".join(raw_parts)
        audio = np.frombuffer(raw, dtype=np.int16)
        t_end = time.perf_counter()
        duration = len(audio) / SAMPLE_RATE
        ttfa = (t_first - t_start) * 1000 if t_first else 0
        total = (t_end - t_start) * 1000

        return audio, {
            "ttfa_ms": round(ttfa),
            "total_ms": round(total),
            "duration_s": round(duration, 2),
            "rtf": round((total / 1000) / duration, 3) if duration > 0 else 0,
        }

    def stream_chunks(self, text, voice=DEFAULT_VOICE, temperature=DEFAULT_TEMP):
        """Yield (audio_int16_chunk, is_first) for streaming playback."""
        first = True
        for data in self._stream_raw(text, voice, temperature):
            chunk = np.frombuffer(data, dtype=np.int16)
            if len(chunk) > 0:
                yield chunk, first
                first = False


# ── Playback ────────────────────────────────────────────────────────────────

def play_streaming(engine, text, voice=DEFAULT_VOICE, temperature=DEFAULT_TEMP):
    """Stream to speakers in real-time, return stats."""
    import sounddevice as sd

    audio_q = queue.Queue()
    finished = threading.Event()

    def callback(outdata, frames, time_info, status):
        try:
            data = audio_q.get_nowait()
        except queue.Empty:
            outdata.fill(0)
            return
        if data is None:
            outdata.fill(0)
            finished.set()
            raise sd.CallbackStop()
        n = min(len(data), frames)
        outdata[:n, 0] = data[:n]
        if n < frames:
            outdata[n:, 0] = 0

    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=2048, callback=callback)
    stream.start()

    t_start = time.perf_counter()
    t_first = None
    total_samples = 0

    for chunk, is_first in engine.stream_chunks(text, voice, temperature):
        if t_first is None:
            t_first = time.perf_counter()
        audio_q.put(chunk)
        total_samples += len(chunk)

    audio_q.put(None)
    finished.wait(timeout=total_samples / SAMPLE_RATE + 2)
    stream.stop()
    stream.close()

    duration = total_samples / SAMPLE_RATE
    ttfa = (t_first - t_start) * 1000 if t_first else 0
    total = (time.perf_counter() - t_start) * 1000

    return {
        "ttfa_ms": round(ttfa),
        "total_ms": round(total),
        "duration_s": round(duration, 2),
        "rtf": round((total / 1000) / duration, 3) if duration > 0 else 0,
    }


# ── Benchmark ───────────────────────────────────────────────────────────────

def run_benchmark(engine):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BENCHMARK_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 65)
    print("  LATENCY BENCHMARK")
    print(f"  Backend: {engine.name}")
    print(f"  Output: {run_dir}")
    print("=" * 65)

    tests = [
        ("Short",    "tara", "Hello there, how are you?"),
        ("Medium",   "tara", "The quick brown fox jumps over the lazy dog on a sunny afternoon."),
        ("Emotion",  "tara", "Oh wow! <laugh> That's amazing! <sigh> But I'm exhausted honestly."),
        ("Long",     "tara", "So here's the thing. We need to talk about the project timeline. "
                             "I know nobody wants to hear that. But we've got some real challenges "
                             "ahead of us. The good news is I think we can pull this off."),
        ("Leah",     "leah", "<sigh> Well, I don't really know what to say. <chuckle> Here we are."),
        ("Leo",      "leo",  "Let me walk you through the plan. I think you'll like this approach."),
        ("Jess+emo", "jess", "Oh my god! <gasp> That's incredible! <laugh> I can't believe it!"),
    ]

    print("\nWarming up...")
    engine.generate("Hello.", voice="tara")
    print()

    print(f"{'Label':<10} {'Voice':<6} {'Chars':>5} {'TTFA':>8} {'Total':>8} {'Audio':>7} {'RTF':>6}")
    print("-" * 65)

    results = []
    for label, voice, text in tests:
        audio, stats = engine.generate(text, voice=voice)
        if "error" not in stats:
            print(f"{label:<10} {voice:<6} {len(text):>5} "
                  f"{stats['ttfa_ms']:>6}ms {stats['total_ms']:>6}ms "
                  f"{stats['duration_s']:>6.1f}s {stats['rtf']:>5.2f}x")

            wav_name = f"{label.lower()}_{voice}.wav"
            save_wav(audio, os.path.join(run_dir, wav_name))
            results.append({**stats, "label": label, "voice": voice, "text": text, "wav": wav_name})

    if results:
        avg_ttfa = sum(r["ttfa_ms"] for r in results) / len(results)
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        print("-" * 65)
        print(f"{'AVERAGE':<10} {'':6} {'':>5} {avg_ttfa:>6.0f}ms {'':>8} {'':>7} {avg_rtf:>5.2f}x")

    report = {
        "timestamp": timestamp,
        "gpu": "N/A",
        "backend": engine.name,
        "model": "orpheus-3b-0.1-ft",
        "avg_ttfa_ms": round(avg_ttfa) if results else 0,
        "avg_rtf": round(avg_rtf, 3) if results else 0,
        "tests": results,
    }
    try:
        import torch
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 65)
    print("TTFA = Time to first audio | RTF = Real-time factor (<1.0 = faster than real-time)")
    print(f"WAVs + results.json saved to: {run_dir}\n")


# ── Batch Benchmark ─────────────────────────────────────────────────────────

def split_sentences(text):
    """Split text on sentence boundaries (. ! ?) keeping the punctuation."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def run_batch_benchmark(engine):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BENCHMARK_DIR, f"batch_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 65)
    print("  BATCH / SENTENCE-CHUNKING BENCHMARK")
    print(f"  Backend: {engine.name}")
    print(f"  Output: {run_dir}")
    print("=" * 65)
    print()
    print("Simulates your chatbot pipeline: split on sentences, generate each,")
    print("measure total wall time and per-sentence stats.")
    print()

    paragraphs = [
        (
            "tara",
            "Hello there! <laugh> It's so wonderful to finally meet you. "
            "I've been looking forward to this conversation for weeks. "
            "<chuckle> You know, I was telling my friend just the other day how excited I was. "
            "<sigh> But honestly, I'm a bit nervous too. "
            "Tell me, how has your day been so far?"
        ),
        (
            "leo",
            "So here's the thing. "
            "We need to talk about the project timeline. "
            "<sigh> I know nobody wants to hear that. "
            "But look, we've got some challenges ahead. "
            "The good news? <chuckle> I think we can actually pull this off if we work together."
        ),
    ]

    print("Warming up...")
    engine.generate("Hello.", voice="tara")
    print()

    all_results = []

    for voice, paragraph in paragraphs:
        sentences = split_sentences(paragraph)
        print(f"── Paragraph ({voice}, {len(sentences)} sentences) ──")
        print(f"  Full text: {paragraph[:80]}...")
        print()

        # --- Full paragraph as one ---
        print("  [Full paragraph - single generation]")
        t0 = time.perf_counter()
        full_audio, full_stats = engine.generate(paragraph, voice=voice)
        full_time = time.perf_counter() - t0
        full_duration = full_stats.get("duration_s", 0)
        print(f"    Total: {full_time*1000:.0f}ms | Audio: {full_duration:.1f}s | "
              f"TTFA: {full_stats['ttfa_ms']}ms | RTF: {full_stats['rtf']:.2f}x")
        save_wav(full_audio, os.path.join(run_dir, f"{voice}_full.wav"))

        # --- Sentence-by-sentence (serial) ---
        print(f"\n  [Sentence-by-sentence - {len(sentences)} chunks]")
        sentence_audios = []
        sentence_stats = []
        t0 = time.perf_counter()
        for i, sent in enumerate(sentences):
            audio, stats = engine.generate(sent, voice=voice)
            sentence_audios.append(audio)
            sentence_stats.append({**stats, "sentence": sent})
            print(f"    [{i+1}] {stats['total_ms']:>5}ms | {stats['duration_s']:.1f}s | "
                  f"TTFA: {stats['ttfa_ms']}ms | \"{sent[:50]}...\"" if len(sent) > 50 else
                  f"    [{i+1}] {stats['total_ms']:>5}ms | {stats['duration_s']:.1f}s | "
                  f"TTFA: {stats['ttfa_ms']}ms | \"{sent}\"")
        chunked_wall = time.perf_counter() - t0

        if sentence_audios:
            combined = np.concatenate(sentence_audios)
            save_wav(combined, os.path.join(run_dir, f"{voice}_chunked.wav"))
            chunked_duration = len(combined) / SAMPLE_RATE
        else:
            chunked_duration = 0

        print()
        print(f"  ┌─ Summary ({voice}) ─────────────────────────────────")
        print(f"  │ Full paragraph:   {full_time*1000:>7.0f}ms wall | {full_duration:.1f}s audio")
        print(f"  │ Serial chunked:   {chunked_wall*1000:>7.0f}ms wall | {chunked_duration:.1f}s audio")
        if full_time > 0:
            overhead = ((chunked_wall - full_time) / full_time * 100)
            print(f"  │ Chunking overhead: {overhead:+.0f}%")
            avg_ttfa = sum(s["ttfa_ms"] for s in sentence_stats) / len(sentence_stats)
            print(f"  │ Avg sentence TTFA: {avg_ttfa:.0f}ms")
            print(f"  │ First sentence playable in: {sentence_stats[0]['ttfa_ms']}ms")
        print(f"  └────────────────────────────────────────────────────")
        print()

        all_results.append({
            "voice": voice,
            "num_sentences": len(sentences),
            "full_wall_ms": round(full_time * 1000),
            "full_audio_s": full_duration,
            "serial_wall_ms": round(chunked_wall * 1000),
            "serial_audio_s": round(chunked_duration, 2),
            "sentences": sentence_stats,
        })

    report = {
        "timestamp": timestamp,
        "type": "batch_benchmark",
        "backend": engine.name,
        "model": "orpheus-3b-0.1-ft",
        "results": all_results,
    }
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 65)
    print(f"WAVs + results.json saved to: {run_dir}\n")


# ── Interactive ─────────────────────────────────────────────────────────────

def run_interactive(engine):
    print("=" * 65)
    print("  INTERACTIVE MODE")
    print(f"  Backend: {engine.name}")
    print("=" * 65)
    print()
    print("Type text and hear it. Emotion tags go inline:")
    print("  Example: Hello! <laugh> That was great. <sigh> Anyway...")
    print()
    print("Commands:")
    print("  /voice tara    - Switch voice")
    print("  /temp 0.4      - Set temperature")
    print("  /save file.wav - Save next gen to WAV")
    print("  /bench         - Re-run latency benchmark")
    print("  /batch         - Re-run batch/chunking benchmark")
    print("  /quit          - Exit")
    print("=" * 65)

    voice = DEFAULT_VOICE
    temp = DEFAULT_TEMP
    save_next = None

    while True:
        try:
            text = input(f"\n[{voice} t={temp}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text:
            continue
        if text == "/quit":
            break

        if text.startswith("/voice "):
            v = text.split()[1].lower()
            if v in VOICES:
                voice = v
                print(f"  Voice: {voice}")
            else:
                print(f"  Pick from: {', '.join(VOICES)}")
            continue

        if text.startswith("/temp "):
            try:
                temp = max(0.1, min(1.5, float(text.split()[1])))
                print(f"  Temperature: {temp}")
            except ValueError:
                print("  Bad value")
            continue

        if text.startswith("/save "):
            save_next = text.split(None, 1)[1]
            print(f"  Next gen saves to: {save_next}")
            continue

        if text == "/bench":
            run_benchmark(engine)
            continue

        if text == "/batch":
            run_batch_benchmark(engine)
            continue

        try:
            if save_next:
                audio, stats = engine.generate(text, voice=voice, temperature=temp)
                save_wav(audio, save_next)
                print(f"  Saved: {save_next} ({stats['duration_s']:.1f}s)")
                print(f"  TTFA: {stats['ttfa_ms']}ms | Total: {stats['total_ms']}ms | RTF: {stats['rtf']:.2f}x")
                save_next = None
            else:
                stats = play_streaming(engine, text, voice=voice, temperature=temp)
                print(f"  TTFA: {stats['ttfa_ms']}ms | Total: {stats['total_ms']}ms | RTF: {stats['rtf']:.2f}x")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


# ── Main ────────────────────────────────────────────────────────────────────

def check_server(url=VLLM_SERVER_URL):
    """Check if vLLM server is reachable."""
    try:
        resp = urlopen(f"{url}/health", timeout=3)
        return resp.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    print()
    print("=" * 45)
    print("  Orpheus TTS")
    print("=" * 45)
    print()

    # Detect available backends
    server_up = check_server()

    print("  Backend:")
    print(f"    A. Local llama.cpp + CUDA")
    print(f"    B. vLLM server (WSL2/Linux)  {'[ONLINE]' if server_up else '[offline]'}")
    print()

    while True:
        backend = input("  Select backend [A/B]: ").strip().upper()
        if backend in ("A", "B"):
            break
        print("  Invalid choice.")

    if backend == "B" and not server_up:
        print(f"\n  Server at {VLLM_SERVER_URL} is not reachable.")
        custom = input(f"  Enter server URL (or Enter for default): ").strip()
        if custom:
            VLLM_SERVER_URL = custom
            server_up = check_server(VLLM_SERVER_URL)
        if not server_up:
            print("  Cannot connect. Make sure tts_server.py is running.")
            raise SystemExit(1)

    print()
    print("  1. Interactive mode")
    print("  2. Latency benchmark")
    print("  3. Batch/chunking benchmark")
    print("  4. All benchmarks + Interactive")
    print("  5. Quit")
    print()

    while True:
        choice = input("  Select [1-5]: ").strip()
        if choice in ("1", "2", "3", "4", "5"):
            break
        print("  Invalid choice.")

    if choice == "5":
        raise SystemExit(0)

    print()
    if backend == "A":
        engine = LocalEngine()
    else:
        engine = ServerEngine(VLLM_SERVER_URL)

    print()

    if choice in ("2", "4"):
        run_benchmark(engine)
    if choice in ("3", "4"):
        run_batch_benchmark(engine)
    if choice in ("1", "4"):
        run_interactive(engine)
