# Amanda

A real-time AI voice assistant that simulates full-duplex conversational
behavior through intelligent pre-caching, vector-embedding search, and
multi-layered latency masking.

Note: PLAiR (Personalised Localised Adaptive Interactive Radio) is a separate
project that shares some of Amanda's tech. They are not the same project —
PLAiR lives at `E:\AI_RADIO`.

## Repository layout

```
E:\Amanda
├── amanda.py                      # Tk UI + event loop + audio input callback
├── config.py                      # Paths, thresholds, ml_executor, custom_print
├── api_secrets.py                 # Gemini API key (do not commit)
├── tts_server_orpheus.py          # Local Orpheus 3B HTTP TTS server (lives here now, was E:\TTS_Local)
├── tts_server_bootstrap.py        # Auto-starts/stops the Orpheus server as amanda's child process
├── tts_test.py                    # CLI exerciser for the TTS server
├── tts_cache_vectorr_management.py# Offline indexing / cache utilities for the Annoy indices
├── meta_data_editor.py            # Standalone tag/metadata editor
├── last_state.json                # Persisted persona + chat log
├── services/
│   ├── audio_service.py           # Playback queue, pygame mixer, visualiser feed
│   ├── brain_service.py           # Gemini client, persona, mood, streaming generate
│   ├── transcription_service.py   # VAD, dual Whisper pipeline, PANNs audio classification
│   ├── tts_service.py             # HTTP client to tts_server_orpheus + MP3 cache writer
│   └── vector_service.py          # T5 encoder + Annoy index loaders + query
└── data/
    ├── audio/{tts,impulse,meta,interject,interrupt,prelude,breath,interlude}/
    └── vector/{tts,impulse,meta,interject,interrupt}_embeddings.{db,ann}
```

The old Flask web app / Spotify DJ integration was removed 2026-04-18. On
2026-04-19 the `E:\TTS_Local` project was merged into `E:\Amanda` and the
shared venv was rebuilt on Python 3.12.

## Runtime requirements

- **Python**: 3.12 (venv lives at `E:\Amanda\.venv`)
- **Torch**: 2.6.0+cu124
- **NumPy**: 2.4.3
- **Transformers**: 5.5.4  (`T5Tokenizer` / `T5EncoderModel` only — the
  unused `lm_head.weight | UNEXPECTED` load warning on `T5EncoderModel` is
  harmless and safe to ignore)
- **faster-whisper** (CTranslate2 backend, float16 on CUDA)
- **llama-cpp-python + orpheus-cpp** for the TTS server (GGUF + CUDA build)
- **Windows-only**: uses `win32gui`, `sounddevice`, `samplerate`, `webrtcvad`
- **GPU strongly recommended** — Whisper + T5 + PANNs + Orpheus all run on CUDA

## Running

```
# One terminal is enough — amanda.py auto-launches and health-checks the
# TTS server via tts_server_bootstrap. No manual server start required.
E:\Amanda\.venv\Scripts\python.exe E:\Amanda\amanda.py
```

Expected startup sequence (watch for the `Lifespan:` lines):

```
Lifespan: TTS bootstrap launching Orpheus subprocess...
Lifespan: VectorService: loading T5 encoder (google/flan-t5-large) on cuda...
Lifespan: VectorService: ready (N indexed items across 5 domains).
Lifespan: AudioService: initialising pygame mixer @ 24kHz mono...
Lifespan: AudioService: ready.
Lifespan: TTSService: HTTP client -> http://localhost:8080 (voice=tara).
Lifespan: BrainService: Gemini client for model gemini-3.1-flash-lite-preview.
Lifespan: BrainService: ready.
Lifespan: TranscriptionService: loading Whisper + PANNs on CUDA...
Lifespan: TranscriptionService: ready.
Lifespan: TTS server is ready (bootstrap /health ok).
```

If any `Lifespan:` line is missing the corresponding service did not come up
cleanly — scan for a preceding `Error:` line.

Standalone TTS server test (server only, skips the app):
```
E:\Amanda\.venv\Scripts\python.exe E:\Amanda\tts_test.py
```

## TTS server lifecycle (tts_server_bootstrap)

`amanda.py` imports `tts_server_bootstrap` and calls `start()` *before* any
service is constructed. Bootstrap behaviour:

1. `start()` — frees port 8080 (via `netstat`/`taskkill` on any stale PID),
   then `Popen`s `tts_server_orpheus.py` using `sys.executable` (the amanda
   venv), and spawns a daemon watcher thread that polls `/health` until it
   returns `{"status":"ok"}` or times out at 180s.
2. `wait_ready(timeout)` — `amanda.py.run()` blocks on this just before it
   starts the asyncio loop, so we never accept keypresses before the server
   is serving.
3. `atexit` registers `stop()` — terminates the subprocess (SIGTERM with a
   5-second grace, then kill). A crash of `amanda.py` still takes the server
   down.

The server itself is a single-worker, FIFO-queue Flask app. One Orpheus engine,
one thread pulling `Job`s, MP3 or PCM streamed back per-HTTP-request. Max
queue depth is 16 — overflow returns HTTP 503.

## Services

All services are instantiated once in `AmandaApp.__init__` and share the
asyncio event loop created in `AmandaApp.run`. Heavy CUDA work
(Whisper `transcribe`, PANNs `inference`, T5 `forward`) goes through
`config.ml_executor` (a dedicated `ThreadPoolExecutor(max_workers=3)`) so a
backlog of realtime chunks can't starve the default executor that
`brain_service`, `tts_service`, and `audio_service` rely on.

### AudioService (`services/audio_service.py`)
- `playback_queue` (asyncio) + single `start_worker` consumer.
- Decodes MP3s via pygame, streams PCM via `sounddevice.RawOutputStream` at
  24 kHz mono int16.
- Interrupt handling: `interrupt()` drains the queue and flips `interrupted`
  so the currently-streaming job aborts.
- Also writes into `visualizer_queue` for the pygame visualiser thread.

### TranscriptionService (`services/transcription_service.py`)
- Dual faster-whisper: `distil-small.en` (realtime, low latency) and
  `large-v3-turbo` (accurate final turn). Both `float16` on CUDA.
- 30 ms VAD frames via `webrtcvad` aggressiveness 3, with additional
  per-frame + per-chunk RMS gates (`MIN_FRAME_RMS`, `MIN_CHUNK_RMS`) and a
  minimum accumulated speech duration (`MIN_SPEECH_DURATION_MS`) before a
  chunk is flushed to the realtime model. This is what keeps
  distil-small.en from hallucinating YouTube-caption phrases on tiny
  near-silent chunks.
- `.transcribe()` always passes `vad_filter=True`, `no_speech_threshold=0.6`,
  `condition_on_previous_text=False`.
- PANNs `AudioTagging` runs every 4 s over a rolling 4-second window for
  speech-presence scoring and ambient-audio tagging.

### VectorService (`services/vector_service.py`)
- Loads FLAN-T5-Large encoder (1024-dim) + five Annoy indices
  (`tts`, `impulse`, `meta`, `interrupt`, `interject`), plus corresponding
  SQLite `_embeddings.db` files with (`filename`, `title`, `subtitle`,
  `embedding`) rows.
- `query(text, domain, ...)` encodes via `tokenizer(...)` →
  `model(**inputs)` → `last_hidden_state[0][-1]`. Errors are logged loudly
  and the query returns `[]` so the caller can fall back.
- Missing `.ann` / `.db` files are tolerated — the index starts empty and
  fills as the cache rebuilds.

### TTSService (`services/tts_service.py`)
- `httpx.AsyncClient` → `POST {TTS_SERVER_URL}/tts` with
  `{"text", "voice", "format": "pcm", "max_tokens"}`.
- **Dynamic Token Sizing**: `max_tokens` is calculated dynamically as `len(text) * 20`. This ensures a generous, relative budget that prevents the "slow-speaking" issue on short sentences while providing enough headroom for long or expressive text without any arbitrary floor or ceiling.
- Streams PCM chunks into an `AudioJob` (handed to `AudioService.play_job`)
  while simultaneously accumulating bytes for post-playback MP3 cache
  write (`lameenc`, tagged via `mutagen` `TIT2`/`TIT3`).
- `generate_silence` writes a tiny silent MP3 so "learned silence" matches
  can still be indexed.

### BrainService (`services/brain_service.py`)
- Single `google.genai.Client` for all Gemini calls.
- `generate_stream` — streaming realtime response used by
  `amanda.process_final_turn` (sentence-by-sentence playback).
- `update_mood` — periodic mood scoring (8 emotional states → RGB blob
  colour).
- `extract_persona` — every 5 turns, condenses persona + profile into
  `last_state.json`.

## Turn flow & impulse architecture

The goal is zero-latency audio feedback while still giving Gemini the full
accurate context.

```
Space pressed
  └─► VAD accumulates frames → flush on silence → fast Whisper chunks
       └─► appended to realtime_text_parts

Space released
  └─► process_final_turn
       1. Build impulse text from realtime_text_parts (fast Whisper chunks)
       2. Fire impulse immediately (cached/generated TTS plays in parallel)
       3. Run accurate Whisper on full raw audio (parallel to impulse)
       4. Wait for impulse dispatch to complete
       5. Log USER accurate transcript FIRST
       6. Log ASSISTANT impulse subtitle SECOND
       7. Call Gemini with impulse_subtitle injected into the current turn
```

Gemini sees the impulse twice: once in the chat history (logged at step 6) and
once inline in the current turn prompt (`[User] ...\n[Assistant] {impulse}`).
This lets Amanda grammatically continue her own thought thread rather than
starting a brand-new response.

### Continuous-speech fallback

The VAD only emits a chunk when it detects a silence tail (`MIN_BUFFER_DURATION_MS`).
If the user speaks continuously and releases space bar immediately, the VAD buffer
is never flushed → `realtime_text_parts` is empty.

When this happens, `process_final_turn` falls back to running **fast Whisper on
the full raw audio** before firing the impulse. This adds ~100–300 ms of latency
(compared to instant when chunks exist) but guarantees an impulse even for
unbroken speech. Accurate Whisper still runs in parallel afterwards.

Trade-off: VAD-chunked realtime can occasionally miss the final tail word/phrase
(because it was still buffering when space was released). The fallback covers the
worst-case empty buffer; for partial buffers the tail is simply absent from the
impulse text. This is acceptable because the impulse is only a filler — Gemini's
actual response is always generated from the full accurate transcript.

## Data flow

```
sounddevice input (default samplerate)
  └─► audio_input_callback (amanda.py)
       ├─► every 4s: stt.run_classification (PANNs)
       ├─► on speech score threshold: play prelude/* once per turn
       └─► stt.process_audio (VAD + RMS + duration gates)
             └─► yields chunks -> handle_realtime_chunk
                    └─► stt.transcribe(mode="realtime")   [ml_executor]
                          └─► vector.query(domain="interject") [ml_executor]
                               └─► audio.play_file  (interject/*.mp3)

on space-release:
  stop_recording → process_final_turn (asyncio)
    1. stt.get_full_audio() → fast fallback OR realtime_text_parts → impulse
    2. stt.get_full_audio() → stt.transcribe(mode="accurate")  (parallel)
    3. wait impulse gate → log user → log impulse → brain.generate_stream(...)
         per sentence:
           vector.query(domain="tts") or tts.generate → audio.play_job
  finally: label resets to IDLE, awaiting_response=False (guarded try/except/finally)
```

## Key config knobs (`config.py`)

- `TTS_SERVER_URL`, `TTS_VOICE` — `tara|leah|jess|leo|dan|mia|zac|zoe`
- `TTS_SIMILARITY_THRESHOLD = 0.95` — TTS cache hit cutoff
- `INTERJECTION_COOLDOWN = 4.0` seconds between realtime interjections
- `MIN_BUFFER_DURATION_MS = 350` — silence tail before VAD flushes (was 100;
  100 caused flushes on every breath/click)
- `MIN_FRAME_RMS`, `MIN_CHUNK_RMS`, `MIN_SPEECH_DURATION_MS` — energy +
  duration gates layered on webrtcvad to eliminate Whisper hallucinations
- `GENERATE_TTS`, `GENERATE_IMPULSE_FOR_CACHE`, `GENERATE_META_FOR_CACHE`,
  `GENERATE_INTERJECT_FOR_CACHE` — feature toggles
- `MAX_CHAT_LOG_ENTRIES = 20` — context window size
- `ml_executor` — shared `ThreadPoolExecutor(max_workers=3)` for Whisper /
  T5 / PANNs

## Performance reference

- Orpheus on RTX 6000 Turing: ~350 ms TTFA, ~1.22× RTF
- Inline emotion tags Orpheus supports:
  `<laugh> <chuckle> <sigh> <gasp> <cough> <sniffle> <groan> <yawn>`

## Cache / index state

All five caches (`tts`, `impulse`, `meta`, `interrupt`, `interject`) were
purged during the Orpheus migration (2026-04-18). They rebuild automatically
as the app runs. Pre-seed via `tts_cache_vectorr_management.py` if needed.
Missing `.ann` / `.db` files are tolerated — services start empty.

## Security

- `api_secrets.GEMINI_API_KEY` — do not commit. Rotate periodically.
- Windows-only (win32gui).
- GPU strongly recommended (CUDA used for Whisper, T5, PANNs, Orpheus).
