# Amanda — Agent Guide

A real-time AI voice assistant that simulates full-duplex conversational
behaviour through intelligent pre-caching, vector-embedding search, and
multi-layered latency masking.

> **Note:** This file reflects the codebase as of the latest edit. `CLAUDE.md`
> is the original agent guide; use this file when they conflict.

## Repository layout

```
E:\Amanda
├── amanda.py                      # Tk UI + event loop + audio input callback
├── config.py                      # Paths, thresholds, ml_executor, custom_print
├── api_secrets.py                 # Gemini API key (do not commit)
├── tts_server_orpheus.py          # Local Orpheus 3B HTTP TTS server
├── tts_server_bootstrap.py        # Auto-starts/stops the Orpheus server as child process
├── tts_test.py                    # CLI exerciser for the TTS server
├── tts_cache_vectorr_management.py# Offline indexing / cache utilities for Annoy indices
├── meta_data_editor.py            # Standalone tag/metadata editor
├── last_state.json                # Persisted persona + chat log
├── CLAUDE.md                      # Original agent guide (legacy)
├── kimi.md                        # This file
├── services/
│   ├── audio_service.py           # Playback queue, pygame mixer, visualiser feed
│   ├── brain_service.py           # Gemini client, persona, mood, streaming generate
│   ├── transcription_service.py   # Silero VAD v5, dual Whisper, PANNs classification
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
- **Windows-only**: uses `win32gui`, `sounddevice`, `samplerate`
- **GPU strongly recommended** — Whisper + T5 + PANNs + Orpheus all run on CUDA

## Running

```powershell
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
```powershell
E:\Amanda\.venv\Scripts\python.exe E:\Amanda\tts_test.py
```

## TTS server lifecycle (`tts_server_bootstrap`)

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
- `get_random_from_dir(domain)` picks a random MP3 from any audio subdirectory.

### TranscriptionService (`services/transcription_service.py`)
- Dual faster-whisper: `distil-small.en` (realtime, low latency) and
  `large-v3-turbo` (accurate final turn). Both `float16` on CUDA.
- **Silero VAD v5** (CPU) for voice-activity detection. Operates on 32 ms
  frames (512 samples @ 16 kHz). Speech probability threshold is
  `VAD_SPEECH_THRESHOLD = 0.4` (0.5 is canonical; 0.4 is more permissive for
  desktop mics).
- Layered energy gates still exist as cheap safety nets:
  `MIN_FRAME_RMS = 0.003`, `MIN_CHUNK_RMS = 0.005`,
  `MIN_SPEECH_DURATION_MS = 150`.
- `.transcribe()` always passes `vad_filter=True`, `no_speech_threshold=0.6`,
  `condition_on_previous_text=False`.
- PANNs `AudioTagging` runs every 4 s over a rolling 4-second window for
  speech-presence scoring and ambient-audio tagging.
- **Trailing silence**: `MIN_BUFFER_DURATION_MS = 250` (was 100, then 350;
  250 keeps inter-sentence boundaries crisp without splitting on mid-word
  pauses with Silero).

### VectorService (`services/vector_service.py`)
- Loads FLAN-T5-Large encoder (1024-dim) + five Annoy indices
  (`tts`, `impulse`, `meta`, `interject`, `interrupt`), plus corresponding
  SQLite `_embeddings.db` files with (`filename`, `title`, `subtitle`,
  `embedding`) rows.
- `query(text, domain, ...)` encodes via `tokenizer(...)` →
  `model(**inputs)` → `last_hidden_state[0][-1]`. Errors are logged loudly
  and the query returns `[]` so the caller can fall back.
- Missing `.ann` / `.db` files are tolerated — the index starts empty and
  fills as the cache rebuilds.
- `get_random_meta_tags(n)` returns random `subtitle` values from the meta
  domain, used by impulse generation to seed emotional direction.

### TTSService (`services/tts_service.py`)
- `httpx.AsyncClient` → `POST {TTS_SERVER_URL}/tts` with
  `{"text", "voice", "format": "pcm", "max_tokens"}`.
- **Dynamic Token Sizing**: `max_tokens = len(text) * 20`. This ensures a
  generous, relative budget that prevents the "slow-speaking" issue on short
  sentences while providing enough headroom for long or expressive text.
- Streams PCM chunks into an `AudioJob` (handed to `AudioService.play_job`)
  while simultaneously accumulating bytes for post-playback MP3 cache
  write (`lameenc`, tagged via `mutagen` `TIT2`/`TIT3`).
- `generate_silence` writes a tiny silent MP3 so "learned silence" matches
  can still be indexed.
- **Priority queue**: high-priority items (default) stream immediately;
  low-priority items (cache-warmth jobs) are parked via an `asyncio.Event`
  until `set_active(False)` is called at turn end.

### BrainService (`services/brain_service.py`)
- Single `google.genai.Client` for all Gemini calls.
- `generate_stream` — streaming realtime response used by
  `amanda.process_final_turn` (sentence-by-sentence playback).
- `update_mood` — periodic mood scoring (8 emotional states → RGB blob
  colour).
- `extract_persona` — condenses persona + profile into `last_state.json`
  (guarded by `DISABLE_EXTRACT_BACKSTORY_AND_NOTES`).
- Prompt builders: `generate_interjection_prompt`, `generate_impulse_prompt`,
  `generate_interrupt_prompt`, `generate_meta_prompt`.

## Data flow

```
sounddevice input (default samplerate)
  └─► audio_input_callback (amanda.py)
       ├─► every 4s: stt.run_classification (PANNs)
       ├─► on speech score threshold: play prelude/* once per turn
       └─► stt.process_audio (Silero VAD + RMS gates)
             └─► yields chunks -> handle_realtime_chunk
                    └─► stt.transcribe(mode="realtime")   [ml_executor]
                          └─► vector.query(domain="interject") [ml_executor]
                               └─► audio.play_file  (interject/*.mp3)

on space-release:
  stop_recording → process_final_turn (asyncio)
    1. stt.get_full_audio() → stt.transcribe(mode="accurate")
    2. trigger_impulse_logic(text) → vector.query("impulse") → _dispatch_utterance
    3. brain.generate_stream(...)  [streams via google-genai]
         per sentence:
           _dispatch_utterance(sentence, "tts")  (meta tags → vector.query("meta"), then tts/cache)
    4. thinking_monitor polls for interlude/* filler while waiting for sentences
  finally: label resets to IDLE, tts.set_active(False) (guarded try/except/finally)
```

### `_dispatch_utterance` (the meta-tag pipeline)

Impulse, interrupt, and streamed TTS sentences all flow through
`_dispatch_utterance(text, domain)`:

1. Parse `*meta tags*` wrapped in asterisks.
2. Query each tag against the `meta` domain (parallel).
3. Strip meta tags from the clean text.
4. Resolve clean text via `vector.query(domain=...)` or `tts.generate(...)`.
5. Enqueue in order: **meta clips first**, then the main utterance.

This lets Gemini embed emotional stage-direction (e.g. `*sigh*`, `*gasp*`)
that maps to pre-cached SFX or short emotional clips before the TTS line.

### Sentence ordering

`sentence_processor_worker` uses a **gate chain** to preserve enqueue order
without serialising preparation:
- Sentence *N* waits on `prev_gate`, then starts its own `next_gate`, and
  only signals `prev_gate` after it has finished enqueuing audio.
- Sentence *N+1* blocks on that `next_gate`, guaranteeing serial playback
  order while letting meta-tag lookups and TTS HTTP POSTs overlap.

## Key config knobs (`config.py`)

- `TTS_SERVER_URL`, `TTS_VOICE` — `tara|leah|jess|leo|dan|mia|zac|zoe`
- `TTS_SIMILARITY_THRESHOLD = 0.75` — TTS cache hit cutoff
- `CACHE_GENERATE_SIMILARITY_CUTOFF = 0.7` — below this, a missed cache
  lookup triggers a background Gemini → TTS generation to warm the cache
- `INTERJECTION_COOLDOWN = 2.0` seconds between realtime interjections
- `MIN_BUFFER_DURATION_MS = 250` — silence tail before VAD flushes
- `VAD_SPEECH_THRESHOLD = 0.4` — Silero speech probability threshold
- `MIN_FRAME_RMS`, `MIN_CHUNK_RMS`, `MIN_SPEECH_DURATION_MS` — energy +
  duration gates layered on Silero
- `META_IMPULSE_MATCH_THRESHOLD = 0.8` / `META_SENTENCE_MATCH_THRESHOLD = 0.95`
  — meta-tag lookup cutoffs
- `GENERATE_TTS`, `GENERATE_IMPULSE_FOR_CACHE`, `GENERATE_META_FOR_CACHE`,
  `GENERATE_INTERJECT_FOR_CACHE` — feature toggles
- `DISABLE_META_TAG_VECTOR_MATCHING_FOR_IMPULSE = True` — if `True`, impulse
  text bypasses meta-tag extraction (legacy toggle)
- `PLAY_RANDOM_SOUND_WHEN_IDLE = True` — idle ambient behaviour
- `MAX_CHAT_LOG_ENTRIES = 10` — short-term context window
- `ml_executor` — shared `ThreadPoolExecutor(max_workers=3)` for Whisper /
  T5 / PANNs
- `INTERLUDE_MONITOR_INTERVAL_S = 1.5` — thinking_monitor poll cadence

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
- Windows-only (`win32gui`).
- GPU strongly recommended (CUDA used for Whisper, T5, PANNs, Orpheus).
