# Amanda

A real-time AI voice assistant that simulates full-duplex conversational
behaviour through intelligent pre-caching, vector-embedding search, and
multi-layered latency masking.

**Repository:** https://github.com/thesystemera/Amanda

Note: PLAiR (Personalised Localised Adaptive Interactive Radio) is a separate
project that shares some of Amanda's tech. They are not the same project —
PLAiR lives at `E:\AI_RADIO`.

## Repository layout

```
E:\Amanda
├── amanda.py                      # Tk UI + event loop + audio input callback
├── config.py                      # Paths, thresholds, ml_executor, custom_print
├── api_secrets.py                 # Loads GEMINI_API_KEY from env or keys/ (gitignored)
├── api_secrets.py.example         # Template showing the loader pattern
├── requirements.txt               # Core Python dependencies
├── tts_server_orpheus.py          # Local Orpheus 3B HTTP TTS server (lives here now, was E:\TTS_Local)
├── tts_server_bootstrap.py        # Auto-starts/stops the Orpheus server as amanda's child process
├── tts_test.py                    # CLI exerciser for the TTS server
├── tts_cache_vectorr_management.py# Offline indexing / cache utilities for the Annoy indices
├── meta_data_editor.py            # Standalone PyQt5 tag/metadata editor (Gemini backend)
├── last_state.json                # Persisted persona + chat log
├── LOCAL_LLM_GUIDE.md             # P6000 (24GB VRAM) local inference reference
├── services/
│   ├── audio_service.py           # Playback queue, pygame mixer, global spatial processor, visualiser feed
│   ├── brain_service.py           # Persona, mood, streaming generate, prompt builders
│   ├── gemini_service.py          # google-genai client wrapper + warm-up pools
│   ├── panns_service.py           # PANNs AudioTagging inference worker
│   ├── room_tone_service.py       # Continuous background room-tone loop
│   ├── spatial_audio_service.py   # StreamingSpatialProcessor + SpatialAudioService (pedalboard pipeline)
│   ├── transcription_service.py   # Silero VAD v5, dual Whisper pipeline, chunk handler
│   ├── tts_service.py             # HTTP client to tts_server_orpheus + MP3 cache writer
│   ├── vector_service.py          # Sentence-transformers encoder + Annoy index loaders + query
│   └── whisper_service.py         # faster-whisper model loader + threaded inference queue
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
- **pedalboard + noise** for real-time spatial audio effects
- **PyQt5** for `meta_data_editor.py`
- **google-genai** for Gemini client (`brain_service`, `gemini_service`, `meta_data_editor`)
- **Windows-only**: uses `win32gui`, `sounddevice`, `samplerate`
- **GPU strongly recommended** — Whisper + T5 + PANNs + Orpheus all run on CUDA

Install dependencies via:
```
E:\Amanda\.venv\Scripts\pip.exe install -r E:\Amanda\requirements.txt
```

## Running

```
# One terminal is enough — amanda.py auto-launches and health-checks the
# TTS server via tts_server_bootstrap. No manual server start required.
E:\Amanda\.venv\Scripts\python.exe E:\Amanda\amanda.py
```

Expected startup sequence (watch for the `Lifespan:` lines):

```
Lifespan: VectorService: loading sentence-transformers/all-MiniLM-L6-v2 on cuda...
Lifespan: VectorService: ready (N indexed items across 5 domains).
Lifespan: AudioService: initialising pygame mixer @ 24kHz mono...
Lifespan: AudioService: ready (global stereo output @ 24000Hz, spatial=True).
Lifespan: TTSService: HTTP client -> http://localhost:8080 (voice=tara).
Lifespan: GeminiService: initializing client...
Lifespan: BrainService: ready.
Lifespan: WhisperService: loading distil-small.en + large-v3-turbo on CUDA...
Lifespan: WhisperService: ready.
Lifespan: TranscriptionService: loading Silero VAD v5 on cuda...
Lifespan: TranscriptionService: ready.
Lifespan: PANNsService: loading AudioTagging on CUDA...
Lifespan: PANNsService: ready.
Lifespan: GeminiService: warming up model pools...
Lifespan: GeminiService: ready (Xms).
Lifespan: TTS server is ready (bootstrap /health ok).
```

If any `Lifespan:` line is missing the corresponding service did not come up
cleanly — scan for a preceding `Error:` line.

Standalone TTS server test (server only, skips the app):
```
E:\Amanda\.venv\Scripts\python.exe E:\Amanda\tts_test.py
```

Standalone metadata editor:
```
E:\Amanda\.venv\Scripts\python.exe E:\Amanda\meta_data_editor.py
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

The server itself is a multi-worker Flask app with a FIFO job queue. Two
Orpheus engines (`TTS_NUM_WORKERS=2`), one thread pulling `Job`s per engine,
MP3 or PCM streamed back per-HTTP-request. Max queue depth is 16 — overflow
returns HTTP 503.

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
  24 kHz **stereo** int16 (global spatial processor outputs stereo).
- **Global spatial processor**: `StreamingSpatialProcessor` lives inside
  `AudioService` and processes *all* TTS/job audio through a single stateful
  pipeline so reverb tails and compressor envelopes carry across utterances.
- Interrupt handling: `interrupt()` drains the queue and flips `interrupted`
  so the currently-streaming job aborts.
- Also writes into `visualizer_queue` for the pygame visualiser thread.

### SpatialAudioService / StreamingSpatialProcessor (`services/spatial_audio_service.py`)

Two classes:

- **`StreamingSpatialProcessor`** — stateful chunk-by-chunk processor used for
  live TTS streams. Maintains reverb tail and compressor envelope across
  chunks with **zero buffering / zero latency penalty**.
- **`SpatialAudioService`** — offline clip processor (200 ms segments + Perlin
  noise + crossfades). Used for file-based playback when a global stream
  processor is not appropriate.

**Parallel dual-board architecture (StreamingSpatialProcessor)**

Instead of a single Pedalboard per chunk, the processor runs two boards in
parallel and mixes their outputs in numpy:

- **`dry_board`** — mic/preamp path: `Compressor -> HighShelfFilter -> LowShelfFilter`.
  Parameters are proximity-morphed (close-talk vs. far) every chunk.
- **`wet_board`** — room ambience: `Reverb` with `dry_level=0.0` (pure wet).
  Receives the raw panned signal. Wet level stays constant; perceived
  wetness increases only because the dry path attenuates with distance.

Both boards are called with **`reset=False`** so internal delay lines and
envelopes persist across chunks. This fixes the previous tail-drop bug where
Pedalboard's default `reset=True` was zeroing the Reverb state every chunk.

Mix formula:
```
dry_gain = dry_near + (dry_far - dry_near) * effective_mix
proc = dry * dry_gain + wet
```

Perlin noise drives sub-sonic pan and mix jitter so the image breathes
organically.

### TranscriptionService (`services/transcription_service.py`)
- Dual faster-whisper: `distil-small.en` (realtime, low latency) and
  `large-v3-turbo` (accurate final turn). Both `float16` on CUDA.
- **Silero VAD v5** (CPU/GPU) for voice-activity detection. Operates on 32 ms
  frames (512 samples @ 16 kHz). Speech probability threshold is
  `VAD_SPEECH_THRESHOLD = 0.4`.
- Layered energy gates still exist as cheap safety nets:
  `MIN_FRAME_RMS = 0.003`, `MIN_CHUNK_RMS = 0.005`,
  `MIN_SPEECH_DURATION_MS = 150`.
- `.transcribe()` always passes `vad_filter=True`, `no_speech_threshold=0.6`,
  `condition_on_previous_text=False`.
- PANNs `AudioTagging` runs every 4 s over a rolling 4-second window for
  speech-presence scoring and ambient-audio tagging.
- **Trailing silence**: `MIN_BUFFER_DURATION_MS = 250` — silence tail before
  VAD flushes (keeps inter-sentence boundaries crisp without splitting on
  mid-word pauses).

### WhisperService (`services/whisper_service.py`)
Dedicated threaded queue wrapper around the two faster-whisper models.
Decouples model loading from `TranscriptionService` so each can be started
and torn down independently.

### PANNsService (`services/panns_service.py`)
Threaded inference worker for `panns_inference.AudioTagging`. Maintains a
rolling 4-second audio buffer resampled to 16 kHz and emits top-k labels
with confidence scores.

### GeminiService (`services/gemini_service.py`)
Thin wrapper around `google.genai.Client`. On startup it warms both the task
and chat model pools with a 1-token dummy generation so the first real
request avoids cold-start latency.

### VectorService (`services/vector_service.py`)
- Loads `sentence-transformers/all-MiniLM-L6-v2` encoder (384-dim) + five Annoy
  indices (`tts`, `impulse`, `meta`, `interject`, `interrupt`), plus
  corresponding SQLite `_embeddings.db` files with (`filename`, `title`,
  `subtitle`, `embedding`) rows.
- `query(text, domain, ...)` encodes via `tokenizer(...)` ->
  `model(**inputs)` -> mean-pooled hidden state. Errors are logged loudly
  and the query returns `[]` so the caller can fall back.
- Missing `.ann` / `.db` files are tolerated — the index starts empty and
  fills as the cache rebuilds.

### TTSService (`services/tts_service.py`)
- `httpx.AsyncClient` -> `POST {TTS_SERVER_URL}/tts` with
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
- Uses `GeminiService` client for all Gemini calls.
- `generate_stream` — streaming realtime response used by
  `amanda.process_final_turn` (sentence-by-sentence playback).
- `update_mood` — periodic mood scoring (8 emotional states -> RGB blob
  colour).
- `extract_persona` — condenses persona + profile into `last_state.json`
  (guarded by `DISABLE_EXTRACT_BACKSTORY_AND_NOTES`).
- Prompt builders: `generate_interjection_prompt`, `generate_impulse_prompt`,
  `generate_interrupt_prompt`, `generate_meta_prompt`.

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
is never flushed -> `realtime_text_parts` is empty.

When this happens, `process_final_turn` falls back to running **fast Whisper on
the full raw audio** before firing the impulse. This adds ~100–300 ms of latency
(compared to instant when chunks exist) but guarantees an impulse even for
unbroken speech. Accurate Whisper still runs in parallel afterwards.

Trade-off: VAD-chunked realtime can occasionally miss the final tail word/phrase
(because it was still buffering when space was released). The fallback covers the
worst-case empty buffer; for partial buffers the tail is simply absent from the
impulse text. This is acceptable because the impulse is only a filler — Gemini's
actual response is always generated from the full accurate transcript.

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

## Data flow

```
sounddevice input (default samplerate)
  └─► audio_input_callback (amanda.py)
       ├─► every 4s: panns.run_classification (PANNsService)
       ├─► on speech score threshold: play prelude/* once per turn
       └─► stt.process_audio (Silero VAD + RMS gates)
             └─► yields chunks -> handle_realtime_chunk
                    └─► whisper.transcribe(mode="realtime")   [ml_executor]
                          └─► vector.query(domain="interject") [ml_executor]
                               └─► audio.play_file  (interject/*.mp3)

on space-release:
  stop_recording -> process_final_turn (asyncio)
    1. stt.get_full_audio() -> fast fallback OR realtime_text_parts -> impulse
    2. stt.get_full_audio() -> whisper.transcribe(mode="accurate")  (parallel)
    3. wait impulse gate -> log user -> log impulse -> brain.generate_stream(...)
         per sentence:
           _dispatch_utterance(sentence, "tts")  (meta tags -> vector.query("meta"), then tts/cache)
    4. thinking_monitor polls for interlude/* filler while waiting for sentences
  finally: label resets to IDLE, tts.set_active(False) (guarded try/except/finally)
```

## Key config knobs (`config.py`)

- `TTS_SERVER_URL`, `TTS_VOICE` — `tara|leah|jess|leo|dan|mia|zac|zoe`
- `TTS_SIMILARITY_THRESHOLD = 0.75` — TTS cache hit cutoff
- `CACHE_SIMILARITY_THRESHOLD = 0.60` — below this, a missed cache lookup
  triggers a background Gemini -> TTS generation to warm the cache
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
- `MAX_CHAT_LOG_ENTRIES = 5` — short-term context window
- `ml_executor` — shared `ThreadPoolExecutor(max_workers=3)` for Whisper /
  T5 / PANNs
- `INTERLUDE_MONITOR_INTERVAL_S = 1.5` — thinking_monitor poll cadence
- `SPATIAL_AUDIO_ENABLED` — master toggle for the global spatial pipeline

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

- `api_secrets.py` is **gitignored**. Do not commit it.
- `GEMINI_API_KEY` is loaded from (in priority order):
  1. `GEMINI_API_KEY` environment variable
  2. `keys/gemini_key.txt`
  3. `keys/api_secrets.txt` (legacy fallback)
- The committed template is `api_secrets.py.example`.
- Rotate keys periodically.
- Windows-only (`win32gui`).
- GPU strongly recommended (CUDA used for Whisper, T5, PANNs, Orpheus).
