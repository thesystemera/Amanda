import os
import queue
import threading
from collections.abc import Callable
import time
import random
import json
import numpy as np
import sounddevice as sd
import pygame
import config
from services.spatial_audio_service import SpatialAudioService
from services.room_tone_service import RoomTonePlayer

def make_mp3_encoder(sample_rate=None, bitrate=None):
    import lameenc
    sample_rate = sample_rate if sample_rate is not None else config.TTS_SAMPLE_RATE
    bitrate = bitrate if bitrate is not None else config.MP3_BITRATE
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(1)
    enc.set_quality(2)
    return enc


class AudioJob:
    def __init__(self, sample_rate=None, event_type="tts", text="", channels=1, proximity=0.3):
        self.sample_rate = sample_rate if sample_rate is not None else config.TTS_SAMPLE_RATE
        self.event_type = event_type
        self.text = text
        self.channels = channels
        self.proximity = proximity
        self.estimated_duration = None
        self.queue = queue.Queue()
        self.pcm_accumulator = bytearray()
        self.cancelled = False
        self.finished = False
        self.on_first_feed: Callable | None = None

    def feed(self, chunk: bytes):
        if not self.cancelled:
            if self.on_first_feed is not None and not self.pcm_accumulator:
                cb = self.on_first_feed
                self.on_first_feed = None
                if callable(cb):
                    cb()
            self.queue.put(chunk)
            self.pcm_accumulator.extend(chunk)

    def finish(self):
        self.finished = True
        self.queue.put(None)

    def cancel(self):
        self.cancelled = True


class AudioService:
    def __init__(self):
        config.custom_print("Lifespan", "AudioService: initialising pygame mixer @ 24kHz mono...")
        self.playback_queue = queue.Queue()
        self.visualizer_queue = queue.Queue(maxsize=100)
        self.is_playing = threading.Event()
        self.current_volume = 0.0
        self.interrupted = False
        self.last_audio_time = time.time()
        self.current_playing_text = ""

        self.manifest_cache = {}
        self.recently_played = {}

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=config.TTS_SAMPLE_RATE, size=-16, channels=1)

        self.spatial = SpatialAudioService()
        self.last_voice_proximity = config.DEFAULT_MIC_PROXIMITY

        if not os.path.exists(config.ROOM_TONE_PATH):
            config.custom_print("Lifespan", f"WARNING: Room tone not found at {config.ROOM_TONE_PATH}")
        self.room_tone = RoomTonePlayer(config.ROOM_TONE_PATH, sample_rate=config.TTS_SAMPLE_RATE, volume_db=-42)
        self.room_tone.start()

        self._output_stream = None
        self._init_output_stream()

        if config.SPATIAL_AUDIO_ENABLED:
            from services.spatial_audio_service import StreamingSpatialProcessor
            self.global_spatial = StreamingSpatialProcessor(
                config.TTS_SAMPLE_RATE,
                speaker=config.TTS_VOICE,
                start_mix=config.DEFAULT_MIC_PROXIMITY,
            )
        else:
            self.global_spatial = None

        self._init_manifests()
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True, name="audio-playback")
        self._playback_thread.start()
        config.custom_print("Lifespan", f"AudioService: ready (global stereo output @ {config.TTS_SAMPLE_RATE}Hz, spatial={config.SPATIAL_AUDIO_ENABLED}).")

    def _init_manifests(self):
        for name in config.RANDOM_AUDIO_SUBDIRS:
            directory = config.AUDIO_DIRS.get(name)
            if directory and os.path.exists(directory):
                self._load_or_create_manifest(directory)

    def _load_or_create_manifest(self, directory):
        dir_name = os.path.basename(os.path.normpath(directory))
        manifest_file = os.path.join(directory, f'small_mp3_manifest_{dir_name}.json')

        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, 'r') as f:
                    files = json.load(f)
                self.manifest_cache[directory] = files
                config.custom_print("Cache", f"Loaded random pool manifest for {dir_name} ({len(files)} files)")
            except Exception as e:
                config.custom_print("Error", f"Failed to load manifest for {dir_name}: {e}")
                self._create_manifest(directory, manifest_file)
        else:
            self._create_manifest(directory, manifest_file)

    def _create_manifest(self, directory, manifest_file):
        dir_name = os.path.basename(os.path.normpath(directory))
        try:
            files = [f for f in os.listdir(directory) if f.endswith('.mp3')]
            with open(manifest_file, 'w') as f:
                json.dump(files, f)
            self.manifest_cache[directory] = files
            config.custom_print("Cache", f"Created manifest for {dir_name} ({len(files)} files)")
        except Exception as e:
            config.custom_print("Error", f"Failed to create manifest for {dir_name}: {e}")

    def play_file(self, file_path, event_type="tts", text="", proximity=None):
        detail = f'"{text}"' if text else os.path.basename(file_path)
        config.custom_print("Play", f"[{event_type}/queue] {detail}")
        if proximity is None:
            proximity = self.last_voice_proximity if event_type in ("breath", "prelude", "interlude") else config.DEFAULT_MIC_PROXIMITY
        if event_type in ("tts", "impulse", "interject", "interrupt", "meta"):
            self.last_voice_proximity = proximity
        self.playback_queue.put((file_path, event_type, text, proximity))

    def play_job(self, job: AudioJob):
        detail = f'"{job.text}"' if job.text else "[stream]"
        config.custom_print("Play", f"[{job.event_type}/queue] {detail}")
        if job.event_type in ("tts", "impulse", "interject", "interrupt", "meta"):
            self.last_voice_proximity = job.proximity
        self.playback_queue.put((job, job.event_type, job.text, job.proximity))

    def interrupt(self):
        self.interrupted = True
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
            except queue.Empty:
                break

    def get_random_from_dir(self, directory):
        if not os.path.exists(directory):
            return None

        dir_name = os.path.basename(os.path.normpath(directory))
        if dir_name in config.RANDOM_AUDIO_SUBDIRS:
            if directory not in self.manifest_cache:
                self._load_or_create_manifest(directory)
            files = self.manifest_cache.get(directory, [])
        else:
            files = [f for f in os.listdir(directory) if f.endswith('.mp3')]

        if not files:
            return None

        current_time = time.time()
        available = [f for f in files if os.path.join(directory, f) not in self.recently_played or
                     (current_time - self.recently_played[os.path.join(directory, f)]) >= config.COOLDOWN_DURATION]

        if not available:
            available = files

        chosen = random.choice(available)
        full_path = os.path.join(directory, chosen)
        self.recently_played[full_path] = current_time
        return full_path

    def _playback_worker(self):
        last_event_type = None
        while True:
            item, event_type, text, proximity = self.playback_queue.get()
            self.interrupted = False
            self.current_playing_text = text

            if text:
                log_detail = f'"{text}"'
            elif isinstance(item, str):
                log_detail = os.path.basename(item)
            else:
                log_detail = "[stream]"
            config.custom_print("Play", f"[{event_type}/play] {log_detail}")

            next_proximity = proximity
            if not self.playback_queue.empty():
                try:
                    next_item = self.playback_queue.queue[0]
                    next_proximity = next_item[3]
                except (IndexError, Exception):
                    pass

            sources = []

            if event_type == "tts" and last_event_type == "tts":
                breath = self.get_random_from_dir(config.AUDIO_DIRS['breath'])
                if breath:
                    config.custom_print("Play", f"[breath/random] {os.path.basename(breath)}")
                    pcm, sr = self._decode_mp3(breath)
                    if pcm:
                        breath_job = AudioJob(sample_rate=sr, event_type="breath", text="", channels=1, proximity=self.last_voice_proximity)
                        breath_job.estimated_duration = len(pcm) / (sr * 2)
                        for i in range(0, len(pcm), config.AUDIO_CHUNK_SIZE):
                            breath_job.feed(pcm[i:i + config.AUDIO_CHUNK_SIZE])
                        breath_job.finish()
                        sources.append((breath_job, self.last_voice_proximity))

            if isinstance(item, str):
                pcm, sr = self._decode_mp3(item)
                if pcm:
                    job = AudioJob(sample_rate=sr, event_type=event_type, text=text, channels=1, proximity=proximity)
                    job.estimated_duration = len(pcm) / (sr * 2)
                    for i in range(0, len(pcm), config.AUDIO_CHUNK_SIZE):
                        job.feed(pcm[i:i + config.AUDIO_CHUNK_SIZE])
                    job.finish()
                    sources.append((job, proximity))
            elif isinstance(item, AudioJob):
                sources.append((item, proximity))

            if sources:
                self.is_playing.set()
                self.room_tone.speech_active()
                for i, (src, src_proximity) in enumerate(sources):
                    if self.global_spatial:
                        next_mix = sources[i + 1][1] if i + 1 < len(sources) else next_proximity
                        self.global_spatial.set_target_mix(
                            src_proximity,
                            next_mix=next_mix,
                            duration_hint=src.estimated_duration,
                        )
                    self._play_job_chunks(src)
                self.is_playing.clear()
                self.room_tone.speech_inactive()
                self.current_volume = 0.0
                self.current_playing_text = ""

            last_event_type = event_type
            self.playback_queue.task_done()

    def _play_job_chunks(self, job):
        pending_odd = b""
        while True:
            if self.interrupted:
                job.cancel()
                break
            try:
                chunk = job.queue.get(timeout=0.1)
            except queue.Empty:
                if job.finished:
                    break
                continue
            if chunk is None:
                break

            raw = pending_odd + chunk
            if len(raw) % 2:
                pending_odd = raw[-1:]
                raw = raw[:-1]
            else:
                pending_odd = b""

            if raw:
                proc = self.global_spatial.process(raw)
                self._write_output(proc)
                self._update_visualizer(proc)

    def _write_output(self, buf):
        try:
            self._output_stream.write(buf)
        except Exception as e:
            config.custom_print("Error", f"AudioService: output stream write failed: {e}")

    def _update_visualizer(self, buf):
        try:
            self.visualizer_queue.put_nowait(buf)
        except queue.Full:
            pass
        audio_array = np.frombuffer(buf, dtype=np.int16)
        self.current_volume = float(np.abs(audio_array).mean() / 327.68)
        self.last_audio_time = time.time()

    def _init_output_stream(self):
        try:
            self._output_stream = sd.RawOutputStream(
                samplerate=config.TTS_SAMPLE_RATE,
                channels=2,
                dtype='int16',
                latency=0.05,
            )
            self._output_stream.start()
        except Exception as e:
            config.custom_print("Error", f"AudioService: Could not open output stream: {e}")

    @staticmethod
    def _decode_mp3(path):
        try:
            sound = pygame.mixer.Sound(path)
            raw = pygame.sndarray.array(sound)
            if len(raw.shape) > 1:
                raw = raw.mean(axis=1).astype(np.int16)
            return raw.tobytes(), config.TTS_SAMPLE_RATE
        except Exception as e:
            config.custom_print("Error", f"AudioService._decode_mp3 failed for {path}: {e}")
            return None, None