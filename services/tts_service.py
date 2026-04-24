import os
import io
import asyncio
import datetime
import httpx
from mutagen.mp3 import MP3
from mutagen.id3 import TIT2, TIT3
import config
from services.audio_service import AudioJob, make_mp3_encoder

_PRIO_HIGH = 0
_PRIO_LOW = 10

def _prio_label(prio):
    return "high" if prio <= _PRIO_HIGH else "low"

def _fmt_tag(folder, prio):
    return f"[{folder}/{_prio_label(prio)}]"

class TTSService:
    def __init__(self):
        config.custom_print("Lifespan", f"TTSService: HTTP client -> {config.TTS_SERVER_URL} (voice={config.TTS_VOICE}).")
        self.client = httpx.AsyncClient(timeout=120)
        self._low_priority_ok = asyncio.Event()
        self._low_priority_ok.set()

    def set_active(self, active: bool):
        if active:
            self._low_priority_ok.clear()
        else:
            self._low_priority_ok.set()

    async def abort_all(self):
        try:
            await self.client.post(f"{config.TTS_SERVER_URL}/abort")
        except Exception as e:
            config.custom_print("Error", f"TTS abort failed: {e}")

    async def generate(self, text, folder, transcribed_text=None, title=None, subtitle=None, priority="high", proximity=0.3):
        job = AudioJob(sample_rate=config.TTS_SAMPLE_RATE, event_type=folder, text=text, proximity=proximity)
        tag_title = title or transcribed_text or text
        tag_subtitle = subtitle or text
        prio = _PRIO_HIGH if priority == "high" else _PRIO_LOW
        asyncio.create_task(self._dispatch(prio, text, job, folder, tag_title, tag_subtitle))
        await asyncio.sleep(0)
        return job

    async def _dispatch(self, prio, text, job, folder, tag_title, tag_subtitle):
        if prio >= _PRIO_LOW and not self._low_priority_ok.is_set():
            config.custom_print("Stream", f"[{folder}/{_prio_label(prio)}] parked — waiting for turn end")
            await self._low_priority_ok.wait()
            config.custom_print("Stream", f"[{folder}/{_prio_label(prio)}] resumed")
        await self._fetch_stream(text, job, folder, tag_title, tag_subtitle, prio)

    async def generate_silence(self, folder, title):
        silence_bytes = int(config.TTS_SAMPLE_RATE * config.SILENCE_CACHE_DURATION_S) * 2
        silence_pcm = b'\x00' * silence_bytes
        asyncio.get_event_loop().run_in_executor(None,
            self._save_cache, silence_pcm, title, "SILENCE", folder)

    async def _fetch_stream(self, text, job, folder, tag_title, tag_subtitle, prio):
        tag = f"[{folder}/{_prio_label(prio)}]"
        max_tokens = max(300, min(len(text) * 12, 2000))
        job.estimated_duration = max_tokens / config.TTS_TOKENS_PER_SECOND
        
        try:
            config.custom_print("Stream", f"{tag} POST {text[:80]} (tokens={max_tokens} est={job.estimated_duration:.1f}s)")
            async with self.client.stream("POST", f"{config.TTS_SERVER_URL}/tts",
                                         json={
                                             "text": text, 
                                             "voice": config.TTS_VOICE, 
                                             "format": "pcm",
                                             "max_tokens": max_tokens,
                                             "min_p": config.TTS_MIN_P,
                                         }) as r:
                if r.status_code != 200:
                    config.custom_print("Error", f"TTS {tag} HTTP {r.status_code} for: {text}")
                    job.finish()
                    return
                async for chunk in r.aiter_bytes():
                    if job.cancelled: break
                    job.feed(chunk)
        except Exception as e:
            config.custom_print("Error", f"TTS Stream Error {tag}: {e}")
        finally:
            job.finish()
            if not job.cancelled and len(job.pcm_accumulator) > 0:
                asyncio.get_event_loop().run_in_executor(None,
                    self._save_cache, job.pcm_accumulator, tag_title, tag_subtitle, folder, prio)

    def _save_cache(self, pcm_data, title, subtitle, folder, prio=_PRIO_HIGH):
        try:
            enc = make_mp3_encoder()
            pcm_bytes = bytes(pcm_data) if isinstance(pcm_data, (bytearray, memoryview)) else pcm_data
            mp3_data = bytes(enc.encode(pcm_bytes)) + bytes(enc.flush())

            # Tag in-memory before writing to disk (avoids mutagen double-write)
            buf = io.BytesIO(mp3_data)
            audio = MP3(buf)
            if audio.tags is None:
                audio.add_tags()
            audio.tags.add(TIT2(encoding=3, text=title))
            audio.tags.add(TIT3(encoding=3, text=subtitle))
            audio.save(buf)
            tagged_data = buf.getvalue()

            folder_path = config.AUDIO_DIRS.get(folder, os.path.join(config.AUDIO_DIR, folder))
            os.makedirs(folder_path, exist_ok=True)

            ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            fname = os.path.join(folder_path, f"TTS_{ts}.mp3")
            with open(fname, 'wb') as f:
                f.write(tagged_data)
            config.custom_print("Cache", f"[{folder}/{_prio_label(prio)}] saved {os.path.basename(fname)} — {subtitle}")
        except Exception as e:
            config.custom_print("Error", f"Cache Save Error ({folder}): {e}")