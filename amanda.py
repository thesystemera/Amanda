import asyncio
import threading
import queue
import collections
import sounddevice as sd
import numpy as np
import config
import time
import re
import os
import pygame
from scipy import interpolate

import tts_server_bootstrap

from services.vector_service import VectorService
from services.audio_service import AudioService
from services.tts_service import TTSService
from services.gemini_service import GeminiService
from services.brain_service import BrainService
from services.transcription_service import TranscriptionService
from services.whisper_service import WhisperService
from services.panns_service import PANNsService

class AmandaApp:
    def __init__(self):
        config.custom_print("Info", "Initializing Amanda Production Build...")
        self.vector = VectorService()
        self.audio = AudioService()
        self.tts = TTSService()
        self.gemini = GeminiService()
        self.brain = BrainService(gemini_service=self.gemini)
        self.whisper = WhisperService()
        self.stt = TranscriptionService(whisper_service=self.whisper)
        self.panns = PANNsService()

        self.recording = False
        self.loop = None

        self.awaiting_response = False
        self.last_impulse_text = ""
        self.impulse_subtitle = ""
        self.turn_impulse_played = False
        self.last_interjection_time = 0
        self.realtime_text_parts = []

        self.sentence_queue = asyncio.Queue()
        self._sentence_final_gate = None
        self._turn_start_gate = None
        self.turn_perf = {}
        self.cache_stats = {d: {"hits": 0, "misses": 0} for d in config.DOMAINS}
        self.turn_cache_stats = {d: {"hits": 0, "misses": 0} for d in config.DOMAINS}

        self.ui_status = "IDLE"
        self.ui_status_color = (255, 255, 255)
        self.ui_transcript = []
        self.ui_panns_tags = ""
        self._ui_lock = threading.Lock()

    def start_recording(self, _):
        if not self.recording:
            config.custom_print("Info", "Space bar pressed: Recording started.")
            self.recording = True
            self.stt.clear_full_buffer()
            self.last_impulse_text = ""
            self.impulse_subtitle = ""
            self.turn_impulse_played = False
            self.last_interjection_time = 0
            self.realtime_text_parts = []
            self.ui_status = "LISTENING..."
            self.ui_status_color = (255, 0, 0)
            
            if self.audio.is_playing.is_set():
                interrupted_text = self.audio.current_playing_text
                self.audio.interrupt()
                self.clear_sentence_queue()
                asyncio.run_coroutine_threadsafe(self.tts.abort_all(), self.loop)
                if interrupted_text:
                    asyncio.run_coroutine_threadsafe(self.trigger_interrupt_logic(interrupted_text), self.loop)
                else:
                    filler = self.audio.get_random_from_dir(config.AUDIO_DIRS['interrupt'])
                    if filler: self.audio.play_file(filler, "interrupt")

    async def trigger_interrupt_logic(self, text):
        matches = await self.vector.query(text, domain="interrupt", threshold=0.0, force_best=True)
        if matches:
            self.cache_stats["interrupt"]["hits"] += 1
            self.turn_cache_stats["interrupt"]["hits"] += 1
            fname, title, subtitle, sim = matches[0]
            config.custom_print("Play", f"[interrupt/cache sim={sim:.2f}] {subtitle}")
            self.audio.play_file(os.path.join(config.AUDIO_DIRS['interrupt'], fname), "interrupt", text=subtitle)
            if sim < config.CACHE_SIMILARITY_THRESHOLD:
                asyncio.create_task(self.generate_and_cache_interrupt(text))
        else:
            self.cache_stats["interrupt"]["misses"] += 1
            self.turn_cache_stats["interrupt"]["misses"] += 1
            asyncio.create_task(self.generate_and_cache_interrupt(text))

    async def generate_and_cache_interrupt(self, interrupted_sentence):
        res = await self.brain.generate_interrupt_prompt(interrupted_sentence)
        if res:
            await self.tts.generate(res, "interrupt", title=interrupted_sentence, subtitle=res, priority="low")

    def clear_sentence_queue(self):
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
            except asyncio.QueueEmpty: break

    def stop_recording(self, _):
        if self.recording:
            config.custom_print("Info", "Space bar released: Recording stopped.")
            self.recording = False
            self.turn_perf = {"space_release": time.time()}
            self.ui_status = "TRANSCRIBING..."
            self.ui_status_color = (255, 255, 0)
            self.awaiting_response = True
            self.loop.call_soon_threadsafe(self.tts.set_active, True)
            prelude = self.audio.get_random_from_dir(config.AUDIO_DIRS['prelude'])
            if prelude:
                config.custom_print("Play", f"[prelude/random] {os.path.basename(prelude)}")
                self.audio.play_file(prelude, "prelude")
            asyncio.run_coroutine_threadsafe(self.process_final_turn(), self.loop)

    def audio_input_callback(self, indata, frames, time_info, status):
        if self.audio.is_playing.is_set() and not self.recording:
            return
        self.panns.feed_audio(indata)
        if self.recording:
            self.stt.feed_raw_audio(indata)

    async def handle_realtime_chunk(self, chunk):
        if self.audio.is_playing.is_set() and not self.recording:
            config.custom_print("VAD", "handle_realtime_chunk: blocked by is_playing")
            return
            
        text = await self.stt.transcribe(chunk, mode="realtime")
        if not text or text == self.last_impulse_text: return
        if text.lower() in ["n/a", "[n/a]"]: return
        self.last_impulse_text = text
        
        config.custom_print("Heard", f"[realtime] {text}")
        if self.recording:
            self.realtime_text_parts.append(text)
            config.custom_print("Heard", f"[realtime-buffer] accumulated {len(self.realtime_text_parts)} parts")
        await self.trigger_interjection_logic(text)

    async def trigger_interjection_logic(self, text):
        if self.audio.is_playing.is_set() and not self.recording:
            config.custom_print("VAD", "trigger_interjection_logic: blocked by is_playing")
            return
        
        now = time.time()
        if now - self.last_interjection_time < config.INTERJECTION_COOLDOWN:
            return

        matches = await self.vector.query(text, domain="interject", threshold=0.0, force_best=True)
        dispatched = False
        
        if matches:
            self.cache_stats["interject"]["hits"] += 1
            self.turn_cache_stats["interject"]["hits"] += 1
            fname, title, subtitle, sim = matches[0]
            
            if sim < config.CACHE_SIMILARITY_THRESHOLD:
                config.custom_print("Miss", f"[interject/weak sim={sim:.2f}] {text}")
                if config.GENERATE_INTERJECT_FOR_CACHE:
                    asyncio.create_task(self.generate_and_cache_interjection(text))
            
            if subtitle.upper() == "SILENCE":
                config.custom_print("Play", f"[interject/cache sim={sim:.2f}] SILENCE (skipped)")
                dispatched = True
            else:
                config.custom_print("Play", f"[interject/cache sim={sim:.2f}] {subtitle}")
                self.add_ui_utterance("amanda", subtitle, domain="interject", cached=True)
                self.audio.play_file(os.path.join(config.AUDIO_DIRS['interject'], fname), "interject", text=subtitle)
                dispatched = True
        else:
            self.cache_stats["interject"]["misses"] += 1
            self.turn_cache_stats["interject"]["misses"] += 1
            config.custom_print("Miss", f"[interject/empty] {text}")
            if config.GENERATE_INTERJECT_FOR_CACHE: 
                asyncio.create_task(self.generate_and_cache_interjection(text))
                dispatched = True
        
        if dispatched:
            self.last_interjection_time = now

    async def trigger_impulse_logic(self, text):
        if self.turn_impulse_played: return
        matches = await self.vector.query(text, domain="impulse", threshold=0.0, force_best=True)

        if matches:
            self.cache_stats["impulse"]["hits"] += 1
            self.turn_cache_stats["impulse"]["hits"] += 1
            fname, title, subtitle, sim = matches[0]

            if sim < config.CACHE_SIMILARITY_THRESHOLD:
                config.custom_print("Miss", f"[impulse/weak sim={sim:.2f}] {text}")
                if config.GENERATE_IMPULSE_FOR_CACHE:
                    asyncio.create_task(self.generate_and_cache_impulse(text))

            if subtitle.upper() == "SILENCE":
                config.custom_print("Play", f"[impulse/cache sim={sim:.2f}] SILENCE (skipped)")
            else:
                config.custom_print("Play", f"[impulse/cache sim={sim:.2f}] {subtitle}")
                self.impulse_subtitle = subtitle
                self.add_ui_utterance("amanda", subtitle, domain="impulse", cached=True)
                await self._dispatch_utterance(subtitle, "impulse", fname=fname)
            self.turn_impulse_played = True
        else:
            self.cache_stats["impulse"]["misses"] += 1
            self.turn_cache_stats["impulse"]["misses"] += 1
            config.custom_print("Miss", f"[impulse/empty] {text}")
            if config.GENERATE_IMPULSE_FOR_CACHE:
                asyncio.create_task(self.generate_and_cache_impulse(text))

    async def _fire_impulse_gated(self, text, gate):
        try:
            await self.trigger_impulse_logic(text)
        finally:
            gate.set()

    async def _resolve_and_play_meta(self, meta_tags, meta_results, proximity=0.3):
        for tag, matches in zip(meta_tags, meta_results):
            if matches:
                self.cache_stats["meta"]["hits"] += 1
                self.turn_cache_stats["meta"]["hits"] += 1
                mfname, _, m_sub, m_sim = matches[0]
                config.custom_print("Play", f"[meta/cache sim={m_sim:.2f}] {m_sub}")
                self.add_ui_utterance("amanda", m_sub, domain="meta", cached=True)
                self.audio.play_file(os.path.join(config.AUDIO_DIRS['meta'], mfname), "meta", text=m_sub, proximity=proximity)
                if m_sim < config.CACHE_SIMILARITY_THRESHOLD and config.GENERATE_META_FOR_CACHE:
                    asyncio.create_task(self._cache_meta_tag(tag))
            else:
                self.cache_stats["meta"]["misses"] += 1
                self.turn_cache_stats["meta"]["misses"] += 1
                config.custom_print("Miss", f"[meta] {tag}")
                if config.GENERATE_META_FOR_CACHE:
                    asyncio.create_task(self._cache_meta_tag(tag))

    @staticmethod
    def _parse_proximity(text):
        match = re.search(r'&(\d+(?:\.\d+)?)&', text)
        if match:
            proximity = float(match.group(1))
            text = re.sub(r'&\d+(?:\.\d+)?&', '', text).strip()
            return text, proximity
        return text, config.DEFAULT_MIC_PROXIMITY

    async def _dispatch_utterance(self, text, domain, fname=None):
        meta_tags = [t.strip() for t in re.findall(r'\*([^*]+)\*', text)]
        clean = re.sub(r'\*[^*]+\*', '', text).strip()
        clean, proximity = self._parse_proximity(clean)

        meta_tasks = [asyncio.create_task(
            self.vector.query(tag, domain="meta", threshold=0.0, force_best=True)
        ) for tag in meta_tags]

        tts_task = None
        if clean and fname is None:
            tts_task = asyncio.create_task(
                self.vector.query(clean, domain=domain, threshold=config.TTS_SIMILARITY_THRESHOLD)
            )

        tts_job = None
        tts_matches = None
        if tts_task:
            tts_matches = await tts_task
            if not tts_matches:
                config.custom_print("Miss", f"[{domain}] {clean}")
                tts_job = await self.tts.generate(clean, domain, subtitle=clean, proximity=proximity)

        meta_results = await asyncio.gather(*meta_tasks) if meta_tasks else []
        await self._resolve_and_play_meta(meta_tags, meta_results, proximity)

        if fname:
            self.add_ui_utterance("amanda", clean or text, domain=domain, cached=True)
            self.audio.play_file(os.path.join(config.AUDIO_DIRS[domain], fname), domain, text=clean or text, proximity=proximity)
        elif clean:
            if tts_matches:
                mfname, _, m_sub, m_sim = tts_matches[0]
                config.custom_print("Play", f"[{domain}/cache sim={m_sim:.2f}] {m_sub}")
                self.add_ui_utterance("amanda", clean, domain=domain, cached=True)
                self.audio.play_file(os.path.join(config.AUDIO_DIRS[domain], mfname), domain, text=clean, proximity=proximity)
            elif tts_job:
                self.add_ui_utterance("amanda", clean, domain=domain, cached=False)
                self.audio.play_job(tts_job)

        return tts_job

    async def _generate_and_cache(self, prompt_func, domain, title, *extra_args, allow_silence=False):
        res = await prompt_func(title, *extra_args)
        if not res:
            return
        if allow_silence and res.upper() == "SILENCE":
            await self.tts.generate_silence(domain, title=title)
        else:
            await self.tts.generate(res, domain, title=title, subtitle=res, priority="low")

    async def generate_and_cache_interjection(self, user_input):
        await self._generate_and_cache(self.brain.generate_interjection_prompt, "interject", user_input, allow_silence=True)

    async def generate_and_cache_impulse(self, user_input):
        await self._generate_and_cache(self.brain.generate_impulse_prompt, "impulse", user_input, self.vector.get_random_meta_tags(num_tags=10), allow_silence=True)

    async def _cache_meta_tag(self, meta_tag):
        await self._generate_and_cache(self.brain.generate_meta_prompt, "meta", meta_tag)

    async def sentence_processor_worker(self):
        gate = asyncio.Event()
        gate.set()
        while True:
            item = await self.sentence_queue.get()
            if item is None:
                self.sentence_queue.task_done()
                break
            sentence, _ = item
            if self._turn_start_gate is not None:
                gate = self._turn_start_gate
                self._turn_start_gate = None
            next_gate = asyncio.Event()
            asyncio.create_task(self._handle_sentence(sentence, gate, next_gate))
            self._sentence_final_gate = next_gate
            gate = next_gate
            self.sentence_queue.task_done()

    async def _handle_sentence(self, sentence, prev_gate, my_gate):
        try:
            meta_tags = [t.strip() for t in re.findall(r'\*([^*]+)\*', sentence)]
            clean = re.sub(r'\*[^*]+\*', '', sentence).strip()
            clean, proximity = self._parse_proximity(clean)
            meta_tasks = [asyncio.create_task(
                self.vector.query(tag, domain="meta", threshold=0.0, force_best=True)
            ) for tag in meta_tags]
            tts_task = asyncio.create_task(
                self.vector.query(clean, domain="tts", threshold=config.TTS_SIMILARITY_THRESHOLD)
            ) if clean else None

            tts_job = None
            tts_matches = None
            if tts_task:
                tts_matches = await tts_task
                if not tts_matches:
                    self.cache_stats["tts"]["misses"] += 1
                    self.turn_cache_stats["tts"]["misses"] += 1
                    config.custom_print("Miss", f"[tts] {clean}")
                    self.turn_perf.setdefault("first_tts_post", time.time())
                    tts_job = await self.tts.generate(clean, "tts", subtitle=clean, proximity=proximity)
                    if "first_audio_byte" not in self.turn_perf:
                        tts_job.on_first_feed = lambda: self.turn_perf.setdefault("first_audio_byte", time.time())

            await prev_gate.wait()

            meta_results = await asyncio.gather(*meta_tasks) if meta_tasks else []
            await self._resolve_and_play_meta(meta_tags, meta_results, proximity)

            if clean:
                if tts_matches:
                    self.cache_stats["tts"]["hits"] += 1
                    self.turn_cache_stats["tts"]["hits"] += 1
                    fname, _, m_sub, m_sim = tts_matches[0]
                    config.custom_print("Play", f"[tts/cache sim={m_sim:.2f}] {m_sub}")
                    self.turn_perf.setdefault("first_sentence_queued", time.time())
                    self.add_ui_utterance("amanda", clean, domain="tts", cached=True)
                    self.audio.play_file(os.path.join(config.AUDIO_DIRS['tts'], fname), "tts", text=clean, proximity=proximity)
                elif tts_job:
                    self.turn_perf.setdefault("first_sentence_queued", time.time())
                    self.add_ui_utterance("amanda", clean, domain="tts", cached=False)
                    self.audio.play_job(tts_job)
        except Exception as e:
            import traceback
            config.custom_print("Error", f"_handle_sentence crashed: {e}")
            config.custom_print("Error", traceback.format_exc())
        finally:
            my_gate.set()

    async def process_final_turn(self):
        for d in config.DOMAINS:
            self.turn_cache_stats[d]["hits"] = 0
            self.turn_cache_stats[d]["misses"] = 0
        try:
            t0 = time.time()
            audio_16k = self.stt.get_full_audio()
            t1 = time.time()
            if len(audio_16k) == 0:
                config.custom_print("Heard", "[final] no audio captured")
                return

            realtime_text = " ".join(self.realtime_text_parts).strip()
            if not realtime_text and len(audio_16k) > 0:
                realtime_text = await self.stt.transcribe(audio_16k, mode="realtime")
                config.custom_print("Impulse", f"[fallback] {realtime_text!r}")
            elif realtime_text:
                config.custom_print("Impulse", f"[realtime] {realtime_text!r}")

            impulse_gate = asyncio.Event()
            if not self.turn_impulse_played and realtime_text:
                asyncio.create_task(self._fire_impulse_gated(realtime_text, impulse_gate))
            else:
                if not realtime_text:
                    config.custom_print("Impulse", "[skip] no text")
                impulse_gate.set()
            self._turn_start_gate = impulse_gate

            text = await self.stt.transcribe(audio_16k, mode="accurate")
            t2 = time.time()
            self.turn_perf["whisper_done"] = time.time()
            config.custom_print("Heard", f"[accurate] {text!r}")

            if not text:
                config.custom_print("Heard", "[final] accurate whisper returned empty")
                return

            config.custom_print("Impulse", "Waiting for impulse gate...")
            await impulse_gate.wait()
            config.custom_print("Impulse", f"Impulse gate open. subtitle={self.impulse_subtitle!r} played={self.turn_impulse_played}")

            self.add_ui_utterance("user", text, domain="user")
            user_log_entry = ' '.join([self.brain.mood, self.stt.last_audio_tags, text.strip()])
            self.brain.add_to_log("User", user_log_entry)
            config.custom_print("Chat Log", f"[User] {user_log_entry}")

            if self.impulse_subtitle and self.turn_impulse_played:
                config.custom_print("Chat Log", f"[Assistant] {self.impulse_subtitle}")
                self.brain.add_to_log("Assistant", self.impulse_subtitle)

            self.ui_status = "GENERATING..."
            self.ui_status_color = (0, 255, 255)
            self._sentence_final_gate = None
            t3 = time.time()
            stream = self.brain.generate_stream(text, audio_tags=self.stt.last_audio_tags, mood_str=self.brain.mood,
                                              meta_tags_prompt=self.vector.get_random_meta_tags(10), impulse_subtitle=self.impulse_subtitle)
            t4 = time.time()
            realtime_gpt_full_response = ""
            current_sentence_buf = ""
            response_parts = []
            self.turn_perf["gemini_request"] = time.time()
            config.custom_print("GPT", "→ request sent")
            config.custom_print("BENCH", f"get_audio={((t1-t0)*1000):.0f}ms | transcribe={((t2-t1)*1000):.0f}ms | impulse_wait={((t3-t2)*1000):.0f}ms | gen_stream_create={((t4-t3)*1000):.0f}ms")
            _last_chunks = []
            _repetition_abort = False

            async def _extract_and_queue(buf):
                while True:
                    m = re.match(r'(?:[^.!?&]|&[^&]*&)*[.!?]+', buf)
                    if not m:
                        break
                    sentence = re.sub(r'\s+', ' ', m.group()).strip()
                    buf = buf[m.end():]
                    if sentence:
                        await self.sentence_queue.put((sentence, False))
                        meta_tags = [t.strip() for t in re.findall(r'\*([^*]+)\*', sentence)]
                        clean = re.sub(r'\*[^*]+\*', '', sentence).strip()
                        response_parts.append((meta_tags, clean))
                        meta_str = f" meta={meta_tags}" if meta_tags else ""
                        config.custom_print("Sentence Splitter", f"SEGMENT: {sentence!r}{meta_str} | clean={clean!r}")
                return buf

            async for chunk in stream:
                if not chunk: continue
                t_now = time.time()
                if "gemini_first_token" not in self.turn_perf:
                    self.turn_perf["gemini_first_token"] = t_now
                    t_req = self.turn_perf.get("gemini_request")
                    latency_str = f" +{(t_now - t_req)*1000:.0f}ms" if t_req else ""
                    config.custom_print("GPT", f"← first token{latency_str}")
                current_sentence_buf += chunk
                realtime_gpt_full_response += chunk
                current_sentence_buf = await _extract_and_queue(current_sentence_buf)
                _last_chunks.append(chunk)
                if len(_last_chunks) > 6:
                    _last_chunks.pop(0)
                if len(_last_chunks) >= 4:
                    stripped = [re.sub(r'[^\w]', '', c.lower()) for c in _last_chunks]
                    if len(set(stripped)) == 1 and len(stripped[0]) > 0:
                        config.custom_print("Error", "ABORTING stream: detected repetitive token loop in Gemini output.")
                        _repetition_abort = True
                        break
                    clean_full = re.sub(r'[^\w]', '', realtime_gpt_full_response.lower())
                    if len(clean_full) > 40:
                        recent = clean_full[-40:]
                        if max(recent.count(c) for c in set(recent)) > 28:
                            config.custom_print("Error", "ABORTING stream: detected character-level repetition loop.")
                            _repetition_abort = True
                            break

            if _repetition_abort:
                current_sentence_buf += " I'm sorry, I got a bit stuck there."
                realtime_gpt_full_response += " I'm sorry, I got a bit stuck there."
                current_sentence_buf = re.sub(r'\[.*?]', '', current_sentence_buf)

            self.turn_perf["gemini_end"] = time.time()
            config.custom_print("Output", realtime_gpt_full_response)

            current_sentence_buf = await _extract_and_queue(current_sentence_buf)

            tail = current_sentence_buf.strip()
            if tail:
                await self.sentence_queue.put((tail, True))
                meta_tags = [t.strip() for t in re.findall(r'\*([^*]+)\*', tail)]
                clean = re.sub(r'\*[^*]+\*', '', tail).strip()
                response_parts.append((meta_tags, clean))
                meta_str = f" meta={meta_tags}" if meta_tags else ""
                config.custom_print("Sentence Splitter", f"TAIL: {tail!r}{meta_str} | clean={clean!r}")

            recon = ""
            for tags, clean in response_parts:
                recon += "".join(["*" + t + "*" for t in tags]) + " " + clean + " "
            if recon.strip():
                for s in re.split(r'(?<=[.!?])\s+', recon.strip()):
                    self.brain.add_to_log("Assistant", s.strip())
            asyncio.create_task(self.brain.update_mood())
            if len(self.brain.chat_log) % 5 == 0: asyncio.create_task(self.brain.extract_persona())
        except Exception as e:
            import traceback
            config.custom_print("Error", f"process_final_turn crashed: {e}")
            config.custom_print("Error", traceback.format_exc())
        finally:
            try:
                await self.sentence_queue.join()
                if self._sentence_final_gate:
                    await self._sentence_final_gate.wait()
                self.awaiting_response = False
                await asyncio.get_event_loop().run_in_executor(None, self.audio.playback_queue.join)
                while self.audio.is_playing.is_set():
                    await asyncio.sleep(0.05)
            finally:
                self._log_turn_perf()
                self.awaiting_response = False
                self.tts.set_active(False)
                self.ui_status = "IDLE"
                self.ui_status_color = (255, 255, 255)

    def _log_turn_perf(self):
        p = self.turn_perf
        t0 = p.get("space_release")
        if not t0: return
        def d(k):
            t = p.get(k)
            return f"{(t - t0) * 1000:.0f}ms" if t else "—"
        def delta(k1, k2):
            t1, t2 = p.get(k1), p.get(k2)
            return f"{(t2 - t1) * 1000:.0f}ms" if (t1 and t2) else "—"
        ttft = ((p["gemini_first_token"] - p["gemini_request"]) * 1000) if ("gemini_first_token" in p and "gemini_request" in p) else None
        ttft_s = f"{ttft:.0f}ms" if ttft is not None else "—"
        config.custom_print("Perf",
            f"whisper={d('whisper_done')}, "
            f"gpt_ttft={delta('whisper_done', 'gemini_first_token')}, "
            f"gpt_first_sentence={delta('whisper_done', 'first_sentence_queued')}, "
            f"first_token={d('gemini_first_token')} (ttft={ttft_s}), "
            f"first_sentence={d('first_sentence_queued')}, "
            f"first_audio_byte={d('first_audio_byte')}")
        parts = []
        for dom in config.DOMAINS:
            th = self.turn_cache_stats[dom]["hits"]
            tm = self.turn_cache_stats[dom]["misses"]
            ch = self.cache_stats[dom]["hits"]
            cm = self.cache_stats[dom]["misses"]
            turn_str = f"{th}/{th+tm}" if (th+tm) > 0 else "-/-"
            cum_str = f"{ch}/{ch+cm}" if (ch+cm) > 0 else "-/-"
            parts.append(f"{dom}={turn_str}→{cum_str}")
        config.custom_print("Cache", " ".join(parts))

    async def thinking_monitor(self):
        while True:
            await asyncio.sleep(config.INTERLUDE_MONITOR_INTERVAL_S)
            if self.awaiting_response and not self.audio.is_playing.is_set() and self.audio.playback_queue.empty() and not self.recording and self.sentence_queue.empty():
                interlude = self.audio.get_random_from_dir(config.AUDIO_DIRS['interlude'])
                if interlude:
                    config.custom_print("Play", f"[interlude/random] {os.path.basename(interlude)}")
                    self.audio.play_file(interlude, "interlude")

    UI_DOMAIN_COLORS = {
        "user": (0, 255, 128),  
        "tts": (255, 255, 255),
        "tts_cached": (128, 200, 255),
        "impulse": (255, 255, 0),
        "meta": (255, 0, 255),
        "interject": (255, 165, 0),
        "interrupt": (255, 64, 64),
        "prelude": (128, 128, 128),
        "breath": (100, 100, 100),
        "interlude": (150, 150, 150),
    }

    def add_ui_utterance(self, speaker, text, domain="tts", cached=False):
        if not text or not text.strip():
            return
        key = "user" if speaker == "user" else ("tts_cached" if cached and domain == "tts" else domain)
        with self._ui_lock:
            self.ui_transcript.append({
                "speaker": speaker,
                "text": text.strip(),
                "color": self.UI_DOMAIN_COLORS.get(key, (255, 255, 255)),
            })
            while len(self.ui_transcript) > 14:
                self.ui_transcript.pop(0)

    def _run_gemini_bench(self, _=None):
        async def _bench():
            text = "Testing 1, 2, 3."
            tags = self.vector.get_random_meta_tags(10)
            t0 = time.time()
            stream = self.brain.generate_stream(text, audio_tags="", mood_str="*neutral*", meta_tags_prompt=tags, impulse_subtitle="")
            t_create = time.time()
            first = None
            async for chunk in stream:
                if chunk:
                    first = time.time()
                    break
            t_end = time.time()
            config.custom_print("BENCH", f"create={((t_create-t0)*1000):.0f}ms | ttft={((first-t0)*1000):.0f}ms | first_chunk='{chunk}'")
        asyncio.run_coroutine_threadsafe(_bench(), self.loop)

    def exit_app(self):
        self.brain.save_state()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

    async def _startup(self):
        await self.brain.warm_up()
        self.stt.set_chunk_handler(lambda chunk: asyncio.run_coroutine_threadsafe(self.handle_realtime_chunk(chunk), self.loop))
        def _on_panns(score, tags):
            self.stt.total_speech_score = score
            self.stt.last_audio_tags = tags
            self.ui_panns_tags = tags
        self.panns.set_result_handler(_on_panns)
        asyncio.create_task(self.sentence_processor_worker())
        asyncio.create_task(self.thinking_monitor())

    def start_visualizer(self):
        pygame.init()
        pygame.font.init()
        win_w, win_h = 800, 1200
        win = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE | pygame.DOUBLEBUF)

        graph_w_tmp = win_w - 40
        graph_h_tmp = 130
        gradient_surf = pygame.Surface((graph_w_tmp, graph_h_tmp))
        for row in range(graph_h_tmp):
            t = 1.0 - (row / graph_h_tmp)
            r = int(8 + t * 100)
            g = int(12 + t * 45)
            b = int(35 * (1.0 - t) + 8)
            pygame.draw.line(gradient_surf, (r, g, b), (0, row), (graph_w_tmp, row))

        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 28, bold=True)
        small_font = pygame.font.SysFont("Arial", 18)
        tiny_font = pygame.font.SysFont("Arial", 14)

        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        blob_history = np.zeros((12, 5))
        mix_history = collections.deque(maxlen=300)
        input_accumulator = np.array([], dtype=np.int16)
        display_color = np.array([255.0, 255.0, 255.0])
        last_status = ""
        status_surface = None
        frequency_bands = [(20, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000)]
        _output_freqs = np.fft.rfftfreq(1024, 1 / config.TTS_SAMPLE_RATE)
        _input_freqs = np.fft.rfftfreq(1024, 1 / config.FS)
        _output_band_idx = [np.where((_output_freqs >= low) & (_output_freqs < high))[0] for low, high in frequency_bands]
        _input_band_idx = [np.where((_input_freqs >= low) & (_input_freqs < high))[0] for low, high in frequency_bands]

        running = True
        show_menu = False
        menu_pos = (0, 0)

        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        self.start_recording(None)
                    elif e.key == pygame.K_b:
                        self._run_gemini_bench()
                elif e.type == pygame.KEYUP:
                    if e.key == pygame.K_SPACE:
                        self.stop_recording(None)
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == 3:
                        show_menu = True
                        menu_pos = e.pos
                    elif e.button == 1 and show_menu:
                        menu_rect = pygame.Rect(menu_pos[0], menu_pos[1], 140, 35)
                        if menu_rect.collidepoint(e.pos):
                            running = False
                        show_menu = False

            chunk_pcm = None
            try:
                while not self.audio.visualizer_queue.empty():
                    chunk_pcm = self.audio.visualizer_queue.get_nowait()
            except queue.Empty:
                pass

            input_chunk = None
            try:
                while not self.stt.input_visualizer_queue.empty():
                    input_chunk = self.stt.input_visualizer_queue.get_nowait()
            except queue.Empty:
                pass

            if input_chunk is not None:
                mono = (input_chunk[:, 0] * 32767).astype(np.int16) if input_chunk.ndim > 1 else (input_chunk * 32767).astype(np.int16)
                input_accumulator = np.concatenate((input_accumulator, mono))
                if len(input_accumulator) > 2048:
                    input_accumulator = input_accumulator[-2048:]
            else:
                fade = min(len(input_accumulator), 200)
                if fade > 0:
                    input_accumulator = input_accumulator[fade:]

            win.fill((10, 10, 10))

            output_sizes = np.full(12, 120.0)
            if chunk_pcm:
                chunk = np.frombuffer(chunk_pcm, dtype=np.int16)
                if len(chunk) > 0:
                    fft_data = np.abs(np.fft.rfft(chunk, n=1024))
                    for idx, relevant in enumerate(_output_band_idx):
                        if len(relevant) > 0:
                            amp = np.mean(fft_data[relevant])
                            output_sizes[idx % 12] = np.interp(amp, [0, 50000], [120, 240])

            input_sizes = np.full(12, 120.0)
            if len(input_accumulator) >= 512:
                fft_data = np.abs(np.fft.rfft(input_accumulator, n=1024))
                for idx, relevant in enumerate(_input_band_idx):
                    if len(relevant) > 0:
                        amp = np.mean(fft_data[relevant])
                        input_sizes[idx % 12] = np.interp(amp, [0, 50000], [120, 240])

            combined_sizes = np.clip(output_sizes + input_sizes - 120.0, 120.0, 360.0)
            blob_history = np.roll(blob_history, -1, axis=1)
            blob_history[:, -1] = combined_sizes
            avg_sizes = np.mean(blob_history, axis=1)
            pts = np.array([avg_sizes * np.cos(angles), avg_sizes * np.sin(angles)]).T
            tck = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0, per=True)[0]
            smooth = interpolate.splev(np.linspace(0, 1, 100), tck)
            poly = np.array(smooth).T + [win_w//2, 300]

            mood_target = np.array(self.brain.mood_color, dtype=np.float32)
            display_color = display_color * 0.92 + mood_target * 0.08

            out_energy = max(0, np.mean(output_sizes) - 120.0)
            in_energy = max(0, np.mean(input_sizes) - 120.0)
            input_color = np.array([0, 255, 128], dtype=np.float32)

            if out_energy < 5 and in_energy < 5:
                final_color = tuple(display_color.astype(int))
            elif out_energy < 5:
                final_color = tuple(input_color.astype(int))
            elif in_energy < 5:
                final_color = tuple(display_color.astype(int))
            else:
                total = out_energy + in_energy
                ratio = in_energy / total
                c = display_color * (1.0 - ratio) + input_color * ratio
                final_color = tuple(np.clip(c, 0, 255).astype(int))

            pygame.draw.polygon(win, final_color, poly.astype(int))

            if self.audio.global_spatial:
                mv = self.audio.global_spatial.last_mix_values
                mix_history.append((mv['spline'], mv['effective'], mv['pan']))
            else:
                mix_history.append((0.0, 0.0, 0.0))

            if self.ui_status != last_status:
                last_status = self.ui_status
                status_surface = font.render(self.ui_status, True, self.ui_status_color)
            if status_surface:
                rect = status_surface.get_rect(center=(win_w//2, 60))
                win.blit(status_surface, rect)

            if self.ui_panns_tags:
                panns_surf = tiny_font.render(self.ui_panns_tags[:100], True, (180, 255, 180))
                win.blit(panns_surf, (20, 20))

            metrics = [
                f"playback: {self.audio.playback_queue.qsize()}",
                f"sentences: {self.sentence_queue.qsize()}",
                f"mood: {self.brain.mood}",
            ]
            y = 80 if self.ui_panns_tags else 20
            for m in metrics:
                surf = tiny_font.render(m, True, (180, 180, 180))
                win.blit(surf, (win_w - surf.get_width() - 20, y))
                y += 18

            graph_x, graph_y = 20, 720
            graph_w, graph_h = win_w - 40, 130
            win.blit(gradient_surf, (graph_x, graph_y))
            pygame.draw.rect(win, (80, 80, 80), (graph_x, graph_y, graph_w, graph_h), 1)

            zone_labels = [(0.0, "0.0", "CLOSE"), (0.5, "0.5", "MID"), (1.0, "1.0", "FAR")]
            for val, num_label, zone_label in zone_labels:
                gy = graph_y + graph_h - int(val * graph_h)
                pygame.draw.line(win, (60, 60, 60, 128), (graph_x, gy), (graph_x + graph_w, gy), 1)
                surf = tiny_font.render(num_label, True, (160, 160, 160))
                win.blit(surf, (graph_x - 30, gy - 6))
                zs = tiny_font.render(zone_label, True, (140, 140, 140))
                win.blit(zs, (graph_x + graph_w + 6, gy - 6))

            if len(mix_history) >= 2:
                step = graph_w / 300
                for i in range(1, len(mix_history)):
                    x0 = graph_x + int((i - 1) * step)
                    x1 = graph_x + int(i * step)
                    y0_s = graph_y + graph_h - int(mix_history[i-1][0] * graph_h)
                    y1_s = graph_y + graph_h - int(mix_history[i][0] * graph_h)
                    pygame.draw.line(win, (0, 240, 255), (x0, y0_s), (x1, y1_s), 2)
                    y0_e = graph_y + graph_h - int(mix_history[i-1][1] * graph_h)
                    y1_e = graph_y + graph_h - int(mix_history[i][1] * graph_h)
                    pygame.draw.line(win, (255, 200, 60), (x0, y0_e), (x1, y1_e), 2)

            win.blit(tiny_font.render("spline", True, (0, 240, 255)), (graph_x + 8, graph_y + 4))
            win.blit(tiny_font.render("effective", True, (255, 200, 60)), (graph_x + 65, graph_y + 4))

            with self._ui_lock:
                lines = list(self.ui_transcript)
            y = win_h - 20
            for entry in reversed(lines[-14:]):
                line = f"{entry['speaker']}: {entry['text'][:70]}"
                surf = small_font.render(line, True, entry['color'])
                rect = surf.get_rect(bottomleft=(20, y))
                win.blit(surf, rect)
                y -= 22

            if show_menu:
                pygame.draw.rect(win, (40, 40, 40), (menu_pos[0], menu_pos[1], 140, 35))
                pygame.draw.rect(win, (200, 200, 200), (menu_pos[0], menu_pos[1], 140, 35), 1)
                surf = small_font.render("Exit", True, (255, 255, 255))
                win.blit(surf, (menu_pos[0] + 10, menu_pos[1] + 8))

            pygame.display.flip()
            clock.tick(60)

        self.exit_app()

    def run(self):
        tts_server_bootstrap.start()
        if not tts_server_bootstrap.is_ready():
            config.custom_print("Info", "Waiting for TTS server to be ready...")
            if not tts_server_bootstrap.wait_ready(timeout=180):
                config.custom_print("Error", "TTS server never became ready. Voice output will fail.")
        self.loop = asyncio.new_event_loop()
        def _run_async():
            asyncio.set_event_loop(self.loop)
            self.loop.create_task(self._startup())
            self.loop.run_forever()
        threading.Thread(target=_run_async, daemon=True).start()
        with sd.InputStream(callback=self.audio_input_callback, samplerate=config.FS, blocksize=64, channels=1):
            self.start_visualizer()

if __name__ == "__main__":
    app = AmandaApp()
    app.run()