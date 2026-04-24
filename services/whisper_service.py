import queue
import threading
import asyncio
from faster_whisper import WhisperModel
import config

class WhisperService:
    def __init__(self):
        config.custom_print("Lifespan", "WhisperService: loading distil-small.en + large-v3-turbo on CUDA...")
        self.model_realtime = WhisperModel("distil-small.en", device="cuda", compute_type="float16")
        self.model_accurate = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

        self._request_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="whisper-worker")
        self._worker_thread.start()

        config.custom_print("Lifespan", "WhisperService: ready.")

    async def transcribe(self, audio_data, mode="realtime"):
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._request_queue.put((audio_data, mode, future))
        return await future

    def _worker(self):
        while True:
            item = self._request_queue.get()
            if item is None:
                break
            audio_data, mode, future = item
            try:
                model = self.model_realtime if mode == "realtime" else self.model_accurate
                vad_filter = mode != "realtime"
                segments, _ = model.transcribe(
                    audio_data,
                    vad_filter=vad_filter,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    beam_size=1,
                )
                text = " ".join([s.text for s in segments]).strip()
            except Exception as e:
                config.custom_print("Error", f"Whisper error ({mode}): {e}")
                text = ""

            if not future.done():
                loop = future.get_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(future.set_result, text)