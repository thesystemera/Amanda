import queue
import threading
import numpy as np
import torch
import samplerate
import config

class TranscriptionService:
    def __init__(self, whisper_service=None):
        config.custom_print("Lifespan", f"TranscriptionService: loading Silero VAD v5 on {config.DEVICE}...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
        )
        self.vad_model = self.vad_model.to(config.DEVICE)
        self.vad_model.eval()

        self.target_sample_rate = 16000
        self.frame_duration = 32 # ms
        self.vad_frame_length = 512

        self.accumulated_audio = np.array([], dtype=np.float32)
        self.full_recording_16k = []

        self.speech_segments = []
        self.speech_frame_count = 0
        self.last_speech_time = 0
        self.buffer_start_time = 0
        self.current_time = 0

        self.last_audio_tags = ""
        self.total_speech_score = 0

        self.input_queue = queue.Queue()
        self.input_visualizer_queue = queue.Queue(maxsize=100)
        self._full_buffer_lock = threading.Lock()
        self._chunk_handler = None

        self.whisper = whisper_service

        self._vad_thread = threading.Thread(target=self._vad_worker, daemon=True, name="vad-processor")
        self._vad_thread.start()

        config.custom_print("Lifespan", "TranscriptionService: ready.")

    def set_chunk_handler(self, handler):
        self._chunk_handler = handler

    def feed_raw_audio(self, indata):
        self.input_queue.put(indata.copy())

    def clear_full_buffer(self):
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        with self._full_buffer_lock:
            self.full_recording_16k = []
        self.vad_model.reset_states()
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.speech_segments = []
        self.speech_frame_count = 0
        self.last_speech_time = 0
        self.buffer_start_time = 0
        self.current_time = 0

    def get_full_audio(self):
        with self._full_buffer_lock:
            if not self.full_recording_16k:
                return np.array([], dtype=np.float32)
            return np.concatenate(self.full_recording_16k)

    def _vad_worker(self):
        while True:
            indata = self.input_queue.get()
            if indata is None:
                break

            resampled = samplerate.resample(indata[:, 0], self.target_sample_rate / config.FS, 'linear')

            try:
                self.input_visualizer_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

            with self._full_buffer_lock:
                self.full_recording_16k.append(resampled)

            self.accumulated_audio = np.concatenate((self.accumulated_audio, resampled))

            realtime_jobs = []
            while len(self.accumulated_audio) >= self.vad_frame_length:
                frame = self.accumulated_audio[:self.vad_frame_length]
                self.accumulated_audio = self.accumulated_audio[self.vad_frame_length:]

                is_speech = self.check_vad(frame)
                has_energy = self._frame_has_energy(frame) or self.total_speech_score > 0.15

                if is_speech and has_energy:
                    if not self.speech_segments: self.buffer_start_time = self.current_time
                    self.speech_segments.append(frame)
                    self.speech_frame_count += 1
                    self.last_speech_time = self.current_time

                self.current_time += self.frame_duration / 1000.0

                if self.speech_segments and (
                    (self.current_time - self.last_speech_time) * 1000 >= config.MIN_BUFFER_DURATION_MS or
                    (self.current_time - self.buffer_start_time) * 1000 >= config.MAX_BUFFER_DURATION_MS
                ):
                    speech_ms = self.speech_frame_count * self.frame_duration
                    min_duration = config.MIN_SPEECH_DURATION_MS if self.total_speech_score < 0.3 else 100

                    if speech_ms >= min_duration:
                        chunk = np.concatenate(self.speech_segments)
                        rms = float(np.sqrt(np.mean(chunk ** 2)))
                        if rms >= config.MIN_CHUNK_RMS or self.total_speech_score > 0.25:
                            realtime_jobs.append(chunk)
                        else:
                            config.custom_print("VAD", f"Rejected low-RMS chunk ({speech_ms}ms, rms={rms:.4f}, panns={self.total_speech_score:.2f}).")
                    else:
                        config.custom_print("VAD", f"Rejected short chunk ({speech_ms}ms < {min_duration}ms, panns={self.total_speech_score:.2f}).")
                    self.speech_segments = []
                    self.speech_frame_count = 0

            for chunk in realtime_jobs:
                if self._chunk_handler:
                    self._chunk_handler(chunk)

    def _frame_has_energy(self, frame_float32):
        rms = float(np.sqrt(np.mean(frame_float32 ** 2)))
        return rms >= config.MIN_FRAME_RMS

    async def transcribe(self, audio_data, mode="realtime"):
        return await self.whisper.transcribe(audio_data, mode=mode)

    def check_vad(self, frame_float32):
        with torch.no_grad():
            tensor = torch.from_numpy(frame_float32).float().to(config.DEVICE)
            prob = self.vad_model(tensor, self.target_sample_rate).item()
        return prob >= config.VAD_SPEECH_THRESHOLD