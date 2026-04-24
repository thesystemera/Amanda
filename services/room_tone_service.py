import threading
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

class RoomTonePlayer:

    def __init__(self, path: str, sample_rate: int = 24000, volume_db: float = -42):
        self.path = path
        self.sample_rate = sample_rate
        self._base_volume = 10 ** (volume_db / 20)
        self._target_volume = self._base_volume
        self._current_volume = self._base_volume
        self._volume_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._data: np.ndarray | None = None

    def start(self):
        if self._thread is not None:
            return

        audio = AudioSegment.from_file(self.path)
        if audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)
        if audio.channels == 1:
            audio = audio.set_channels(2)

        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        if audio.channels == 2:
            self._data = samples.reshape((-1, 2))
        else:
            self._data = samples.reshape(-1, 1)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="room-tone")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def speech_active(self):
        with self._volume_lock:
            self._target_volume = self._base_volume

    def speech_inactive(self):
        with self._volume_lock:
            self._target_volume = 0.0

    def _run(self):
        idx = 0
        blocksize = 2048
        ramp_coeff = 0.06

        def callback(outdata, frames, time_info, status):
            nonlocal idx
            with self._volume_lock:
                target = self._target_volume
            self._current_volume = self._current_volume * (1.0 - ramp_coeff) + target * ramp_coeff
            vol = self._current_volume
            needed = frames
            out_pos = 0
            while needed > 0:
                avail = min(needed, len(self._data) - idx)
                outdata[out_pos:out_pos + avail] = self._data[idx:idx + avail] * vol
                out_pos += avail
                needed -= avail
                idx = (idx + avail) % len(self._data)

        channels = self._data.shape[1]
        self._data = np.ascontiguousarray(self._data, dtype=np.float32)

        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=channels,
            dtype='float32',
            blocksize=blocksize,
            callback=callback,
        ):
            while not self._stop_event.is_set():
                self._stop_event.wait(0.1)