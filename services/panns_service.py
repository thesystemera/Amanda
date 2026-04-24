import queue
import threading
import time
import numpy as np
import samplerate
from panns_inference import AudioTagging
import config

SPEECH_DETECTION_TAGS = {
    'speech': True,
    'male speech, man speaking': True,
    'female speech, woman speaking': True,
    'child speech, kid speaking': True,
    'conversation': True,
    'narration, monologue': True,
}
SPEECH_DETECTION_TAGS_LOWER = {tag.lower() for tag in SPEECH_DETECTION_TAGS}

_CLEAR = object()
_SHUTDOWN = object()

class PANNsService:
    def __init__(self):
        config.custom_print("Lifespan", "PANNsService: loading AudioTagging on CUDA...")
        self.at = AudioTagging(checkpoint_path=None, device="cuda")

        self.target_sample_rate = 16000
        self.audio_buffer = []
        self._buffer_samples = 0

        self.total_speech_score = 0.0
        self.top_tags_str = ""
        self.prediction_accumulator_speech = {}
        self.prediction_accumulator_other = {}
        self.baseline = {}
        self.prev_top_speech_tags = []
        self.prev_top_other_tags = []

        self.input_queue = queue.Queue()
        self._result_handler = None
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="panns-classifier")
        self._worker_thread.start()

        config.custom_print("Lifespan", "PANNsService: ready.")

    def set_result_handler(self, handler):
        self._result_handler = handler

    def feed_audio(self, indata):
        self.input_queue.put(indata.copy())

    def clear_buffer(self):
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        self.input_queue.put(_CLEAR)

    def _worker(self):
        last_classification_time = 0
        while True:
            item = self.input_queue.get()
            if item is _SHUTDOWN:
                break
            if item is _CLEAR:
                self.audio_buffer = []
                self._buffer_samples = 0
                self.prediction_accumulator_speech = {}
                self.prediction_accumulator_other = {}
                last_classification_time = time.time()
                continue

            resampled = samplerate.resample(item[:, 0], self.target_sample_rate / config.FS, 'linear')
            self.audio_buffer.append(resampled)
            self._buffer_samples += len(resampled)
            while self._buffer_samples > self.target_sample_rate * 4 and self.audio_buffer:
                self._buffer_samples -= len(self.audio_buffer.pop(0))

            now = time.time()
            if now - last_classification_time >= config.AUDIO_CLASSIFICATION_INTERVAL_S:
                self._run_classification()
                last_classification_time = now

    def _run_classification(self):
        if not self.audio_buffer:
            config.custom_print("Audio Classification", "PANNs: buffer empty, skipping")
            return
        audio = np.concatenate(self.audio_buffer)
        if len(audio) < self.target_sample_rate * 3:
            config.custom_print("Audio Classification", f"PANNs: buffer too short ({len(audio)} < {self.target_sample_rate * 3}), skipping")
            return
        config.custom_print("Audio Classification", f"PANNs: running inference on {len(audio)} samples ({len(audio)/self.target_sample_rate:.1f}s)")

        audio = audio[None, :]
        try:
            (clipwise, _) = self.at.inference(audio)

            speech_accum = {}
            other_accum = {}
            for label, prob in zip(self.at.labels, clipwise[0]):
                prob = float(prob)
                label_lower = label.lower()

                self.baseline[label] = self.baseline.get(label, 0.0) * 0.995 + prob * 0.005

                if label_lower in SPEECH_DETECTION_TAGS_LOWER:
                    self.prediction_accumulator_speech[label] = self.prediction_accumulator_speech.get(label, 0.0) * config.SPEECH_DETECTION_DECAY_FACTOR + prob
                    speech_accum[label] = self.prediction_accumulator_speech[label]
                else:
                    self.prediction_accumulator_other[label] = self.prediction_accumulator_other.get(label, 0.0) * config.OTHER_AUDIO_DECAY_FACTOR + prob
                    other_accum[label] = self.prediction_accumulator_other[label]

            speech_ranked = sorted(
                [(l, s - self.baseline.get(l, 0.0)) for l, s in speech_accum.items()],
                key=lambda x: x[1], reverse=True
            )
            other_ranked = sorted(
                [(l, s - self.baseline.get(l, 0.0)) for l, s in other_accum.items()],
                key=lambda x: x[1], reverse=True
            )

            top_speech_tags = [tag for tag in speech_ranked if tag[1] > config.CLASSIFICATION_SPEECH_DETECTION_THRESHOLD][:5]
            top_other_tags = [tag for tag in other_ranked if tag[1] > config.CLASSIFICATION_OTHER_AUDIO_THRESHOLD][:5]

            self.total_speech_score = sum(float(score) for _, score in top_speech_tags)

            self.top_tags_str = ", ".join(f"{label} ({score:.2f})" for label, score in top_other_tags[:5])
            config.custom_print("Audio Classification", f"PANNs result: speech_score={self.total_speech_score:.2f} | tags={self.top_tags_str or '(none)'}")

            if top_speech_tags != self.prev_top_speech_tags:
                self.prev_top_speech_tags = top_speech_tags
                if top_speech_tags:
                    config.custom_print("Audio Classification", f"Top speech detection: {', '.join(f'{l} ({s:.2f})' for l, s in top_speech_tags)}")

            if top_other_tags != self.prev_top_other_tags:
                self.prev_top_other_tags = top_other_tags
                if top_other_tags:
                    config.custom_print("Audio Classification", f"Top other predictions: {', '.join(f'{l} ({s:.2f})' for l, s in top_other_tags)}")

            if len(self.prediction_accumulator_speech) > 50: self.prediction_accumulator_speech = {}
            if len(self.prediction_accumulator_other) > 50: self.prediction_accumulator_other = {}

            if self._result_handler:
                self._result_handler(self.total_speech_score, self.top_tags_str)

        except Exception as e:
            config.custom_print("Error", f"PANNs classification error: {e}")