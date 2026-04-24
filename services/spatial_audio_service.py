import io
import os
import time
import numpy as np
from pydub import AudioSegment
from pedalboard import Pedalboard, Reverb, Gain, HighShelfFilter, LowShelfFilter, Compressor  # type: ignore
from noise import pnoise1  # type: ignore
import config


class StreamingSpatialProcessor:
    """
    Stateful chunk-by-chunk spatial processor for live TTS streams.
    Maintains reverb tail across chunks — no buffering, no latency penalty.
    Think of it as someone holding a volume knob, drifting toward target mix.
    """

    def __init__(self, sample_rate: int, speaker: str = None, start_mix: float = 0.0):
        self.sample_rate = sample_rate
        self.base_pan = config.SPATIAL_SPEAKER_PAN.get(speaker, 0.0) if speaker else 0.0
        self.current_mix = float(start_mix)
        self._pan_noise_idx = 0.0
        self._mix_noise_idx = 0.0
        self.fx_cfg = config.SPATIAL_AUDIO_CONFIG

        # Spline state — reset on every set_target_mix()
        self._src_start_time = None
        self._src_start_mix = None
        self._src_target_mix = None
        self._src_next_mix = None
        self._src_duration = None
        self._ramp_ratio = 0.25  # first/last 25% of clip for ramps

        # Exposed for real-time visualizer
        self.last_mix_values = {'spline': 0.0, 'effective': 0.0, 'pan': 0.0}

        # Parallel signal path:
        #   Dry: compressor → EQ (mic/preamp path, proximity-morphed)
        #   Wet: reverb only (room ambience, receives raw panned signal)
        fx = self.fx_cfg
        c_cfg = fx['compressor']

        # Dry board — mic signal: compression + proximity EQ
        self.dry_board = Pedalboard()
        self.compressor = Compressor(
            threshold_db=c_cfg['threshold_near'],
            ratio=c_cfg['ratio_near'],
            attack_ms=c_cfg['attack_ms'],
            release_ms=c_cfg['release_ms'],
        )
        self.high_shelf = HighShelfFilter(
            cutoff_frequency_hz=fx['eq']['high_cut_freq_near'],
            gain_db=fx['eq']['high_cut_near'],
        )
        self.low_shelf = LowShelfFilter(
            cutoff_frequency_hz=fx['eq']['low_boost_freq_near'],
            gain_db=fx['eq']['low_boost_near'],
        )
        self.dry_board.append(self.compressor)
        self.dry_board.append(self.high_shelf)
        self.dry_board.append(self.low_shelf)

        # Wet board — room ambience: pure reverb, no dry pass-through
        self.wet_board = Pedalboard()
        self.reverb = Reverb(
            room_size=fx['reverb']['room_size'],
            damping=fx['reverb']['damping'],
            wet_level=fx['reverb']['wet_level'],
            dry_level=0.0,  # wet only; dry is handled by dry_board + gain scaling
        )
        self.wet_board.append(self.reverb)

    def set_target_mix(self, mix: float, next_mix: float = None, duration_hint: float = None):
        self._src_start_time = time.time()
        self._src_start_mix = float(self.current_mix)
        self._src_target_mix = float(mix)
        self._src_next_mix = float(next_mix) if next_mix is not None else float(mix)
        self._src_duration = float(duration_hint) if duration_hint is not None else None

    def _compute_mix(self):
        """Time-based cosine spline: start ramp → hold → end ramp."""
        if self._src_duration is None or self._src_duration <= 0:
            # No duration hint — gentle exponential drift
            return self.current_mix * 0.88 + self._src_target_mix * 0.12

        elapsed = time.time() - self._src_start_time
        ramp_time = self._src_duration * self._ramp_ratio

        if elapsed < ramp_time:
            # Start ramp: start_mix → target_mix (cosine ease-in-out)
            t = elapsed / ramp_time
            return self._src_start_mix + (self._src_target_mix - self._src_start_mix) * (0.5 - 0.5 * np.cos(t * np.pi))
        elif elapsed < self._src_duration - ramp_time:
            # Hold at target
            return self._src_target_mix
        else:
            # End ramp: target_mix → next_mix (cosine ease-in-out)
            t = min(1.0, (elapsed - (self._src_duration - ramp_time)) / ramp_time)
            return self._src_target_mix + (self._src_next_mix - self._src_target_mix) * (0.5 - 0.5 * np.cos(t * np.pi))

    def process(self, pcm_mono_bytes: bytes) -> bytes:
        if not pcm_mono_bytes:
            return b""

        # Spline-based mix + Perlin jitter
        self.current_mix = self._compute_mix()
        mix_jitter = pnoise1(self._mix_noise_idx, octaves=1) * 0.04
        pan_jitter = pnoise1(self._pan_noise_idx, octaves=2, persistence=0.3, base=42) * 0.06
        # Advance noise proportional to chunk length so timbre stays consistent
        # regardless of AUDIO_CHUNK_SIZE. Tuned for 4096-byte (2048-frame) chunks.
        frame_ratio = len(pcm_mono_bytes) / 4096.0
        self._mix_noise_idx += 0.15 * frame_ratio
        self._pan_noise_idx += 0.02 * frame_ratio

        effective_mix = np.clip(self.current_mix + mix_jitter, 0.0, 1.0)
        pan = np.clip(self.base_pan + pan_jitter, -1.0, 1.0)

        # Expose for visualizer
        self.last_mix_values = {
            'spline': float(self.current_mix),
            'effective': float(effective_mix),
            'pan': float(pan),
        }

        # Update dry-path parameters — proximity-morphed compressor + EQ
        fx = self.fx_cfg
        self.high_shelf.gain_db = fx['eq']['high_cut_near'] + (fx['eq']['high_cut_far'] - fx['eq']['high_cut_near']) * effective_mix
        self.high_shelf.cutoff_frequency_hz = fx['eq']['high_cut_freq_near'] + (fx['eq']['high_cut_freq_far'] - fx['eq']['high_cut_freq_near']) * effective_mix
        self.low_shelf.gain_db = fx['eq']['low_boost_near'] + (fx['eq']['low_boost_far'] - fx['eq']['low_boost_near']) * effective_mix
        self.low_shelf.cutoff_frequency_hz = fx['eq']['low_boost_freq_near'] + (fx['eq']['low_boost_freq_far'] - fx['eq']['low_boost_freq_near']) * effective_mix
        c = fx['compressor']
        self.compressor.threshold_db = c['threshold_near'] + (c['threshold_far'] - c['threshold_near']) * effective_mix
        self.compressor.ratio = c['ratio_near'] + (c['ratio_far'] - c['ratio_near']) * effective_mix

        # Mono int16 → float32 stereo
        samples = np.frombuffer(pcm_mono_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        stereo = np.column_stack((samples, samples))

        # Pan
        left_gain = np.cos((pan + 1) * np.pi / 4)
        right_gain = np.sin((pan + 1) * np.pi / 4)
        stereo[:, 0] *= left_gain
        stereo[:, 1] *= right_gain

        # Parallel processing: raw → room reverb  |  raw → compressor → EQ → dry
        wet = self.wet_board(stereo, sample_rate=self.sample_rate, reset=False)
        dry = self.dry_board(stereo, sample_rate=self.sample_rate, reset=False)

        # Mix: dry attenuates with proximity, wet stays constant (fixed room)
        dry_gain = fx['reverb']['dry_near'] + (fx['reverb']['dry_far'] - fx['reverb']['dry_near']) * effective_mix
        proc = dry * dry_gain + wet

        # Clip and convert back to int16 bytes
        max_abs = np.max(np.abs(proc))
        if max_abs > 1.0:
            proc = proc / max_abs
        proc = (proc * 32767).astype(np.int16)
        return proc.tobytes()


class SpatialAudioService:
    """
    Real-time spatial audio processor ported from AI_RADIO.
    Splits audio into 200 ms segments, applies Perlin-noise-driven
    reverb / EQ / gain / pan variation, then crossfades back together.
    """

    def __init__(self):
        self.segment_length_ms = 200
        self.crossfade_ratio = 0.5
        self.noise_scale = 0.3
        self.variation_amount = 0.1
        self.pan_noise_scale = 0.02
        self.pan_variation = 0.08
        self.fade_duration = 100

    def _normalize_noise(self, values: np.ndarray) -> np.ndarray:
        mn, mx = np.min(values), np.max(values)
        if mx - mn < 1e-9:
            return np.zeros_like(values)
        return (values - mn) / (mx - mn) * 2 - 1

    def process(
        self,
        audio_input,
        sample_rate: int = 24000,
        speaker: str = None,
        mix: float = 1.0,
        previous_segment_end_mix: float = None,
        next_segment_start_mix: float = None,
    ) -> AudioSegment:
        """
        Process an audio clip through the spatial pipeline.

        Args:
            audio_input: file path (str), raw bytes, or pydub AudioSegment
            sample_rate: target sample rate (only used when input is raw bytes)
            speaker: voice name for base pan position
            mix: 0.0 = dry, 1.0 = fully wet
            previous_segment_end_mix: optional ramp from previous clip
            next_segment_start_mix: optional ramp to next clip
        """
        if not config.SPATIAL_AUDIO_ENABLED:
            if isinstance(audio_input, AudioSegment):
                return audio_input
            if isinstance(audio_input, str):
                return AudioSegment.from_file(audio_input, format="mp3")
            if isinstance(audio_input, bytes):
                return AudioSegment.from_file(io.BytesIO(audio_input), format="mp3")
            raise ValueError("Invalid audio input type")

        # ── load into pydub ───────────────────────────────────────────────
        if isinstance(audio_input, str):
            audio = AudioSegment.from_file(audio_input, format="mp3")
        elif isinstance(audio_input, bytes):
            audio = AudioSegment.from_file(io.BytesIO(audio_input), format="mp3")
        elif isinstance(audio_input, AudioSegment):
            audio = audio_input
        else:
            raise ValueError("Invalid audio input type")

        if audio.channels == 1:
            audio = audio.set_channels(2)

        audio = audio.fade_in(self.fade_duration).fade_out(self.fade_duration)

        # ── segmentation ──────────────────────────────────────────────────
        crossfade_ms = int(self.segment_length_ms * self.crossfade_ratio)
        effective_segment_length = self.segment_length_ms - crossfade_ms
        num_segments = (len(audio) // effective_segment_length) + 1

        # Perlin noise for organic variation
        mix_noise = self._normalize_noise(
            np.array([pnoise1(i * self.noise_scale, octaves=1) for i in range(num_segments)])
        )
        pan_noise = self._normalize_noise(
            np.array([pnoise1(i * self.pan_noise_scale, octaves=2, persistence=0.3, base=42)
                      for i in range(num_segments)])
        )

        # Base mix values with optional cross-clip ramps
        base_values = np.full(num_segments, mix)
        if previous_segment_end_mix is not None:
            ramp_length = num_segments // 3
            ramp = np.cos(np.linspace(np.pi, 2 * np.pi, ramp_length)) * 0.5 + 0.5
            base_values[:ramp_length] = previous_segment_end_mix + (
                mix - previous_segment_end_mix
            ) * ramp
        if next_segment_start_mix is not None:
            ramp_length = num_segments // 3
            ramp = np.cos(np.linspace(0, np.pi, ramp_length)) * 0.5 + 0.5
            base_values[-ramp_length:] = mix + (next_segment_start_mix - mix) * ramp

        variation_range = min(
            self.variation_amount,
            np.min(base_values) * 0.5,
            (1 - np.max(base_values)) * 0.5,
        )
        mix_values = np.clip(base_values + (mix_noise * variation_range), 0.0, 1.0)

        base_pan = config.SPATIAL_SPEAKER_PAN.get(speaker, 0.0) if speaker else 0.0
        pan_values = base_pan + (pan_noise * self.pan_variation)

        fx_cfg = config.SPATIAL_AUDIO_CONFIG

        # ── process segments ──────────────────────────────────────────────
        processed_segments = []
        for i in range(num_segments):
            start_ms = i * effective_segment_length
            end_ms = min(start_ms + self.segment_length_ms, len(audio))
            segment = audio[start_ms:end_ms]

            samples = np.array(segment.get_array_of_samples()).astype(np.float32) / 32768.0
            if segment.channels == 2:  # type: ignore
                samples = samples.reshape((-1, 2))

            segment_mix = float(mix_values[i])
            pan_position = float(pan_values[i])

            # Panning
            if samples.shape[1] == 2:
                left_gain = np.cos((pan_position + 1) * np.pi / 4)
                right_gain = np.sin((pan_position + 1) * np.pi / 4)
                samples[:, 0] *= left_gain
                samples[:, 1] *= right_gain

            # Build pedalboard — physical model: compressor → reverb → EQ
            board = Pedalboard()

            c_cfg = fx_cfg['compressor']
            comp_thresh = c_cfg['threshold_near'] + (c_cfg['threshold_far'] - c_cfg['threshold_near']) * segment_mix
            comp_ratio = c_cfg['ratio_near'] + (c_cfg['ratio_far'] - c_cfg['ratio_near']) * segment_mix
            board.append(Compressor(
                threshold_db=comp_thresh,
                ratio=comp_ratio,
                attack_ms=c_cfg['attack_ms'],
                release_ms=c_cfg['release_ms'],
            ))

            reverb_cfg = fx_cfg['reverb']
            dry = reverb_cfg['dry_near'] + (reverb_cfg['dry_far'] - reverb_cfg['dry_near']) * segment_mix
            board.append(Reverb(
                room_size=reverb_cfg['room_size'],
                damping=reverb_cfg['damping'],
                wet_level=reverb_cfg['wet_level'],
                dry_level=dry,
            ))

            high_cut = fx_cfg['eq']['high_cut_near'] + (fx_cfg['eq']['high_cut_far'] - fx_cfg['eq']['high_cut_near']) * segment_mix
            high_cut_freq = fx_cfg['eq']['high_cut_freq_near'] + (fx_cfg['eq']['high_cut_freq_far'] - fx_cfg['eq']['high_cut_freq_near']) * segment_mix
            board.append(HighShelfFilter(cutoff_frequency_hz=high_cut_freq, gain_db=high_cut))

            low_boost = fx_cfg['eq']['low_boost_near'] + (fx_cfg['eq']['low_boost_far'] - fx_cfg['eq']['low_boost_near']) * segment_mix
            low_boost_freq = fx_cfg['eq']['low_boost_freq_near'] + (fx_cfg['eq']['low_boost_freq_far'] - fx_cfg['eq']['low_boost_freq_near']) * segment_mix
            board.append(LowShelfFilter(cutoff_frequency_hz=low_boost_freq, gain_db=low_boost))

            proc = board(samples, sample_rate=segment.frame_rate)

            if proc.size == 0:
                return AudioSegment.empty()

            max_abs = np.max(np.abs(proc))
            if max_abs > 1.0:
                proc = proc / max_abs

            proc = (proc * 32767).astype(np.int16)
            proc_seg = AudioSegment(
                proc.tobytes(),  # type: ignore
                frame_rate=segment.frame_rate,  # type: ignore
                sample_width=2,
                channels=2,
            )
            processed_segments.append(proc_seg)

        # ── crossfade stitch ──────────────────────────────────────────────
        final_audio = processed_segments[0]
        for seg in processed_segments[1:]:
            cf = min(len(seg), len(final_audio), crossfade_ms)
            final_audio = final_audio.append(seg, crossfade=cf)

        return final_audio

    def process_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        speaker: str = None,
        mix: float = 1.0,
        previous_segment_end_mix: float = None,
        next_segment_start_mix: float = None,
    ) -> bytes:
        """
        Convenience wrapper: raw mono int16 bytes → spatial stereo int16 bytes.
        """
        audio = AudioSegment(
            data=pcm_bytes,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )
        processed = self.process(
            audio,
            sample_rate,
            speaker,
            mix,
            previous_segment_end_mix=previous_segment_end_mix,
            next_segment_start_mix=next_segment_start_mix,
        )
        return processed.raw_data


def generate_room_tone(path: str, duration_s: float = 60.0, sample_rate: int = 24000):
    """
    Generate a looping brown-noise room-tone WAV.
    Brown noise is deeper and more natural than white noise for room hiss.
    """
    import soundfile as sf

    config.custom_print("Lifespan", f"Generating {duration_s:.0f}s brown-noise room tone → {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    num_samples = int(duration_s * sample_rate)

    # Brown noise: integrated white noise
    white = np.random.randn(num_samples)
    brown = np.cumsum(white)
    # Normalize and high-pass slightly to remove DC drift
    brown = brown - np.mean(brown)
    brown = np.diff(brown, prepend=brown[0])
    brown = brown / (np.max(np.abs(brown)) + 1e-9)

    # Fade in/out for seamless looping
    fade_samples = int(0.5 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    brown[:fade_samples] *= fade_in
    brown[-fade_samples:] *= fade_out

    # Reduce to very low level (room tone should be subtle)
    brown *= 0.015  # ~-36 dBFS

    sf.write(path, brown, sample_rate, subtype='PCM_16')
    config.custom_print("Lifespan", f"Room tone saved: {path}")
