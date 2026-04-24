"""
Validation script for Priority 3 performance optimizations.
Run with: .venv\Scripts\python.exe validate_priority3.py
"""
import time
import io
import os
import tempfile
import numpy as np
import re

print("=" * 60)
print("PRIORITY 3 VALIDATION")
print("=" * 60)

# ── 1. PANNs tag-list rebuild ──────────────────────────────────────────────
print("\n1. PANNs: list-comprehension inside label loop vs precomputed set")

SPEECH_DETECTION_TAGS = {
    'speech': True,
    'male speech, man speaking': True,
    'female speech, woman speaking': True,
    'child speech, kid speaking': True,
    'conversation': True,
    'narration, monologue': True,
}
labels = [f"label_{i}" for i in range(527)]  # ~527 labels like PANNs

# Current approach
t0 = time.perf_counter()
for _ in range(1000):
    for label in labels:
        label_lower = label.lower()
        _ = label_lower in [tag.lower() for tag in SPEECH_DETECTION_TAGS]
t_old = time.perf_counter() - t0

# Optimized approach
SPEECH_SET = {tag.lower() for tag in SPEECH_DETECTION_TAGS}
t0 = time.perf_counter()
for _ in range(1000):
    for label in labels:
        label_lower = label.lower()
        _ = label_lower in SPEECH_SET
t_new = time.perf_counter() - t0

print(f"   Current (list comp):  {t_old*1000:.2f} ms")
print(f"   Precomputed set:      {t_new*1000:.2f} ms")
print(f"   Speedup:              {t_old/t_new:.1f}x")
print(f"   VALID:                {'PASS' if t_new < t_old else 'FAIL'}")

# ── 2. PCM accumulator copy on event loop ──────────────────────────────────
print("\n2. TTS: bytes() copy vs memoryview() overhead")

# Simulate a 10-second utterance @ 24kHz mono int16 = ~480 KB
pcm = bytearray(b'\x00\x01' * (24000 * 10))

t0 = time.perf_counter()
for _ in range(10000):
    _ = bytes(pcm)
t_old = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(10000):
    _ = memoryview(pcm)
t_new = time.perf_counter() - t0

print(f"   bytes(pcm):           {t_old*1000:.2f} ms (10k iterations)")
print(f"   memoryview(pcm):      {t_new*1000:.2f} ms (10k iterations)")
print(f"   Speedup:              {t_old/t_new:.1f}x")
print(f"   VALID:                {'PASS' if t_new < t_old else 'FAIL'}")

# ── 3. Mutagen double-write ────────────────────────────────────────────────
print("\n3. Mutagen: verify double-write & test in-memory tagging")

from mutagen.mp3 import MP3
from mutagen.id3 import TIT2, TIT3

# Create dummy MP3 bytes using a minimal valid MP3 frame
# (sync word + MPEG1/L3/44100/stereo/frame length = 417 bytes of zeros)
mp3_frame = b'\xff\xfb\x90\x00' + b'\x00' * 413
mp3_data = mp3_frame * 10

with tempfile.TemporaryDirectory() as tmpdir:
    fname = os.path.join(tmpdir, "test.mp3")

    # Current approach: write, then reload, tag, save (second write)
    t0 = time.perf_counter()
    with open(fname, 'wb') as f:
        f.write(mp3_data)
    audio = MP3(fname)
    if audio.tags is None:
        audio.add_tags()
    audio.tags.add(TIT2(encoding=3, text="title"))
    audio.tags.add(TIT3(encoding=3, text="subtitle"))
    audio.save()
    t_old = time.perf_counter() - t0
    old_size = os.path.getsize(fname)

    # Proposed approach: tag in-memory, write once
    t0 = time.perf_counter()
    buf = io.BytesIO(mp3_data)
    audio2 = MP3(buf)
    if audio2.tags is None:
        audio2.add_tags()
    audio2.tags.add(TIT2(encoding=3, text="title"))
    audio2.tags.add(TIT3(encoding=3, text="subtitle"))
    audio2.save(buf)
    with open(fname, 'wb') as f:
        f.write(buf.getvalue())
    t_new = time.perf_counter() - t0
    new_size = os.path.getsize(fname)

    print(f"   Current (write+tag+rewrite):  {t_old*1000:.3f} ms  size={old_size}")
    print(f"   Proposed (tag-mem+write-once): {t_new*1000:.3f} ms  size={new_size}")
    print(f"   Speedup:                      {t_old/t_new:.1f}x")
    print(f"   VALID:                        {'PASS' if t_new < t_old else 'FAIL'}")

# ── 4. pygame decode array copies ──────────────────────────────────────────
print("\n4. AudioService: pygame decode copy overhead")

import pygame
pygame.mixer.init(frequency=24000, size=-16, channels=1)

# Create a tiny silent MP3 via lameenc for realistic test
import lameenc
enc = lameenc.Encoder()
enc.set_bit_rate(128)
enc.set_in_sample_rate(24000)
enc.set_channels(1)
enc.set_quality(2)
silent_pcm = b'\x00\x00' * 24000  # 1 second
mp3_bytes = bytes(enc.encode(silent_pcm)) + bytes(enc.flush())

with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
    f.write(mp3_bytes)
    tmp_mp3 = f.name

t0 = time.perf_counter()
for _ in range(100):
    sound = pygame.mixer.Sound(tmp_mp3)
    raw = pygame.sndarray.array(sound)
    if len(raw.shape) > 1:
        raw = raw.mean(axis=1).astype(np.int16)
    result = raw.tobytes()
t_old = time.perf_counter() - t0

# Proposed: pydub-style direct decode (if available)
try:
    from pydub import AudioSegment
    t0 = time.perf_counter()
    for _ in range(100):
        seg = AudioSegment.from_mp3(tmp_mp3)
        # Ensure mono, 16-bit
        if seg.channels > 1:
            seg = seg.set_channels(1)
        result2 = seg.raw_data
    t_new = time.perf_counter() - t0
    has_pydub = True
except ImportError:
    t_new = None
    has_pydub = False

os.unlink(tmp_mp3)

print(f"   Current (pygame path):        {t_old*1000:.2f} ms (100 iterations)")
if has_pydub:
    print(f"   Proposed (pydub path):        {t_new*1000:.2f} ms (100 iterations)")
    print(f"   Speedup:                      {t_old/t_new:.1f}x")
    print(f"   VALID:                        {'PASS' if t_new < t_old else 'FAIL'}")
else:
    print("   pydub not installed — skipping proposed path benchmark")
    print("   NOTE: Current path does 3-4 intermediate array allocations.")

# ── 5. Visualizer FFT bin precomputation ───────────────────────────────────
print("\n5. Visualizer: FFT bin search every frame vs precomputed")

frequency_bands = [(20, 200), (200, 500), (500, 1000), (1000, 2000),
                   (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000)]
freqs = np.fft.rfftfreq(1024, 1 / 24000)

t0 = time.perf_counter()
for _ in range(10000):
    for low, high in frequency_bands:
        relevant = np.where((freqs >= low) & (freqs < high))[0]
t_old = time.perf_counter() - t0

# Precompute
precomputed = [np.where((freqs >= low) & (freqs < high))[0] for low, high in frequency_bands]
t0 = time.perf_counter()
for _ in range(10000):
    for relevant in precomputed:
        _ = relevant
t_new = time.perf_counter() - t0

print(f"   On-the-fly np.where:    {t_old*1000:.2f} ms (10k frames)")
print(f"   Precomputed indices:    {t_new*1000:.2f} ms (10k frames)")
print(f"   Speedup:                {t_old/t_new:.1f}x")
print(f"   VALID:                  {'PASS' if t_new < t_old else 'FAIL'}")

# ── 6. Regex recompilation per chunk ───────────────────────────────────────
print("\n6. Sentence processor: compile regex once vs per-chunk")

test_buf = "This is a test [noise] sentence. " * 50

t0 = time.perf_counter()
for _ in range(100000):
    _ = re.sub(r'\[.*?]', '', test_buf)
t_old = time.perf_counter() - t0

_strip_brackets = re.compile(r'\[.*?]')
t0 = time.perf_counter()
for _ in range(100000):
    _ = _strip_brackets.sub('', test_buf)
t_new = time.perf_counter() - t0

print(f"   re.sub raw string:      {t_old*1000:.2f} ms (100k iterations)")
print(f"   compiled regex.sub:     {t_new*1000:.2f} ms (100k iterations)")
print(f"   Speedup:                {t_old/t_new:.1f}x")
print(f"   VALID:                  {'PASS' if t_new < t_old else 'FAIL'}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
