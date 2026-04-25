import os
import datetime
import torch
import sounddevice as sd
import api_secrets
from colorama import Fore, Back, Style
from threading import Lock

GEMINI_API_KEY = api_secrets.GEMINI_API_KEY

DATA_DIR = "data"
VECTOR_DIR = os.path.join(DATA_DIR, "vector")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
LAST_STATE_PATH = "last_state.json"

DOMAINS = ['tts', 'impulse', 'meta', 'interrupt', 'interject']
INSTANT_DOMAINS = ['impulse', 'interrupt', 'interject']
AUDIO_SUBDIRS = DOMAINS + ['prelude', 'breath', 'interlude']
RANDOM_AUDIO_SUBDIRS = ['prelude', 'breath', 'interlude']

AUDIO_DIRS = {name: os.path.join(AUDIO_DIR, name) for name in AUDIO_SUBDIRS}

for d in [DATA_DIR, VECTOR_DIR, AUDIO_DIR, *AUDIO_DIRS.values()]:
    os.makedirs(d, exist_ok=True)

def get_default_device_settings():
    try:
        default_input_device_info = sd.query_devices(kind='input')
        default_output_device_info = sd.query_devices(kind='output')
        return int(default_input_device_info['default_samplerate']), int(default_output_device_info['default_samplerate'])
    except:
        return 16000, 48000

FS, FS_OUTPUT = get_default_device_settings()

MAX_CHAT_LOG_ENTRIES = 5
INTERJECTION_COOLDOWN = 2.0
COOLDOWN_DURATION = 120
SHOTGUN_COOLDOWN = 300  # seconds (5 min); universal cooldown to prevent repeat plays across all domains

# ── Spatial Audio Effects (ported from AI_RADIO) ───────────────────────────
SPATIAL_AUDIO_ENABLED = True
SPATIAL_AUDIO_CONFIG = {
    # Physical model: fixed mic in a fixed room.
    # As source moves away, direct sound drops (inverse square).
    # Room reverb stays constant — perceived wetness increases because
    # direct gets quieter, not because reverb gets louder.
    'reverb': {
        'room_size': 0.35,
        'damping': 0.65,
        'wet_level': 0.05,  # constant room reverb
        'dry_near': 1.0,    # 1cm from mic: strong direct
        'dry_far': 0.15,    # 1m away: direct fades, reverb dominates
    },
    # Compressor: mic/preamp saturation when close-talking
    # Close = heavy squash (warm, flat, "radio voice")
    # Far   = natural dynamics (breathy, open)
    'compressor': {
        'threshold_near': -22,   # close: catches almost everything
        'threshold_far': 0,      # far: barely touches
        'ratio_near': 4.0,       # close: heavy squash
        'ratio_far': 1.1,        # far: gentle
        'attack_ms': 2.0,
        'release_ms': 50.0,
    },
    # EQ: proximity effect + subtle air absorption
    # Everything morphs with the mic proximity spline — gains AND frequencies
    'eq': {
        'high_cut_near': 0,           # close: flat
        'high_cut_far': -3,           # far: slight air loss
        'high_cut_freq_near': 5000,   # close: shelf starts lower (gentler)
        'high_cut_freq_far': 8000,    # far: shelf starts higher (airier)
        'low_boost_near': 6,          # close: proximity effect
        'low_boost_far': 0,           # far: no boost
        'low_boost_freq_near': 80,    # close: deep sub-bass boost
        'low_boost_freq_far': 250,    # far: broader low-mid warmth
    },
}
SPATIAL_SPEAKER_PAN = {
    'tara': -0.05,
    'leah': 0.05,
    'jess': -0.08,
    'leo': 0.08,
    'dan': 0.0,
    'mia': -0.03,
    'zac': 0.06,
    'zoe': -0.06,
}
ROOM_TONE_PATH = os.path.join(AUDIO_DIR, "room_tone.mp3")
DEFAULT_MIC_PROXIMITY = 0.3  # 0=close/dry, 1=far/wet
TTS_TOKENS_PER_SECOND = 130   # rough heuristic: max_tokens / audio_seconds, calibrate over time

TTS_SIMILARITY_THRESHOLD = 0.75
CACHE_SIMILARITY_THRESHOLD = 0.60

CLASSIFICATION_SPEECH_DETECTION_THRESHOLD = 0.1
CLASSIFICATION_OTHER_AUDIO_THRESHOLD = 0.2
SPEECH_DETECTION_DECAY_FACTOR = 0.5
OTHER_AUDIO_DECAY_FACTOR = 0.75

AUDIO_CLASSIFICATION_INTERVAL_S = 4.0
INTERLUDE_MONITOR_INTERVAL_S = 1.5

ADVANCED_GPT_TEMPERATURE = 0.7
MOOD_GPT_TEMPERATURE = 0.2
REALTIME_GPT_GEMINI_TEMPERATURE = 0.7

GEMINI_TASK_MODEL_NAME = "gemini-3.1-flash-lite-preview"
GEMINI_CHAT_MODEL_NAME = "gemini-3.1-flash-lite-preview"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

TTS_SERVER_URL = "http://localhost:8080"
TTS_VOICE = "tara"
TTS_SAMPLE_RATE = 24000
MP3_BITRATE = 128
# Unified audio chunk size (mono int16 bytes). This drives slice resolution.
# 1024 bytes @ 24kHz mono = 512 frames = ~21ms → ~47 slices/sec
# Bump to 2048 if CPU struggles. Must be divisible by 4.
AUDIO_CHUNK_SIZE = 512
SILENCE_CACHE_DURATION_S = 0.1
GENERATE_TTS = True

# ── Orpheus / llama.cpp tuning ─────────────────────────────────────────────
TTS_QUALITY = "AVERAGE"           # "FASTEST" | "AVERAGE" | "BEST"
TTS_NUM_WORKERS = 2                 # parallel OrpheusCpp engines (tune to VRAM)
TTS_REPEAT_PENALTY = 1.2            # token-level repetition deterrence (1.0 = off)
TTS_FREQUENCY_PENALTY = 0.1         # whole-generation frequency penalty (0.0 = off)
TTS_MIN_P = 0.1                     # min-p sampler (0.05 = default, lower = more conservative)
TTS_TOP_P = 0.9                     # top-p nucleus sampling
TTS_N_CTX = 4096                    # KV cache cap (lower = less VRAM; 2048 also safe for short sentences)
TTS_N_BATCH = 128                   # compute scratch buffer size (lower = less VRAM)
TTS_N_UBATCH = 128                  # micro-batch size (should match n_batch for single-seq)
GENERATE_IMPULSE_FOR_CACHE = True
GENERATE_META_FOR_CACHE = True
GENERATE_INTERJECT_FOR_CACHE = True

DISABLE_EXTRACT_BACKSTORY_AND_NOTES = False
DISABLE_META_TAG_VECTOR_MATCHING_FOR_IMPULSE = True
PLAY_RANDOM_SOUND_WHEN_IDLE = True

EMOTIONAL_STATES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'contemptuous']
EMOTIONAL_STATE_COLORS = {
    'angry': (255, 0, 0), 'disgusted': (102, 51, 0), 'fearful': (0, 128, 255),
    'happy': (255, 255, 128), 'neutral': (255, 255, 255), 'sad': (0, 0, 255),
    'surprised': (255, 165, 0), 'contemptuous': (153, 0, 76)
}

MIN_BUFFER_DURATION_MS = 250
MAX_BUFFER_DURATION_MS = 5000
VAD_SPEECH_THRESHOLD = 0.4
MIN_FRAME_RMS = 0.003
MIN_CHUNK_RMS = 0.005
MIN_SPEECH_DURATION_MS = 150

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

header_colors = {
    "Lifespan":             Fore.LIGHTWHITE_EX + Back.BLUE + Style.BRIGHT,
    "Info":                 Fore.YELLOW,
    "Error":                Fore.RED + Style.BRIGHT,
    "Heard":                Fore.LIGHTWHITE_EX + Back.MAGENTA + Style.BRIGHT,
    "Play":                 Fore.LIGHTGREEN_EX,
    "Miss":                 Fore.LIGHTRED_EX,
    "Stream":               Fore.LIGHTBLUE_EX,
    "Cache":                Fore.LIGHTMAGENTA_EX,
    "TTS-Perf":             Fore.BLACK + Back.LIGHTYELLOW_EX + Style.BRIGHT,
    "Perf":                 Fore.BLACK + Back.LIGHTGREEN_EX + Style.BRIGHT,
    "GPT":                  Fore.CYAN + Style.BRIGHT,
    "VAD":                  Fore.LIGHTBLACK_EX,
    "Audio Classification": Fore.GREEN,
    "Mood":                 Fore.LIGHTCYAN_EX,
    "Prompt":               Fore.CYAN,
    "System":               Fore.LIGHTGREEN_EX,
    "User":                 Fore.LIGHTYELLOW_EX,
    "Output":               Fore.LIGHTMAGENTA_EX,
    "Sentence Splitter":    Fore.LIGHTBLUE_EX,
    "Persona & Profile":    Fore.GREEN,
    "Vector":               Fore.LIGHTWHITE_EX + Back.LIGHTBLUE_EX + Style.BRIGHT,
}
header_enabled = {
    "Lifespan":             True,
    "Info":                 True,
    "Error":                True,
    "Heard":                True,
    "Play":                 True,
    "Miss":                 True,
    "Stream":               True,
    "Cache":                True,
    "TTS-Perf":             True,
    "Perf":                 True,
    "GPT":                  True,
    "VAD":                  True,
    "Audio Classification": False,
    "Mood":                 False,
    "Prompt":               True,
    "System":               False,
    "User":                 False,
    "Output":               True,
    "Sentence Splitter":    True,
    "Persona & Profile":    False,
    "Vector":               True,
}
print_lock = Lock()

def custom_print(header=None, message=None):
    if header is not None and header in header_enabled and not header_enabled[header]:
        return
    with print_lock:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ts_str = f"{Fore.LIGHTBLACK_EX}{ts}{Style.RESET_ALL} "
        if header and message is not None:
            color = header_colors.get(header, Fore.WHITE)
            print(f"{ts_str}{color}{header}: {Style.RESET_ALL}{message}")
        else:
            print(f"{ts_str}{message or header}")