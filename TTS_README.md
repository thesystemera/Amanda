# Orpheus TTS Test Suite for PLAiR

Comprehensive testing environment for evaluating [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) as a local replacement for ElevenLabs in the PLAiR project.

## Overview

**Why Orpheus TTS?**
- ✅ Native emotional control with inline tags (`<laugh>`, `<sigh>`, etc.)
- ✅ Natural filler words ("uhm", "uh") without hacking
- ✅ Zero-shot voice cloning from 10-30s samples
- ✅ Apache 2.0 license (full commercial freedom)
- ✅ 24kHz output quality
- ✅ ~200ms time-to-first-audio (streaming)

**Target Hardware:**
- GPU: NVIDIA P6000 (24GB VRAM)
- TTS Allocation: ~8GB VRAM
- Model: Orpheus 1B (sweet spot for quality/resources)

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run setup
python setup.py
```

### 2. Quick Test

```bash
# Basic synthesis
python quick_test.py

# With emotion
python quick_test.py --emotion laugh --text "Hello! <laugh> Nice to meet you!"

# Different voice
python quick_test.py --voice leo --text "Testing a male voice."
```

### 3. Full Test Suite

```bash
# Run all tests
python orpheus_tts_test.py

# With custom settings
python orpheus_tts_test.py --model canopylabs/orpheus-3b --max-vram 12

# With voice cloning (provide reference audio)
python orpheus_tts_test.py --voice-clone-ref ./my_voice_sample.wav
```

## Project Structure

```
TTS_Local/
├── requirements.txt          # Python dependencies
├── setup.py                  # Setup and installation helper
├── quick_test.py             # Quick single test script
├── orpheus_tts_test.py       # Comprehensive test suite
├── README.md                 # This file
├── test_outputs/             # Generated audio files
│   ├── basic_1_tara.wav
│   ├── emotion_laugh_tara.wav
│   ├── voice_tara.wav
│   └── test_report.json
└── .venv/                    # Virtual environment
```

## Available Emotion Tags

Orpheus supports 8 built-in emotion tags:

| Tag | Example Usage |
|-----|---------------|
| `<laugh>` | "That's hilarious! <laugh> I can't believe it!" |
| `<chuckle>` | "Oh, you. <chuckle> Always the joker." |
| `<sigh>` | "I guess so. <sigh> Whatever you say." |
| `<gasp>` | "Look out! <gasp> That was close!" |
| `<cough>` | "Excuse me. <cough> Allergies." |
| `<sniffle>` | "It's so sad. <sniffle> Poor thing." |
| `<groan>` | "Monday again. <groan> I need coffee." |
| `<yawn>` | "It's late. <yawn> Time for bed." |

## Available Voices

8 built-in voices (all rated for naturalness):

- **tara** - Most natural (recommended default)
- **leah** - Warm, friendly
- **jess** - Energetic
- **leo** - Male, professional
- **dan** - Male, casual
- **mia** - Soft, gentle
- **zac** - Male, young
- **zoe** - Female, expressive

## Model Options

| Model | VRAM | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| `canopylabs/orpheus-400m` | ~3-4GB | Good | Fastest | Low-resource, edge devices |
| `canopylabs/orpheus-1b` | ~6-8GB | Excellent | Fast | **Recommended for PLAiR** |
| `canopylabs/orpheus-3b` | ~10-14GB | Superior | Moderate | Maximum quality |

## Inference Backends

| Backend | Speed | VRAM | Best For |
|---------|-------|------|----------|
| **vLLM** | Fastest | Higher | Production, high throughput |
| **Transformers** | Fast | Medium | Development, flexibility |
| **llama.cpp (GGUF)** | Moderate | Lowest | CPU fallback, edge |

### GGUF Quantization (for llama.cpp)

| Quant | Tokens/sec | VRAM | Quality |
|-------|------------|------|---------|
| Q4_K_M | ~129 | ~4GB | Excellent |
| Q5_K_M | ~100 | ~5GB | Superior |
| Q8_0 | ~70 | ~7GB | Near-lossless |

## Test Coverage

The comprehensive test suite (`orpheus_tts_test.py`) covers:

1. **Basic Synthesis** - Plain text without emotions
2. **Emotion Tags** - All 8 built-in emotions
3. **Conversational Flow** - Natural fillers + emotions
4. **Voice Comparison** - All 8 voices
5. **Voice Cloning** - Zero-shot from reference audio
6. **Latency Benchmark** - RTF measurements
7. **VRAM Constraints** - Memory usage validation

## Expected Performance

On P6000 (24GB VRAM):

| Metric | Expected | Target |
|--------|----------|--------|
| Time to First Audio | ~200ms | <300ms ✅ |
| RTF (Real-Time Factor) | ~0.7-1.0 | <1.0 ✅ |
| Tokens/sec | ~80-130 | >50 ✅ |
| VRAM Usage | ~6-8GB | <8GB ✅ |
| Output Quality | 24kHz | 24kHz ✅ |

## Troubleshooting

### CUDA Out of Memory

```python
# Use smaller model
python orpheus_tts_test.py --model canopylabs/orpheus-400m

# Or enable CPU offloading (slower)
# Edit script: device_map="auto" with max_memory settings
```

### Slow Inference

```python
# Use vLLM backend for production
# pip install vllm

# Or use GGUF quantized model with llama.cpp
```

### Audio Quality Issues

- Ensure SNAC is installed: `pip install snac`
- Check reference audio for voice cloning (10-30s, clear speech)
- Try different voices (tara is most natural)

## Integration with PLAiR

Example integration pattern:

```python
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf

class PLAiRTTS:
    def __init__(self, model_name="canopylabs/orpheus-1b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def speak(self, text: str, voice: str = "tara", emotion: str = None):
        # Add emotion tag if specified
        if emotion:
            text = f"{text} <{emotion}>"
        
        # Format prompt
        prompt = f"{voice}|{text}"
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=2048)
        
        # Decode to audio (requires SNAC)
        audio = self._decode_snac(outputs)
        return audio
```

## License

Orpheus TTS is licensed under Apache 2.0, allowing full commercial use in PLAiR.

## Resources

- [Orpheus TTS GitHub](https://github.com/canopyai/Orpheus-TTS)
- [Hugging Face Model](https://huggingface.co/canopylabs/orpheus-1b)
- [SNAC Decoder](https://github.com/hubertsiuzdak/snac)

## Support

For issues with this test suite, check:
1. `test_outputs/test_report.json` for detailed results
2. VRAM usage during testing
3. Model download status in HuggingFace cache
