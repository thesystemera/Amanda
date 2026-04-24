# Local LLM Guide — P6000 (24GB VRAM)

> Reference for running local inference on a second Quadro P6000 alongside the main Amanda stack.
> Date: April 2026

---

## Hardware Reality Check

The P6000 hasn't changed. Plan around its fixed profile:

| Spec | Implication |
|------|-------------|
| **24GB GDDR5X** | Can fit large quantized models. Sweet spot is **30B-34B at Q4_K_M** or **13B-14B at Q5_K_M**. |
| **Pascal / CC 6.1** | **No tensor cores.** FP16/INT8 are emulated and slower than RTX. Stick to **GGUF Q4/Q5** via llama.cpp. |
| **3840 CUDA cores** | Expect roughly **1/3 to 1/2 the tok/sec** of a modern RTX 4090 for the same model. |

---

## Model Sizing Table

Look for the best open-weights release (Llama 4, Qwen 3, DeepSeek v4, Mistral Large 3, etc.) and fit it into this frame:

| Size | Quantization | VRAM | Quality | Speed (Pascal, est.) |
|------|-------------|------|---------|---------------------|
| **7B-8B** | Q6_K / Q8_0 | ~6-8GB | Excellent | 30-50 tok/s |
| **13B-14B** | Q5_K_M | ~10GB | Excellent | 20-35 tok/s |
| **30B-34B** | Q4_K_M | ~20-22GB | Very Good | 10-18 tok/s |
| **70B** | IQ4_XS / Q3_K_M | ~23-24GB | Good | 5-10 tok/s |

**Recommendation:**
- **Tag suggestion / lightweight tasks:** 13B-14B at Q5_K_M is overkill and fast.
- **General intelligent assistant:** 30B-34B at Q4_K_M is the 24GB sweet spot.
- Avoid cramming a 70B at Q3 unless you specifically need the parameter count — quality per GB is worse than a roomy 30B.

---

## Hosting Options

The stack is stable. Use whichever wrapper you prefer:

| Tool | Best For | Notes |
|------|----------|-------|
| **llama.cpp server** | Raw speed on Pascal | Native GGUF, best Pascal performance |
| **Ollama** | Drop-in OpenAI API compatibility | `http://localhost:11434/v1` |
| **LM Studio** | GUI + server mode | Good for experimenting |
| **vLLM** | High-throughput batching | Weaker Pascal support than llama.cpp |

### Example: pointing meta_data_editor at local inference

```python
from openai import OpenAI

# Ollama
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# llama.cpp server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")
```

---

## Recommended Architecture

Split workloads across GPUs so the P6000 is a dedicated inference worker:

```
Main GPU  → Orpheus TTS (llama.cpp) + Amanda runtime
P6000     → Local LLM server (llama.cpp / Ollama / LM Studio)
```

This keeps TTS latency unaffected while offloading all LLM inference to the second card.

---

## Quick-Start (llama.cpp server)

```bash
# Download a 30B-class GGUF and serve it
./server -m ./models/<model>-30b-Q4_K_M.gguf \
         --host 0.0.0.0 \
         --port 8080 \
         -ngl 999 \
         --ctx-size 8192
```

Then set `meta_data_editor.py` or any Amanda service to hit `http://localhost:8080/v1`.

---

## Integration TODO

- [ ] Make `meta_data_editor.py` backend configurable (Gemini cloud vs. local endpoint)
- [ ] Add `LOCAL_LLM_URL` and `LOCAL_LLM_MODEL` to `config.py`
- [ ] Benchmark tok/sec on the P6000 for the chosen model
