import os
import sys

_torch_lib = os.path.abspath(
    os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib")
)
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(_torch_lib)

import llama_cpp as _llama_cpp
import config as _config

_orig_llama_init = _llama_cpp.Llama.__init__

def _patched_llama_init(self, *args, **kwargs):
    kwargs.setdefault("flash_attn", True)
    # Cap n_ctx to keep VRAM bounded; orpheus_cpp passes 0 which can mean 128K.
    if kwargs.get("n_ctx") == 0:
        kwargs["n_ctx"] = _config.TTS_N_CTX
    kwargs.setdefault("n_batch", _config.TTS_N_BATCH)
    kwargs.setdefault("n_ubatch", _config.TTS_N_UBATCH)
    model_path = kwargs.get("model_path", "?")
    print(f"  [bootstrap] Loading model: {model_path} | n_ctx={kwargs.get('n_ctx')} n_batch={kwargs.get('n_batch')}")
    return _orig_llama_init(self, *args, **kwargs)

_llama_cpp.Llama.__init__ = _patched_llama_init

_orig_llama_call = _llama_cpp.Llama.__call__

def _patched_llama_call(self, *args, **kwargs):
    kwargs.setdefault("repeat_penalty", _config.TTS_REPEAT_PENALTY)
    if _config.TTS_FREQUENCY_PENALTY != 0.0:
        kwargs.setdefault("frequency_penalty", _config.TTS_FREQUENCY_PENALTY)
    return _orig_llama_call(self, *args, **kwargs)

_llama_cpp.Llama.__call__ = _patched_llama_call
