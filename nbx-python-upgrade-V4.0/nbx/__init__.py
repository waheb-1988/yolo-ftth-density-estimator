"""NBX: agentic Next-Best-Action / Next-Best-Offer engine."""
import os

# V4.0 local only: no usage analytics leave the machine. Set before gradio / huggingface / vllm are imported
# (every entry point imports `nbx` first). setdefault: an explicit shell variable still wins.
for _k, _v in {"GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
               "VLLM_NO_USAGE_STATS": "1", "ANONYMIZED_TELEMETRY": "False"}.items():
    os.environ.setdefault(_k, _v)

# vowpalwabbit (the V1.0 bandit only) must be loaded BEFORE pandas / pyarrow / matplotlib on Windows: loaded
# after them it crashes the process (two copies of the C++ runtime msvcp140.dll). The V4.0 training / scoring
# pipeline never uses it. So it is preloaded here only when that is still safe: in a fresh process (CLI, API, UI,
# tests) where none of those libraries is loaded yet, and never inside a Jupyter kernel (the kernel already holds
# its own DLLs; the notebooks do not need it). FSDR_PRELOAD_VW=false switches the preload off everywhere.
import sys as _sys

_late = any(m in _sys.modules for m in ("pandas", "pyarrow", "matplotlib", "ipykernel"))
if os.getenv("FSDR_PRELOAD_VW", "true").lower() == "true" and not _late:
    try:
        import vowpalwabbit  # noqa: F401
    except ImportError:  # pragma: no cover - optional outside the NBA modules
        pass

__version__ = "4.0.0"
