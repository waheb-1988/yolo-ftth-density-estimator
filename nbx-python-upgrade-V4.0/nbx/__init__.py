"""NBX: agentic Next-Best-Action / Next-Best-Offer engine."""
import os

# V4.0 local only: no usage analytics leave the machine. Set before gradio / huggingface / vllm are imported
# (every entry point imports `nbx` first). setdefault: an explicit shell variable still wins.
for _k, _v in {"GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
               "VLLM_NO_USAGE_STATS": "1", "ANONYMIZED_TELEMETRY": "False"}.items():
    os.environ.setdefault(_k, _v)

# vowpalwabbit must be loaded before pandas / pyarrow: the other order
# segfaults on Windows (conflicting native libraries, seen with
# vowpalwabbit 9.11.6 + pyarrow 17). Every entry point imports `nbx` first,
# so doing it here fixes the order for the API, UI, CLI and tests.
try:
    import vowpalwabbit  # noqa: F401
except ImportError:  # pragma: no cover - optional outside the NBA modules
    pass

__version__ = "4.0.0"
