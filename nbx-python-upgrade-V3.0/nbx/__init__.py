"""NBX: agentic Next-Best-Action / Next-Best-Offer engine."""

# vowpalwabbit must be loaded before pandas / pyarrow: the other order
# segfaults on Windows (conflicting native libraries, seen with
# vowpalwabbit 9.11.6 + pyarrow 17). Every entry point imports `nbx` first,
# so doing it here fixes the order for the API, UI, CLI and tests.
try:
    import vowpalwabbit  # noqa: F401
except ImportError:  # pragma: no cover - optional outside the NBA modules
    pass

__version__ = "1.0.0"
