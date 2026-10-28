"""LangSmith tracing of the FSDR agents and pipelines (optional).

Off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set; then every LLM call, tool call, agent, decision,
journey step and weekly pipeline step is a run in the LangSmith project LANGSMITH_PROJECT (default
"fsdr-recommender"), nested as they happen:

    fsdr_weekly ── fsdr_weekly:train / score / offers ── fsdr_batch ── fsdr_decision ── Best Action agent ── llm / tool
    fsdr_journey:<step> ── fsdr_decision stage ── agent ── llm / tool

Data protection: traces leave the machine. The MSISDN never appears (prompts carry the salted pseudonym of
nbx/core/privacy.py, spans tag `customer_ref`), but tool outputs hold the customer's features (scores, revenue,
plan, nationality group). For real customer data use a self-hosted LangSmith (LANGSMITH_ENDPOINT), or set
LANGSMITH_HIDE_INPUTS=true / LANGSMITH_HIDE_OUTPUTS=true to send only the structure, timings and errors.

Tracing never changes a decision and never fails a run: any LangSmith error is logged and ignored.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any

log = logging.getLogger(__name__)
MAX_CHARS = 100_000                      # per inputs / outputs payload


def enabled() -> bool:
    if os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower() not in ("true", "1"):
        return False
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return False
    try:
        import langsmith  # noqa: F401
    except ImportError:
        return False
    return True


def safe(obj: Any) -> dict:
    """JSON-safe, size-capped payload (LangSmith wants a dict)."""
    if obj is None:
        return {}
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = json.dumps(str(obj))
    if len(s) > MAX_CHARS:
        return {"truncated": True, "chars": len(s), "head": s[:MAX_CHARS]}
    out = json.loads(s)
    return out if isinstance(out, dict) else {"value": out}


class _NoSpan:
    def end(self, outputs: Any = None, error: str | None = None) -> None:
        pass


class _Span:
    def __init__(self, run):
        self.run = run

    def end(self, outputs: Any = None, error: str | None = None) -> None:
        try:
            self.run.end(outputs=safe(outputs), error=error)
        except Exception as e:  # noqa: BLE001  (tracing must never break a run)
            log.debug("langsmith end failed: %s", e)


@contextmanager
def span(name: str, run_type: str = "chain", inputs: Any = None, metadata: dict | None = None, tags: list[str] | None = None):
    """A LangSmith run around a block; yields an object with end(outputs). A no-op when tracing is off."""
    if not enabled():
        yield _NoSpan()
        return
    try:
        from langsmith.run_helpers import trace

        cm = trace(name=name, run_type=run_type, inputs=safe(inputs), tags=["fsdr", *(tags or [])],
                   metadata={"app": "fsdr-recommender", **(metadata or {})},
                   project_name=os.getenv("LANGSMITH_PROJECT", "fsdr-recommender"))
        run = cm.__enter__()
    except Exception as e:  # noqa: BLE001
        log.warning("langsmith tracing unavailable (%s): continuing without it", e)
        yield _NoSpan()
        return
    exc_info = (None, None, None)
    try:
        yield _Span(run)
    except BaseException as e:
        exc_info = (type(e), e, e.__traceback__)
        raise
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception as e:  # noqa: BLE001
            log.debug("langsmith exit failed: %s", e)


def traced_pipeline(name: str):
    """Decorator for a pipeline step returning a report dict: inputs = the run folder, outputs = the report's
    scalar fields (counts, versions, timings -- no customer rows)."""
    def deco(fn):
        @wraps(fn)
        def wrapper(data_dir, *args, **kwargs):
            if not enabled():
                return fn(data_dir, *args, **kwargs)
            with span(name, "chain", inputs={"run": str(getattr(data_dir, "name", data_dir)).split("\\")[-1].split("/")[-1],
                                              **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}}) as sp:
                rep = fn(data_dir, *args, **kwargs)
                sp.end(outputs={k: v for k, v in (rep or {}).items() if isinstance(v, (str, int, float, bool)) or k in
                                ("evaluation", "labels", "promotion_rule", "drift", "arms")})
                return rep
        return wrapper
    return deco
