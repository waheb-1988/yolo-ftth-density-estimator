"""Tool base of the agents: a named function the ReAct loop can call, and the error that goes back to the
model as an observation (instead of crashing the run). The FSDR tools are in nbx/fsdr/agents/tools.py and
nbx/fsdr/agents/ops_agents.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ToolError(Exception):
    """Raised for bad arguments; the message goes back to the agent as an observation so it can correct
    itself instead of the run failing."""


@dataclass
class Tool:
    name: str
    description: str
    args: str  # human/LLM-readable signature, e.g. '{"offers": [codes]}'
    fn: Callable[[Any, dict], dict]


def call_tool(ctx: Any, name: str, args: dict, registry: dict[str, Tool]) -> dict:
    from nbx.core import tracing
    from nbx.core.privacy import pseudonym

    with tracing.span(f"tool:{name}", "tool", inputs={"args": args or {}},
                      metadata={"customer_ref": pseudonym(getattr(ctx, "msisdn", None))}) as sp:
        out = _call_tool(ctx, name, args, registry)
        sp.end(outputs=out, error=out.get("error") if isinstance(out, dict) else None)
        return out


def _call_tool(ctx: Any, name: str, args: dict, registry: dict[str, Tool]) -> dict:
    tool = registry[name]
    try:
        return tool.fn(ctx, args or {})
    except ToolError as e:
        return {"error": str(e)}
    except (TypeError, ValueError) as e:
        return {"error": f"bad arguments for {name}: {e}. Expected {tool.args}"}
