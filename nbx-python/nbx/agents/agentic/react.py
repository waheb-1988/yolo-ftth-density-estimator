"""Policy-enforced ReAct loop for small local models.

The model chooses every step (which tool, with what arguments, when to
stop), but the controller -- not the prompt -- enforces the policy:
  - one JSON object per step, either {"thought","tool","args"} or
    {"thought","final"}; anything else goes back as an error observation;
  - unknown tools / bad arguments come back as observations, so the model
    can correct itself instead of the run crashing;
  - required evidence: an agent may not finalize before it has called its
    mandatory tools (e.g. the NBA strategist must look at risk scores and
    the bandit before choosing an action);
  - the final answer is schema-validated by the agent's own validator;
  - an identical repeated call is refused with an explicit next step; a
    second repeat ends the agent (loop guard);
  - a step budget, after which the caller finalizes deterministically from
    the evidence gathered so far.

Works with any backend behind llms/llm_client.py (Ollama, vLLM, llama.cpp
server) since it only needs single-prompt JSON-mode generation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from nbx.llms import llm_client
from nbx.agents.agentic.tools import TOOLS, RunContext, call_tool

Emit = Callable[[dict], None]


class LLMUnavailable(Exception):
    pass


@dataclass
class AgentSpec:
    key: str  # "nba" | "nbo" | "copilot" -- also the UI lane
    name: str  # display name
    role: str  # system prompt
    goal: str
    tools: list[str]
    required_tools: list[str]
    final_schema: str  # shown to the model
    validate_final: Callable[[Any, RunContext], tuple[Any, str | None]]
    max_steps: int = 6


@dataclass
class AgentResult:
    final: Any | None
    evidence: list[dict] = field(default_factory=list)  # [{"tool","args","observation"}]
    steps: int = 0
    llm_calls: int = 0
    finalized_by: str = "llm"  # "llm" | "budget" | "fallback"


def _compact(obj: Any, limit: int = 900) -> str:
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def build_prompt(spec: AgentSpec, ctx: RunContext, history: list[dict], extra_context: str, feedback: str | None) -> str:
    tool_lines = "\n".join(f"- {n}{TOOLS[n].args}: {TOOLS[n].description}" for n in spec.tools)
    called = {h["tool"] for h in history if h.get("tool")}
    missing = [t for t in spec.required_tools if t not in called]
    parts = [
        f"CUSTOMER msisdn={ctx.msisdn}." if ctx.msisdn else "SCOPE: the whole customer portfolio.",
        f"GOAL: {spec.goal}",
        f"TOOLS (call exactly one per step):\n{tool_lines}",
        "REPLY with exactly one JSON object, either\n"
        '  {"thought": "<one short sentence>", "tool": "<tool name>", "args": {...}}\n'
        "or, once you have enough evidence,\n"
        f'  {{"thought": "<one short sentence>", "final": {spec.final_schema}}}',
    ]
    if missing:
        parts.append(f"You MUST call these before \"final\": {', '.join(missing)}.")
    if extra_context:
        parts.append(f"CONTEXT: {extra_context}")
    if feedback:
        parts.append(f"CRITIC FEEDBACK (fix this in your final answer): {feedback}")
    if history:
        lines = []
        for i, h in enumerate(history, 1):
            if h.get("tool"):
                lines.append(f"{i}. {h['tool']}({_compact(h.get('args') or {}, 200)}) -> {_compact(h['observation'])}")
            else:
                lines.append(f"{i}. {_compact(h['observation'], 400)}")
        parts.append("EVIDENCE SO FAR:\n" + "\n".join(lines))
    else:
        parts.append("EVIDENCE SO FAR: none yet.")
    parts.append("Next step JSON:")
    return "\n\n".join(parts)


FINAL_TOOL_NAMES = {"final", "finish", "final_answer", "answer", "done"}


def _coerce_step(parsed: dict, spec: AgentSpec, history: list[dict]) -> tuple[dict, str | None]:
    """Repairs the two near-misses small models make most, instead of burning a
    step on an "unknown tool" error:
      - {"tool": "final", "args": {...}}  -> treated as {"final": {...}};
      - several tools in one step ("get_risk_scores{..}, bandit_recommendation"
        or a list) -> the first named tool not yet called runs this step."""
    if "final" in parsed:
        return parsed, None
    tool = parsed.get("tool")
    if isinstance(tool, str) and tool.strip().lower() in FINAL_TOOL_NAMES:
        final = parsed.get("args") if isinstance(parsed.get("args"), dict) and parsed.get("args") else parsed.get("output")
        if isinstance(final, dict):
            return {**parsed, "final": final}, "Reply used tool \"final\"; read it as the final answer."
        return parsed, None
    if tool in spec.tools:
        return parsed, None
    text = " ".join(map(str, tool)) if isinstance(tool, list) else str(tool or "")
    named = sorted((text.find(t), t) for t in spec.tools if t in text)
    if not named:
        return parsed, None
    called = {h["tool"] for h in history if h.get("tool")}
    order = [t for _, t in named]
    pick = next((t for t in order if t not in called), order[0])
    args = parsed.get("args") if isinstance(parsed.get("args"), dict) and len(order) == 1 else {}
    note = (f"Reply named {', '.join(order)} in one step; running {pick} (one tool per step)."
            if len(order) > 1 else f"Reply tool '{text[:60]}' read as {pick}.")
    return {**parsed, "tool": pick, "args": args}, note


async def run_react(
    spec: AgentSpec,
    ctx: RunContext,
    emit: Emit,
    model: str,
    extra_context: str = "",
    feedback: str | None = None,
    prior: list[dict] | None = None,
    max_steps: int | None = None,
) -> AgentResult:
    history: list[dict] = list(prior or [])
    result = AgentResult(final=None, evidence=history)
    budget = max_steps or spec.max_steps
    seen: dict[str, dict] = {f"{h['tool']}:{_compact(h.get('args') or {})}": h["observation"] for h in history if h.get("tool")}
    repeats: dict[str, int] = {}

    for _ in range(budget):
        result.steps += 1
        prompt = build_prompt(spec, ctx, history, extra_context, feedback)
        t0 = time.time()
        res = await llm_client.generate(model=model, prompt=prompt, system=spec.role, force_json=True)
        result.llm_calls += 1
        if res.get("error"):
            raise LLMUnavailable(res["error"])
        parsed = llm_client.extract_json(res["content"])
        llm_ms = int((time.time() - t0) * 1000)

        if not isinstance(parsed, dict):
            obs = {"error": "Reply was not one JSON object. Use the REPLY format exactly."}
            history.append({"tool": None, "observation": obs})
            emit({"type": "policy", "agent": spec.key, "text": "Malformed reply rejected; asked the model to retry."})
            continue

        thought = str(parsed.get("thought") or "").strip()
        if thought:
            emit({"type": "thought", "agent": spec.key, "text": thought, "llm_ms": llm_ms})

        parsed, note = _coerce_step(parsed, spec, history)
        if note:
            emit({"type": "policy", "agent": spec.key, "text": note})

        if "final" in parsed:
            called = {h["tool"] for h in history if h.get("tool")}
            missing = [t for t in spec.required_tools if t not in called]
            if missing:
                obs = {"error": f"Not allowed to finalize yet. First call: {', '.join(missing)}."}
                history.append({"tool": None, "observation": obs})
                emit({"type": "policy", "agent": spec.key, "text": f"Finalize blocked: required evidence missing ({', '.join(missing)})."})
                continue
            final, err = spec.validate_final(parsed["final"], ctx)
            if err:
                history.append({"tool": None, "observation": {"error": f"Invalid final answer: {err}"}})
                emit({"type": "policy", "agent": spec.key, "text": f"Final answer rejected: {err}"})
                continue
            result.final = final
            return result

        tool = parsed.get("tool")
        args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
        if tool not in spec.tools:
            obs = {"error": f"Unknown tool '{tool}'. Valid tools: {', '.join(spec.tools)}."}
            history.append({"tool": None, "observation": obs})
            emit({"type": "policy", "agent": spec.key, "text": f"Rejected unknown tool '{tool}'."})
            continue

        key = f"{tool}:{_compact(args)}"
        if key in seen:
            # Small models loop on the same call; refuse it, say what to do next,
            # and stop the agent if it keeps repeating itself.
            repeats[key] = repeats.get(key, 0) + 1
            called = {h["tool"] for h in history if h.get("tool")}
            todo = [t for t in spec.required_tools if t not in called]
            nxt = f"call {todo[0]}" if todo else 'reply with "final"'
            if repeats[key] >= 2:
                emit({"type": "policy", "agent": spec.key, "text": f"Loop detected ({tool} repeated); finalizing from the evidence gathered."})
                result.finalized_by = "loop_guard"
                return result
            history.append({"tool": None, "observation": {"error": f"Duplicate call refused: you already have the {tool} result above. Next you must {nxt}."}})
            emit({"type": "policy", "agent": spec.key, "text": f"Duplicate {tool} call refused; told the agent to {nxt}."})
            continue
        emit({"type": "tool_call", "agent": spec.key, "tool": tool, "args": args})
        t1 = time.time()
        obs = call_tool(ctx, tool, args)
        seen[key] = obs
        emit({"type": "tool_result", "agent": spec.key, "tool": tool, "result": obs, "tool_ms": int((time.time() - t1) * 1000)})
        history.append({"tool": tool, "args": args, "observation": obs})

    result.finalized_by = "budget"
    emit({"type": "policy", "agent": spec.key, "text": f"Step budget ({budget}) reached; finalizing from the evidence gathered."})
    return result


def call_tool_traced(spec_key: str, ctx: RunContext, tool: str, args: dict, emit: Emit, history: list[dict]) -> dict:
    """Deterministic (fallback) tool call with the same trace events as the
    LLM path, so the UI renders both identically."""
    emit({"type": "tool_call", "agent": spec_key, "tool": tool, "args": args})
    t1 = time.time()
    obs = call_tool(ctx, tool, args)
    emit({"type": "tool_result", "agent": spec_key, "tool": tool, "result": obs, "tool_ms": int((time.time() - t1) * 1000)})
    history.append({"tool": tool, "args": args, "observation": obs})
    return obs


def evidence_of(history: list[dict], tool: str) -> dict | None:
    for h in reversed(history):
        if h.get("tool") == tool and "error" not in h["observation"]:
            return h["observation"]
    return None
