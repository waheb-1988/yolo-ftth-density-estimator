"""FSDR operations agents (audit L1): data, training, scoring and monitor.

Same pattern as the council and the recommender: one JSON reply per step (a tool call or a final answer),
a step cap, required evidence before finishing, repeated calls refused (nbx/agents/agentic/react.py).
Each agent has a rule-based fallback that calls the same tools, used when no LLM is reachable.

The LLM proposes; the code decides what is allowed:
  data      never "proceed" on blocked data; never generates over real (non-generated) data
  training  hyper-parameters only inside the allowed ranges; promotion only if the challenger's holdout IPS
            is >= the champion's (registry.can_promote) -- "promote" is downgraded to "keep" otherwise
  scoring   a stale or incompatible model is refused, whatever the agent says
  monitor   read-only: report and alerts
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from nbx.agents.agentic import council
from nbx.agents.agentic.react import AgentSpec, LLMUnavailable, call_tool_traced, run_react
from nbx.agents.agentic.tools import Tool, ToolError
from nbx.fsdr import bandit, campaign_log, config, registry
from nbx.fsdr.features import model_features

OPS_AGENTS = ["data", "training", "scoring", "monitor"]


@dataclass
class OpsContext:
    """Run-level context (no customer): build_prompt shows 'SCOPE: the whole customer portfolio'."""
    data_dir: Path
    run_date: date
    models_root: Path | None = None
    msisdn: None = None
    cache: dict = field(default_factory=dict)


def _context(data_dir: str | Path, models_root: Path | None) -> OpsContext:
    d = Path(data_dir)
    try:
        run_date = date.fromisoformat(d.name)
    except ValueError:
        m = d / config.MANIFEST_FILE
        run_date = date.fromisoformat(json.loads(m.read_text(encoding="utf-8"))["run_date"])
    return OpsContext(d, run_date, models_root)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def list_runs(ctx: OpsContext, args: dict) -> dict:
    root = ctx.data_dir.parent
    runs = []
    for p in sorted(root.glob("20??-??-??")):
        m = p / config.MANIFEST_FILE
        man = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
        runs.append({"run": p.name, "source": man.get("source", "unknown"), "files": sum(1 for _ in p.glob("*.json*"))})
    return {"runs": runs[-8:], "this_run": ctx.data_dir.name, "exists": ctx.data_dir.is_dir()}


def validate_run(ctx: OpsContext, args: dict) -> dict:
    from nbx.fsdr.validation import validate_folder

    if not ctx.data_dir.is_dir():
        return {"status": "missing", "message": f"no folder {ctx.data_dir.name}: generate or upload the data first"}
    rep = validate_folder(ctx.data_dir, ctx.run_date)
    ctx.cache["validation"] = rep.status
    issues = [{"level": i.level, "check": i.check, "table": i.table, "message": i.message[:160]} for i in rep.issues
              if i.level != "info"][:8]
    return {"status": rep.status, "counts": rep.counts(), "issues": issues}


def generate_data(ctx: OpsContext, args: dict) -> dict:
    from nbx.fsdr.synth import generator, scenarios

    m = ctx.data_dir / config.MANIFEST_FILE
    if ctx.data_dir.exists() and any(ctx.data_dir.glob("*.json*")) and not (
            m.exists() and json.loads(m.read_text(encoding="utf-8")).get("source") == "generator"):
        raise ToolError(f"{ctx.data_dir.name} holds real (non-generated) data: generation refused")
    n = int(args.get("n", config.GENERATOR_DEFAULT_N))
    if not 100 <= n <= config.GENERATOR_MAX_N:
        raise ToolError(f"n must be between 100 and {config.GENERATOR_MAX_N}")
    sc = scenarios.parse(args.get("scenarios") or "baseline")
    run = generator.generate(n, ctx.run_date, sc, config.GENERATOR_SEED, ctx.data_dir)
    return {"generated": ctx.data_dir.name, "n": n, "scenarios": sc, "tables": len(run.tables)}


def training_options(ctx: OpsContext, args: dict) -> dict:
    lo, hi = config.TARGETING_WINDOW
    log = campaign_log.read_log(campaign_log.root_for(ctx.data_dir), ctx.run_date + timedelta(days=lo), ctx.run_date + timedelta(days=hi))
    return {"modes": ["logged", "simulation"], "default_mode": config.TRAIN_MODE, "rewards": config.REWARD_DEFINITIONS,
            "default_reward": config.REWARD_DEFINITION, "passes_range": list(config.TRAIN_PASSES_RANGE),
            "epsilon_range": list(config.VW_EPSILON_RANGE), "default_epsilon": config.VW_EPSILON,
            "campaign_log_rows_last_week": int(len(log)),
            "note": "exact_offer / same_service rewards are meaningful once the campaign log holds last week's campaign"}


def champion_metrics(ctx: OpsContext, args: dict) -> dict:
    champ = registry.champion(ctx.models_root)
    if champ is None:
        return {"champion": None, "note": "no champion yet: the first model is promoted"}
    m = registry.models(ctx.models_root)["models"][champ]
    return {"champion": champ, **{k: m.get(k) for k in ("run_date", "holdout_ips_take_rate", "holdout_take_rate",
                                                        "holdout_taker_match", "train_mode", "reward_definition")}}


def train_challenger(ctx: OpsContext, args: dict) -> dict:
    from nbx.fsdr.pipelines.train import run_training

    mode = args.get("mode", config.TRAIN_MODE)
    reward = args.get("reward", config.REWARD_DEFINITION)
    passes = int(args.get("passes", config.TRAIN_PASSES))
    eps = float(args.get("epsilon", config.VW_EPSILON))
    if mode not in ("logged", "simulation") or reward not in config.REWARD_DEFINITIONS:
        raise ToolError(f"mode must be logged|simulation and reward one of {config.REWARD_DEFINITIONS}")
    if not config.TRAIN_PASSES_RANGE[0] <= passes <= config.TRAIN_PASSES_RANGE[1]:
        raise ToolError(f"passes must be in {list(config.TRAIN_PASSES_RANGE)}")
    if not config.VW_EPSILON_RANGE[0] <= eps <= config.VW_EPSILON_RANGE[1]:
        raise ToolError(f"epsilon must be in {list(config.VW_EPSILON_RANGE)}")
    rep = run_training(ctx.data_dir, mode=mode, reward_def=reward, passes=passes, epsilon=eps, promote="never",
                       root=ctx.models_root)
    ok, why = registry.can_promote(rep["version"], ctx.models_root)
    ctx.cache["challenger"] = rep["version"]
    ev = rep["evaluation"]
    ctx.cache["challenger_report"] = out = {"challenger": rep["version"], "mode": mode, "reward": reward, "passes": passes, "epsilon": eps,
            "label_source": rep["labels"]["label_source"], "labelled_rows": rep["labels"]["labelled_rows"],
            "holdout_ips_take_rate": ev.get("holdout_ips_take_rate"), "holdout_take_rate": ev.get("holdout_take_rate"),
            "holdout_taker_match": ev.get("holdout_taker_match"), "promotable": ok, "promotion_rule": why}
    return out


def model_status(ctx: OpsContext, args: dict) -> dict:
    champ = registry.champion(ctx.models_root)
    if champ is None:
        return {"champion": None, "usable": False, "reasons": ["no champion model: train first"]}
    meta = registry.meta(champ, ctx.models_root)
    reasons = []
    age = (ctx.run_date - date.fromisoformat(meta["run_date"])).days
    if age > config.MAX_MODEL_AGE_DAYS:
        reasons.append(f"model is {age} days old (max {config.MAX_MODEL_AGE_DAYS})")
    try:
        bandit.check_compatible(meta, model_features())
    except bandit.IncompatibleModel as e:
        reasons.append(str(e))
    return {"champion": champ, "trained_on": meta["run_date"], "age_days": age, "usable": not reasons, "reasons": reasons}


def score(ctx: OpsContext, args: dict) -> dict:
    from nbx.fsdr.pipelines.score import run_scoring

    status = model_status(ctx, {})
    if not status["usable"]:                                        # code-enforced: no override
        raise ToolError("scoring refused: " + "; ".join(status["reasons"]))
    policy = args.get("policy", config.SCORING_POLICY)
    if policy not in ("sample", "greedy"):
        raise ToolError("policy must be sample or greedy")
    rep = run_scoring(ctx.data_dir, policy=policy, root=ctx.models_root)
    ctx.cache["scored"] = True
    d = rep["drift"]
    ctx.cache["score_report"] = out = {"scored": True, "model_version": rep["model_version"], "policy": policy, "max_psi": d["max_psi"],
            "psi_alerts": d["psi_alerts"], "action_mix_shift_tv": d["action_mix_shift_tv"]}
    return out


def weekly_report_tool(ctx: OpsContext, args: dict) -> dict:
    from nbx.fsdr.pipelines.feedback import weekly_report

    week = date.fromisoformat(args["week"]) if args.get("week") else ctx.run_date - timedelta(days=7)
    rep = weekly_report(campaign_log.root_for(ctx.data_dir), week)
    ctx.cache["report"] = rep
    return rep


def drift_summary(ctx: OpsContext, args: dict) -> dict:
    p = ctx.data_dir / "outputs" / "nba_scoring_report.json"
    if not p.exists():
        return {"available": False, "note": "no scoring report for this run"}
    d = json.loads(p.read_text(encoding="utf-8"))["drift"]
    top = sorted(d["psi"].items(), key=lambda kv: -kv[1])[:5]
    return {"available": True, "max_psi": d["max_psi"], "psi_alert_threshold": config.PSI_ALERT, "top_psi": dict(top),
            "psi_alerts": d["psi_alerts"], "action_mix_shift_tv": d["action_mix_shift_tv"]}


OPS_TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("list_runs", "Run folders of the data root and their source (generator / real).", "{}", list_runs),
    Tool("validate_run", "Validation of this run: status ok / warnings / blocked and the main issues.", "{}", validate_run),
    Tool("generate_data", "Dummy data for this run date (refused over real data).", '{"n"?: int, "scenarios"?: str}', generate_data),
    Tool("training_options", "Allowed training modes, rewards and hyper-parameter ranges; campaign-log rows.", "{}", training_options),
    Tool("champion_metrics", "The champion model and its holdout metrics.", "{}", champion_metrics),
    Tool("train_challenger", "Train a challenger (not promoted) and get its holdout metrics and promotability.",
         '{"mode"?: str, "reward"?: str, "passes"?: int, "epsilon"?: float}', train_challenger),
    Tool("model_status", "Champion model age and compatibility with this run's features.", "{}", model_status),
    Tool("score", "Score the test sample with the champion (refused if the model is stale or incompatible).",
         '{"policy"?: "sample"|"greedy"}', score),
    Tool("weekly_report", "Take rate and revenue per arm (cg / free / cstr / llm), uplift vs control, llm agreement.",
         '{"week"?: "YYYY-MM-DD"}', weekly_report_tool),
    Tool("drift_summary", "Feature PSI and action-mix shift of this run's scoring vs the training data.", "{}", drift_summary),
]}


# ---------------------------------------------------------------------------
# Final-answer validators (structure) and code guards (what is allowed)
# ---------------------------------------------------------------------------
def _v_data(final, ctx):
    if not isinstance(final, dict) or final.get("decision") not in ("proceed", "stop"):
        return None, 'final must be {"decision": "proceed"|"stop", "reason": str}'
    return {"decision": final["decision"], "reason": str(final.get("reason") or "").strip()}, None


def _v_training(final, ctx):
    if not isinstance(final, dict) or final.get("decision") not in ("promote", "keep", "ask_human"):
        return None, 'final.decision must be "promote", "keep" or "ask_human"'
    if not str(final.get("justification") or "").strip():
        return None, "a written justification is required"
    if "challenger" not in ctx.cache:
        return None, "train a challenger first (train_challenger)"
    return {"decision": final["decision"], "justification": str(final["justification"]).strip()}, None


def _v_scoring(final, ctx):
    if not isinstance(final, dict) or final.get("decision") not in ("scored", "refused"):
        return None, 'final.decision must be "scored" or "refused"'
    return {"decision": final["decision"], "reason": str(final.get("reason") or "").strip(),
            "drift_alerts": [str(a) for a in final.get("drift_alerts") or []]}, None


def _v_monitor(final, ctx):
    if not isinstance(final, dict) or not str(final.get("summary") or "").strip():
        return None, 'final must be {"summary": str, "alerts": [str]}'
    return {"summary": str(final["summary"]).strip(), "alerts": [str(a) for a in final.get("alerts") or []]}, None


SPECS = {
    "data": AgentSpec("fsdr_data", "Data agent",
                      f"You are the {config.OPERATOR_NAME} data steward. Make sure this week's run folder exists and passes the "
                      "data contract before any training or scoring. Reply with JSON only.",
                      "Check this run's data and decide whether the weekly cycle can proceed.",
                      ["list_runs", "validate_run", "generate_data"], ["validate_run"],
                      '{"decision": "proceed"|"stop", "reason": str}', _v_data, max_steps=4),
    "training": AgentSpec("fsdr_training", "Training agent",
                          f"You are the {config.OPERATOR_NAME} model owner. Choose the training mode, reward definition and "
                          "hyper-parameters within the allowed ranges, train a challenger, compare it with the champion and "
                          "decide: promote, keep the champion, or ask a human. Reply with JSON only.",
                          "Train this week's challenger and decide on promotion, with a written justification.",
                          ["training_options", "champion_metrics", "train_challenger"], ["training_options", "train_challenger"],
                          '{"decision": "promote"|"keep"|"ask_human", "justification": str}', _v_training, max_steps=5),
    "scoring": AgentSpec("fsdr_scoring", "Scoring agent",
                         f"You are the {config.OPERATOR_NAME} scoring operator. Check the champion model's age and compatibility, "
                         "score the test sample if it is safe, and flag feature drift. Reply with JSON only.",
                         "Score this week's test sample with a safe model and report drift.",
                         ["model_status", "score", "drift_summary"], ["model_status"],
                         '{"decision": "scored"|"refused", "reason": str, "drift_alerts": [str]}', _v_scoring, max_steps=4),
    "monitor": AgentSpec("fsdr_monitor", "Monitor agent",
                         f"You are the {config.OPERATOR_NAME} campaign analyst. Summarise last week's results per arm "
                         "(cg / free / cstr / llm): take rate, revenue, uplift vs control, bandit vs LLM agreement, drift. "
                         "Reply with JSON only.",
                         "Write the weekly report and list the alerts.",
                         ["weekly_report", "drift_summary"], ["weekly_report", "drift_summary"],
                         '{"summary": str, "alerts": [str]}', _v_monitor, max_steps=4),
}


def _fallback(kind: str, ctx: OpsContext, emit, hist: list[dict]) -> dict:
    """Rule-based path: the same tools, default choices."""
    def call(tool, args=None):
        return call_tool_traced(SPECS[kind].key, ctx, tool, args or {}, emit, hist, OPS_TOOLS)

    if kind == "data":
        v = call("validate_run")
        ok = v.get("status") in ("ok", "warnings")
        return {"decision": "proceed" if ok else "stop", "reason": f"validation {v.get('status')}"}
    if kind == "training":
        call("training_options")
        call("champion_metrics")
        t = ctx.cache.get("challenger_report") or call("train_challenger")     # reuse a challenger the LLM trained
        if "error" in t:
            return {"decision": "keep", "justification": t["error"]}
        return {"decision": "promote" if t["promotable"] else "keep", "justification": t["promotion_rule"]}
    if kind == "scoring":
        s = call("model_status")
        if not s.get("usable"):
            return {"decision": "refused", "reason": "; ".join(s.get("reasons", [])), "drift_alerts": []}
        r = ctx.cache.get("score_report") or call("score")
        if "error" in r:
            return {"decision": "refused", "reason": r["error"], "drift_alerts": []}
        return {"decision": "scored", "reason": f"model {r['model_version']}",
                "drift_alerts": [f"{f}: PSI {p}" for f, p in r["psi_alerts"].items()]}
    rep = call("weekly_report")
    drift = call("drift_summary")
    return _monitor_text(rep, drift)


def _monitor_text(rep: dict, drift: dict) -> dict:
    arms = rep.get("arms") or {}
    if not arms:
        return {"summary": f"No rewards yet for the week of {rep.get('week')}: run the feedback of the following week.",
                "alerts": []}
    parts, alerts = [], []
    for arm, m in arms.items():
        up = m.get("uplift_vs_control")
        parts.append(f"{arm}: take rate {m['take_rate_any_btl']:.1%}, revenue/line {m['revenue_per_line']:.2f}"
                     + (f", uplift {up:+.1%}" if up is not None else ""))
        if up is not None and up < 0:
            alerts.append(f"{arm} below the control by {abs(up):.1%}")
    if "llm" in rep:
        parts.append(f"llm agrees with the bandit on {rep['llm']['bandit_agreement']:.0%} of lines")
    if drift.get("available") and drift.get("psi_alerts"):
        alerts.append("feature drift: " + ", ".join(f"{f} PSI {p}" for f, p in list(drift["psi_alerts"].items())[:3]))
    return {"summary": f"Week {rep['week']}: " + "; ".join(parts) + ".", "alerts": alerts}


def _guard(kind: str, final: dict, ctx: OpsContext, findings: list[str]) -> dict:
    """Code-enforced rules on the agent's final answer."""
    if kind == "data" and final["decision"] == "proceed" and ctx.cache.get("validation") == "blocked":
        findings.append("proceed -> stop: the validation is blocked")
        final = {**final, "decision": "stop"}
    if kind == "training":
        version = ctx.cache.get("challenger")
        ok, why = registry.can_promote(version, ctx.models_root) if version else (False, "no challenger")
        if final["decision"] == "promote" and not ok:
            findings.append(f"promote -> keep: {why}")
            final = {**final, "decision": "keep"}
        if final["decision"] == "promote":
            registry.promote(version, ctx.models_root, reason=f"training agent: {final['justification']}")
        final = {**final, "challenger": version, "promotion_rule": why}
    if kind == "scoring" and final["decision"] == "scored" and not ctx.cache.get("scored"):
        findings.append("scored -> refused: the score tool did not run successfully")
        final = {**final, "decision": "refused"}
    return final


async def run_ops_agent(kind: str, data_dir: str | Path, llm: str | None = None, models_root: Path | None = None,
                        emit=None) -> dict:
    if kind not in SPECS:
        raise ValueError(f"agent must be one of {OPS_AGENTS}")
    spec, ctx = SPECS[kind], _context(data_dir, models_root)
    events: list[dict] = []
    t0 = time.time()

    def _emit(ev):
        ev = {"t_ms": int((time.time() - t0) * 1000), **ev}
        events.append(ev)
        if emit:
            emit(ev)

    model = await council._resolve_model(llm)
    mode = "agentic" if model else "fallback"
    _emit({"type": "run_start", "agent": spec.key, "mode": mode, "model": model})
    final, finalized_by, hist = None, "policy", []
    if model:
        try:
            res = await run_react(spec, ctx, _emit, model, registry=OPS_TOOLS)
            final, finalized_by, hist = res.final, ("llm" if res.final else res.finalized_by), res.evidence
        except LLMUnavailable as e:
            _emit({"type": "policy", "agent": spec.key, "text": f"LLM unavailable ({e}); rule-based path."})
            mode = "partial_fallback"
    if final is None:                                   # no LLM, or it ran out of steps: rule-based, same tools
        final = _fallback(kind, ctx, _emit, list(hist))
        finalized_by = "policy"
    findings: list[str] = []
    final = _guard(kind, final, ctx, findings)
    _emit({"type": "agent_final", "agent": spec.key, "output": final, "finalized_by": finalized_by})
    return {"agent": kind, "run": ctx.data_dir.name, "mode": mode, "model": model, "final": final,
            "findings": findings, "finalized_by": finalized_by, "trace": events,
            "seconds": round(time.time() - t0, 2)}
