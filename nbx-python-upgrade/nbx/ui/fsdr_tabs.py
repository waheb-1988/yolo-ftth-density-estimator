"""FSDR tabs of the Gradio app, added after the 12 NBX tabs.

Same theme, CSS classes and renderers as the other tabs (nbx/ui/theme.py);
every button calls the FastAPI /fsdr endpoints -- no business logic here.
"""
from __future__ import annotations

import csv
import json
import tempfile
from datetime import date

import gradio as gr
import httpx

from nbx.core import config as nbx_config
from nbx.fsdr import config
from nbx.ui import theme as ui

API_BASE = f"http://localhost:{nbx_config.FASTAPI_PORT}"
TIMEOUT = httpx.Timeout(900.0, connect=10.0)
esc = ui.esc


async def _get(path: str, **params):
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.get(f"{API_BASE}{path}", params=params)
        if r.status_code >= 400:
            raise gr.Error(f"{path}: {r.json().get('detail', r.text)}")
        return r.json()


async def _post(path: str, body: dict | None = None, **params):
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(f"{API_BASE}{path}", json=body or {}, params=params)
        if r.status_code >= 400:
            raise gr.Error(f"{path}: {r.json().get('detail', r.text)}")
        return r.json()


async def _post_files(path: str, paths: list[str], **params):
    from pathlib import Path

    files = [("files", (Path(p).name, Path(p).read_bytes(), "application/json")) for p in paths]
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(f"{API_BASE}{path}", files=files, params=params)
        if r.status_code >= 400:
            raise gr.Error(f"{path}: {r.json().get('detail', r.text)}")
        return r.json()


def _section(title: str, body: str) -> str:
    return f'<div class="nbx-section"><h3>{esc(title)}</h3>{body}</div>'


def _table(rows: list[dict], cols: list[str], labels: dict | None = None) -> str:
    if not rows:
        return '<div class="nbx-muted">Nothing to show.</div>'
    head = "".join(f"<th>{esc((labels or {}).get(c, c))}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div style="overflow-x:auto"><table class="nbx-table"><tr>{head}</tr>{body}</table></div>'


def _kv(d: dict) -> str:
    return _table([{"k": k, "v": v if not isinstance(v, (dict, list)) else json.dumps(v)} for k, v in d.items()], ["k", "v"],
                  {"k": "Metric", "v": "Value"})


def _badges(items: list[str], cls: str = "") -> str:
    return "".join(f'<span class="nbx-badge {cls}">{esc(i)}</span>' for i in items)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
async def load_runs():
    try:
        runs = await _get("/fsdr/runs")
    except Exception:  # noqa: BLE001  (API not up yet: leave the dropdowns empty)
        runs = []
    names = [r["run"] for r in runs]
    value = names[0] if names else None
    rows = [{**r, "scenarios": ",".join(r.get("scenarios") or []), "arms_recommended": ",".join(r["arms_recommended"]) or "-"} for r in runs]
    html = _section("FSDR run folders (data/fsdr/<date>)", _table(rows, ["run", "anchor_date", "source", "n", "scenarios", "validation",
                                                                          "trained", "scored", "arms_recommended"]))
    upd = gr.update(choices=names, value=value)
    return upd, upd, upd, upd, upd, html


async def do_generate(run_date: str, n: float, scenarios: list[str], seed: float):
    res = await _post("/fsdr/generate", {"date": run_date or None, "n": int(n), "scenarios": scenarios or ["baseline"], "seed": int(seed)})
    counts = res["row_counts"]
    return (f"Generated **{len(counts)} tables** for D={res['run']} (T={res['anchor_date']}), scenarios "
            f"{', '.join(res['scenarios'])}, in {res['seconds']} s. Click **Refresh runs**, then **Validate**.")


async def do_validate(run: str):
    if not run:
        raise gr.Error("Pick a run first.")
    return (await _post(f"/fsdr/validate/{run}"))["markdown"]


async def do_preview(run: str, table: str, limit: float):
    if not run or not table:
        raise gr.Error("Pick a run and a table.")
    res = await _get(f"/fsdr/tables/{run}/{table}", limit=int(limit))
    return gr.update(value={"headers": res["columns"], "data": [[r.get(c) for c in res["columns"]] for r in res["records"]]}), \
        f"`{table}`: {res['rows']:,} rows (showing {len(res['records'])})"


# ---------------------------------------------------------------------------
# Training, scoring, registry
# ---------------------------------------------------------------------------
async def do_train(run: str, mode: str, reward: str, passes: float, epsilon: float, promote: str):
    if not run:
        raise gr.Error("Pick a run first.")
    r = await _post(f"/fsdr/train/{run}", {"mode": mode, "reward": reward, "passes": int(passes), "epsilon": float(epsilon),
                                           "promote": promote})
    ev, lab = r["evaluation"], r["labels"]
    head = (f'<div class="nbx-action"><div class="label">Model</div><div class="name">{esc(r["version"])}</div>'
            f'{_badges(["promoted" if r["promoted"] else "challenger only", "champion: " + str(r["champion"])], "red" if r["promoted"] else "")}'
            f'<div class="nbx-muted" style="margin-top:8px">{esc(r["promotion_rule"])} · {r["seconds"]} s</div></div>')
    by_action = [{"action": a, **v} for a, v in lab["by_action"].items()]
    metrics = {k: ev.get(k) for k in ("n_holdout", "holdout_take_rate", "holdout_ips_take_rate", "holdout_snips_take_rate",
                                      "holdout_taker_match")}
    dist = [{"action": a, "share": f"{p:.1%}"} for a, p in (r.get("scored_base_action_distribution") or {}).items()]
    return (head + _section("Labels (targeting D-8..D-2, reward D-7..D-1)",
                            _kv({k: lab[k] for k in ("labelled_rows", "nursering_excluded", "no_segment_dropped", "reward_definition",
                                                     "take_rate")}) + _table(by_action, ["action", "rows", "take_rate"]))
            + _section("Holdout evaluation (model policy vs logged policy)", _kv(metrics))
            + _section("Action mix on the eligible base (greedy)", _table(dist, ["action", "share"])))


async def do_score(run: str, population: str, policy: str, allow_stale: bool):
    if not run:
        raise gr.Error("Pick a run first.")
    r = await _post(f"/fsdr/score/{run}", {"population": population, "policy": policy, "allow_stale": allow_stale})
    per_arm = r["action_distribution_per_arm"]
    arms = list(per_arm)
    rows = [{"action": a, **{arm: f"{per_arm[arm].get(a, 0):.1%}" for arm in arms}} for a in config.ACTIONS]
    drift = r["drift"]
    return (_section(f"Action distribution per arm · model {r['model_version']} ({r['model_age_days']} days old), policy {r['policy']}",
                     _table(rows, ["action"] + arms))
            + _section("Drift vs training", _kv({"max PSI": drift["max_psi"], "PSI alerts": drift["psi_alerts"] or "none",
                                                  "action-mix shift (TV distance)": drift["action_mix_shift_tv"]})))


async def do_models():
    m = await _get("/fsdr/models")
    rows = [{"version": v, "champion": "★" if v == m["champion"] else "", **s} for v, s in m["models"].items()]
    return _section("Model registry", _table(rows, ["version", "champion", "run_date", "train_mode", "reward_definition", "n_train",
                                                    "holdout_take_rate", "holdout_ips_take_rate", "trained_at"]))


async def do_promote(version: str):
    if not version:
        raise gr.Error("Type a version, e.g. v0002")
    await _post(f"/fsdr/models/{version.strip()}/promote")
    return await do_models()


async def do_rollback():
    await _post("/fsdr/models/rollback")
    return await do_models()


# ---------------------------------------------------------------------------
# Offers per arm
# ---------------------------------------------------------------------------
async def do_recommend(run: str, arm: str, mapping: str):
    if not run:
        raise gr.Error("Pick a run first.")
    r = await _post(f"/fsdr/recommend/{run}", {"arm": arm, "mapping": mapping})
    recos = await _get(f"/fsdr/recos/{run}/{arm}", limit=5000)
    rows = recos["records"]
    if arm == "llm":
        summary = _kv({k: r[k] for k in ("customers", "decided_now", "reused_from_checkpoint", "modes", "policy_source",
                                          "override_rate", "competitor_verdicts", "offers", "seconds")})
    else:
        summary = _kv({k: r[k] for k in ("customers", "with_offers", "without_offer", "offers", "offers_per_pass",
                                          "offers_per_validity", "mean_price", "mapping", "ranker", "seconds")})
    cols = ["ACCESS_METHOD_VAL", "ACTION", "BTL_NAME", "SERVICE", "GENEROSITY", "VALIDITY", "PRICE", "ANCHOR", "PASS_NO"]
    if arm == "llm":
        cols += ["POLICY_SOURCE", "PROPENSITY"]
    path = tempfile.NamedTemporaryFile(prefix=f"fsdr_recos_{arm}_", suffix=".csv", delete=False).name
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    shown = [{**x, "ANCHOR": round(x["ANCHOR"], 3) if isinstance(x.get("ANCHOR"), float) else x.get("ANCHOR")} for x in rows[:200]]
    return (_section(f"Arm {arm} · summary", summary)
            + _section(f"Offers ({len(rows)} rows, first 200 shown) · anchor = ARPU anchor, pass = price-band pass (BR-78..80)",
                       _table(shown, cols))), path


# ---------------------------------------------------------------------------
# Agentic recommender (action -> offer -> competitor)
# ---------------------------------------------------------------------------
async def load_customers(run: str):
    if not run:
        return gr.update(choices=[], value=None)
    try:
        rows = await _get(f"/fsdr/customers/{run}", arm="llm", limit=300)
    except gr.Error:
        return gr.update(choices=[], value=None)
    rows.sort(key=lambda r: -r["churn"])
    return gr.update(choices=[(f"{r['msisdn']} · {r['action']} · churn {r['churn']:.2f}", r["msisdn"]) for r in rows], value=None)


def render_decision(d: dict) -> str:
    badges = [f"Mode: {d['mode']}", f"Policy source: {d['policy_source']}", f"Propensity: {d['propensity']:.3f}",
              f"Bandit top: {d['bandit_top_action']}", f"Overrides: {d['override_count']}", f"Critic: {d['critic']['verdict']}"]
    reason = d["justification"].get("override_reason") or "; ".join(d["justification"].get("signals_used") or [])
    head = (f'<div class="nbx-action"><div class="label">Recommended action</div><div class="name">{esc(d["action"])}</div>'
            f'<div class="nbx-muted" style="font-size:14px;color:#d4d4d8">{esc(reason)}</div><div style="margin-top:10px">{_badges(badges)}</div></div>')
    pmf = sorted(d["bandit_pmf"].items(), key=lambda kv: -kv[1])
    pmf_html = _table([{"action": a, "p": f"{p:.1%}"} for a, p in pmf], ["action", "p"], {"p": "Bandit probability"})
    offers = _table([{**o, "anchor": round(o["anchor"], 3) if o.get("anchor") is not None else "-", "pass": o.get("pass") or "-"}
                     for o in d["offers"]], ["offer", "service", "generosity", "validity", "price", "anchor", "pass"])
    findings = "".join(f"<li>{esc(f)}</li>" for f in d["critic"]["findings"]) or "<li>All hard rules passed.</li>"
    overrides = _table([{**o, "model": json.dumps(o["model"]) if isinstance(o["model"], list) else o["model"],
                         "agent": json.dumps(o["agent"]) if isinstance(o["agent"], list) else o["agent"]} for o in d["overrides"]],
                       ["type", "model", "agent", "reason"], {"model": "Model / rules", "agent": "Agent chose"}) if d["overrides"] else \
        '<div class="nbx-muted">The agents followed the models.</div>'
    return (head + _section("Offers", offers) + _section("Bandit evidence", pmf_html)
            + _section("Overrides (logged for learning)", overrides) + _section("Critic (hard rules)", f"<ul>{findings}</ul>"))


def render_competitor(d: dict) -> str:
    c = d["competitor"]
    if not c.get("verdict"):
        return _section("Competitor agent", '<div class="nbx-muted">No offer to compare.</div>')
    rows = []
    for r in c["comparisons"]:
        for x in r.get("closest") or [{}]:
            rows.append({"offer": r["offer"], "our_monthly": r.get("our_monthly_equiv"), "operator": x.get("operator", "-"),
                         "plan": x.get("plan", "-"), "their_monthly": x.get("monthly_equiv", "-"), "delta": x.get("monthly_delta", "-"),
                         "verdict": x.get("verdict", r.get("position", "-")), "at_risk": "yes" if r.get("at_risk") else ""})
    swaps = "".join(f"<li>{esc(s['out'])} → {esc(s['in'])}: {esc(s['reason'])}</li>" for s in c["swaps_applied"])
    return _section(f"Competitor agent · verdict: {c['verdict']}",
                    f'<div class="nbx-muted" style="margin-bottom:8px">{esc(c["summary"])}</div>'
                    + _table(rows, ["offer", "our_monthly", "operator", "plan", "their_monthly", "delta", "verdict", "at_risk"],
                             {"our_monthly": "Ours / month", "their_monthly": "Theirs / month", "delta": "Delta"})
                    + (f"<b>Swaps applied</b><ul>{swaps}</ul>" if swaps else ""))


def render_message(d: dict) -> str:
    m = d["message"]
    if not m["sms_en"]:
        return ""
    en = "".join(f"<p>{esc(s)}</p>" for s in m["sms_en"])
    ar = "".join(f'<p dir="rtl">{esc(s)}</p>' for s in m["sms_ar"])
    return _section("Customer message (BTL scripts)", f"<b>English</b>{en}<b>Arabic</b>{ar}")


async def run_decision(run: str, msisdn: str):
    msisdn = (msisdn or "").strip()
    if not run or not msisdn:
        raise gr.Error("Pick a run and a customer.")
    events: list[dict] = []
    empty = ""
    yield ui.render_pipeline([], True), '<div class="nbx-muted">Starting the agents…</div>', empty, empty, empty
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("GET", f"{API_BASE}/fsdr/stream/{run}", params={"msisdn": msisdn}) as resp:
            if resp.status_code >= 400:
                raise gr.Error(f"Backend refused the run: {(await resp.aread()).decode()[:300]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev["type"] == "error":
                    raise gr.Error(ev.get("detail", "Agent run failed"))
                if ev["type"] == "decision":
                    d = ev["decision"]
                    yield (ui.render_pipeline(events, False), ui.render_trace(events), render_decision(d), render_competitor(d),
                           render_message(d))
                    return
                events.append(ev)
                yield ui.render_pipeline(events, True), ui.render_trace(events), empty, empty, empty


# ---------------------------------------------------------------------------
# Batch: action -> top 4 offers -> top 2 roaming -> international
# ---------------------------------------------------------------------------
BATCH_COLS = ["msisdn", "action", "offer_1", "offer_2", "offer_3", "offer_4", "roaming_1", "roaming_2", "international",
              "competitor", "critic", "mode", "error"]


def _parse_list(text: str) -> list[str]:
    import re

    out = []
    for tok in re.split(r"[\s,;]+", str(text or "").strip()):
        tok = tok.lstrip("+")
        if tok.isdigit() and tok not in out:
            out.append(tok)
    return out


def _batch_html(rows: list[dict]) -> str:
    def cell(r, c):
        v = r.get(c) or ""
        if c == "action":
            return f"<span class='nbx-badge red'>{esc(v)}</span>"
        if c in ("roaming_1", "roaming_2") and v:
            return f"<span class='nbx-badge'>{esc(v)}</span>"
        if c == "error" and v:
            return f"<span style='color:#ff5470'>{esc(v)}</span>"
        return esc(v)

    head = "".join(f"<th>{esc(c.replace('_', ' '))}</th>" for c in BATCH_COLS)
    body = "".join("<tr>" + "".join(f"<td>{cell(r, c)}</td>" for c in BATCH_COLS) + "</tr>" for r in rows)
    return f'<div class="nbx-section" style="overflow-x:auto"><table class="nbx-table"><tr>{head}</tr>{body}</table></div>'


async def run_batch_flow(run: str, text: str, arm: str, batch_size: float, limit: float):
    """Splits the list into batches and calls POST /fsdr/batch once per batch; the table grows as batches finish."""
    if not run:
        raise gr.Error("Pick a run first.")
    msisdns = _parse_list(text)
    if not msisdns:
        msisdns = [r["msisdn"] for r in await _get(f"/fsdr/customers/{run}", arm=arm, limit=int(limit))]
    msisdns = msisdns[: int(limit)]
    size = max(1, int(batch_size))
    batches = [msisdns[i:i + size] for i in range(0, len(msisdns), size)]
    rows: list[dict] = []
    for i, b in enumerate(batches, 1):
        res = await _post(f"/fsdr/batch/{run}", {"msisdns": b, "arm": arm})
        rows += res["table"]
        yield f"Batch **{i} / {len(batches)}** done · {len(rows)} customers", _batch_html(rows), None
    path = tempfile.NamedTemporaryFile(prefix=f"fsdr_batch_{arm}_", suffix=".csv", delete=False).name
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=BATCH_COLS + ["propensity"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with_roam = sum(1 for r in rows if r.get("roaming_1"))
    with_intl = sum(1 for r in rows if r.get("international") and "BTL_" in r["international"])
    yield (f"**{len(rows)} customers** in {len(batches)} batch(es) · with main offers: "
           f"{sum(1 for r in rows if r.get('offer_1'))} · roaming: {with_roam} · international: {with_intl}"), _batch_html(rows), path


# ---------------------------------------------------------------------------
# Layout (called inside the app's gr.Tabs, after the existing tabs)
# ---------------------------------------------------------------------------
async def do_upload(run_date: str, files, force: bool):
    paths = [f if isinstance(f, str) else f.name for f in (files or [])]
    if not paths:
        raise gr.Error("Choose the JSON files to upload (one <table>.json or .jsonl per table).")
    r = await _post_files(f"/fsdr/upload/{run_date.strip()}", paths, force=str(bool(force)).lower())
    v = r["validation"]
    extra = f"\n\nNot a contract table (kept, ignored): {', '.join(r['unknown_files'])}" if r["unknown_files"] else ""
    return f"**{len(r['saved'])} table file(s) saved in `data/fsdr/{r['run']}/`** — validation: **{v['status']}**{extra}\n\n{v['markdown']}"


def _arms_html(arms: dict, title: str) -> str:
    rows = [{"arm": a, "lines": m["lines"], "take": f"{m['take_rate_any_btl']:.2%}", "same": f"{m['take_rate_same_service']:.2%}",
             "exact": f"{m['take_rate_exact_offer']:.2%}", "rev": f"{m['revenue_per_line']:.3f}",
             "uplift": f"{m['uplift_vs_control']:+.2%}" if "uplift_vs_control" in m else "—"} for a, m in arms.items()]
    return _section(title, _table(rows, ["arm", "lines", "take", "same", "exact", "rev", "uplift"],
                                  {"arm": "Arm", "lines": "Lines", "take": "Take rate (any BTL)", "same": "Same service",
                                   "exact": "Exact offer", "rev": f"Revenue / line ({config.CURRENCY})", "uplift": "Uplift vs control"}))


async def do_simulate(start: str, weeks: float, n: float, scenarios: list[str], arms: list[str], seed: float):
    rep = await _post("/fsdr/simulate", {"start": start.strip() or None, "weeks": int(weeks), "n": int(n),
                                         "scenarios": ",".join(scenarios or ["baseline"]), "arms": arms or ["cstr"], "seed": int(seed)})
    parts = [_section("Simulation", _kv({"weeks": rep["weeks"], "lines": rep["n"], "arms": ", ".join(rep["arms"]),
                                         "scenarios": ", ".join(rep["scenarios"]), "seconds": rep["seconds"]}))]
    rows = [{"arm": a, "take": f"{m['take_rate']:.2%}", "rev": f"{m['revenue_per_line']:.3f}",
             "uplift": f"{m['uplift_vs_control']:+.2%}" if "uplift_vs_control" in m else "—"} for a, m in rep["summary"].items()]
    parts.append(_section("Mean over the weeks", _table(rows, ["arm", "take", "rev", "uplift"],
                                                        {"arm": "Arm", "take": "Take rate", "rev": "Revenue / line", "uplift": "Uplift vs control"})))
    for w in rep["weekly"]:
        tr = w.get("training", {})
        title = f"Week {w['week']} — model {tr.get('version')} ({tr.get('label_source')} labels)"
        if w.get("results"):
            parts.append(_arms_html(w["results"]["arms"], title))
    return "".join(parts)


async def do_ops_agent(kind: str, run: str):
    if not run:
        raise gr.Error("Pick a run.")
    out = await _post(f"/fsdr/agents/{kind}/{run}")
    trace = _table([{"t": e["t_ms"], "type": e["type"], "what": e.get("tool") or e.get("text") or ""} for e in out["trace"]],
                   ["t", "type", "what"], {"t": "ms", "type": "Event", "what": "Tool / note"})
    head = _kv({"agent": out["agent"], "run": out["run"], "mode": out["mode"], "finalized by": out["finalized_by"],
                **{k: v for k, v in out["final"].items()}, **({"guards": out["findings"]} if out["findings"] else {})})
    return _section(f"{kind.title()} agent", head) + _section("Trace", trace)


async def do_feedback(run: str):
    if not run:
        raise gr.Error("Pick the run folder of the week AFTER the campaign (its fulfilments hold the responses).")
    rep = await _post(f"/fsdr/feedback/{run}")
    head = _kv({"campaign": " → ".join(rep["campaign_window"]), "fulfilments": " → ".join(rep["reward_window"]),
                "rows": rep["rows"], "new reward rows": rep["new_reward_rows"]})
    return _section("Feedback", head) + _arms_html(rep["arms"], "Rewards per arm")


async def do_report(week: str):
    rep = await _get(f"/fsdr/report/{week.strip()}")
    head = _kv({"week": rep["week"], "logged lines": rep["logged_lines"], "rewarded lines": rep["rewarded_lines"],
                "logged per arm": rep["logged_per_arm"], **({"llm override rate": rep["llm"]["override_rate"],
                                                           "llm / bandit agreement": rep["llm"]["bandit_agreement"]} if "llm" in rep else {})})
    body = _arms_html(rep["arms"], "Per arm") if rep.get("arms") else '<div class="nbx-muted">No rewards yet: run the feedback of the next week.</div>'
    return _section("Weekly report", head) + body


async def do_campaign_log(arm: str, limit: float):
    r = await _get("/fsdr/campaign-log", limit=int(limit), **({"arm": arm} if arm and arm != "all" else {}))
    rows = [{**x, "offers": ", ".join(o["code"] for o in (x.get("offers") or [])), "propensity": round(float(x.get("propensity") or 0), 4)}
            for x in r["rows"]]
    cols = ["run_date", "arm", "msisdn", "action", "offers", "policy_source", "propensity", "model_version", "llm_model"]
    return _section(f"Campaign log — {r['total']} rows {r['per_arm']}", _table(rows, cols))


def build() -> dict:
    run_kw = {"label": "FSDR run (data/fsdr/<date>)", "choices": [], "allow_custom_value": False, "scale": 3}

    with gr.Tab("FSDR Data", id="fsdr-data"):
        gr.HTML('<div class="nbx-muted">FSDR data contract: one JSON file per table in <code>data/fsdr/&lt;date&gt;/</code>. '
                "Generate dummy data with the same structure as the real exports, validate it, and preview any table.</div>")
        with gr.Row(equal_height=True):
            data_run = gr.Dropdown(**run_kw)
            refresh_btn = gr.Button("Refresh runs", scale=1)
            validate_btn = gr.Button("Validate", variant="primary", scale=1)
        runs_html = gr.HTML()
        with gr.Accordion("Generate dummy data", open=False):
            with gr.Row():
                gen_date = gr.Textbox(label="Run date D (YYYY-MM-DD)", value=date.today().isoformat())
                gen_n = gr.Number(label="Lines", value=2000, precision=0)
                gen_seed = gr.Number(label="Seed", value=config.GENERATOR_SEED, precision=0)
            gen_scen = gr.CheckboxGroup(config.SCENARIOS, value=["baseline"], label="Scenarios (combinable)")
            gen_btn = gr.Button("Generate", variant="primary")
            gen_md = gr.Markdown()
        with gr.Accordion("Upload real JSON exports", open=False):
            gr.HTML('<div class="nbx-muted">One file per table, named exactly like the table (<code>base_prepaid.json</code>, '
                    "<code>activities_w1_score.jsonl</code>, …). They are saved in <code>data/fsdr/&lt;date&gt;/</code> and validated "
                    "against the contract. Real data never leaves this machine.</div>")
            with gr.Row(equal_height=True):
                up_date = gr.Textbox(label="Run date D (YYYY-MM-DD)", value=date.today().isoformat(), scale=2)
                up_force = gr.Checkbox(label="Replace generated data", value=False, scale=1)
            up_files = gr.File(label="JSON / JSONL files", file_count="multiple", file_types=[".json", ".jsonl"])
            up_btn = gr.Button("Upload and validate", variant="primary")
            up_md = gr.Markdown()
        validation_md = gr.Markdown()
        with gr.Accordion("Preview a table", open=False):
            from nbx.fsdr import schemas

            with gr.Row(equal_height=True):
                prev_table = gr.Dropdown(list(schemas.TABLES), label="Table", scale=3)
                prev_limit = gr.Number(label="Rows", value=20, precision=0, scale=1)
                prev_btn = gr.Button("Preview", scale=1)
            prev_info = gr.Markdown()
            prev_df = gr.Dataframe(interactive=False, wrap=True)

    with gr.Tab("FSDR Training & Scoring", id="fsdr-train"):
        gr.HTML('<div class="nbx-muted">Weekly NBA cycle: train the Vowpal Wabbit bandit on last week\'s targeting and outcomes '
                "(T-anchored tables), then score the test sample (free / cstr / llm arms) with the champion model "
                "(D-anchored tables).</div>")
        train_run = gr.Dropdown(**run_kw)
        with gr.Row():
            t_mode = gr.Radio(["logged", "simulation"], value=config.TRAIN_MODE, label="Training mode")
            t_reward = gr.Dropdown(config.REWARD_DEFINITIONS, value=config.REWARD_DEFINITION, label="Reward")
            t_promote = gr.Radio(["auto", "never", "always"], value="auto", label="Promote")
        with gr.Row():
            t_passes = gr.Slider(*config.TRAIN_PASSES_RANGE, value=config.TRAIN_PASSES, step=1, label="Passes")
            t_eps = gr.Slider(*config.VW_EPSILON_RANGE, value=config.VW_EPSILON, step=0.01, label="Exploration epsilon")
        train_btn = gr.Button("Train", variant="primary")
        train_html = gr.HTML()
        gr.Markdown("---")
        with gr.Row():
            s_pop = gr.Radio(["target", "base"], value="target", label="Population")
            s_policy = gr.Radio(["sample", "greedy"], value=config.SCORING_POLICY, label="Policy")
            s_stale = gr.Checkbox(label="Allow a stale model", value=False)
        score_btn = gr.Button("Score", variant="primary")
        score_html = gr.HTML()
        gr.Markdown("---")
        with gr.Row(equal_height=True):
            models_btn = gr.Button("Refresh registry", scale=1)
            promote_box = gr.Textbox(label="Version to promote", placeholder="v0002", scale=2)
            promote_btn = gr.Button("Promote", scale=1)
            rollback_btn = gr.Button("Roll back", variant="stop", scale=1)
        models_html = gr.HTML()

    with gr.Tab("FSDR Offers", id="fsdr-offers"):
        gr.HTML('<div class="nbx-muted">Offers per arm. <b>cstr</b>: the NBA action restricts services and generosity (BR-66); '
                "<b>free</b>: L2 DATA + COMBO only (BR-67); both use the ARPU anchor and the 3-pass price bands. "
                "<b>llm</b>: the action, offer and competitor agents decide (rule-based when no LLM is connected).</div>")
        with gr.Row(equal_height=True):
            offers_run = gr.Dropdown(**run_kw)
            o_arm = gr.Radio(["cstr", "free", "llm"], value="cstr", label="Arm", scale=2)
            o_map = gr.Radio(["all_L2", "normal"], value=config.GENEROSITY_MAPPING, label="Generosity mapping", scale=2)
            offers_btn = gr.Button("Recommend", variant="primary", scale=1)
        offers_html = gr.HTML()
        offers_file = gr.File(label="Download offers (CSV)", interactive=False)

    with gr.Tab("FSDR Agentic Recommender", id="fsdr-agents"):
        gr.HTML('<div class="nbx-muted"><b>Action agent → Offer agent → Competitor agent.</b> The bandit, the segment policy and the '
                "price band are evidence the agents call as tools; they may override them with a written reason. Hard rules "
                "(eligibility, churn block, offer limits, conflicts) are enforced in code, and every decision is logged with its "
                "propensity.</div>")
        with gr.Row(equal_height=True):
            agent_run = gr.Dropdown(**run_kw)
            agent_pick = gr.Dropdown(label="…llm-arm customer (sorted by churn)", choices=[], scale=4)
            agent_msisdn = gr.Textbox(label="MSISDN", scale=2)
            agent_btn = gr.Button("Run the agents", variant="primary", scale=1)
        agent_pick.change(lambda v: gr.update(value=v or ""), inputs=[agent_pick], outputs=[agent_msisdn])
        pipeline = gr.HTML()
        with gr.Row():
            with gr.Column(scale=5):
                trace = gr.HTML()
            with gr.Column(scale=7):
                decision = gr.HTML()
        competitor = gr.HTML()
        message = gr.HTML()
        with gr.Accordion("Batch: action → top 4 offers → top 2 roaming → international offers", open=True):
            gr.HTML('<div class="nbx-muted">One click for a list of customers, processed batch by batch: (1) the action, '
                    "(2) the top 4 offers for it, (3) the top 2 roaming packs, (4) international packs only if the customer "
                    "made international calls last month. Leave the list empty to take the first customers of the arm.</div>")
            with gr.Row(equal_height=True):
                b_text = gr.Textbox(label="MSISDN list (optional)", lines=2, scale=4)
                b_arm = gr.Radio(["llm", "cstr", "free"], value="llm", label="Arm", scale=2)
            with gr.Row(equal_height=True):
                b_size = gr.Number(label="Batch size", value=config.BATCH_SIZE, precision=0)
                b_limit = gr.Number(label="Max customers", value=100, precision=0)
                b_btn = gr.Button("Recommend for the batch", variant="primary")
            b_status = gr.Markdown()
            b_table = gr.HTML()
            b_file = gr.File(label="Download batch (CSV)", interactive=False)
        b_btn.click(run_batch_flow, inputs=[agent_run, b_text, b_arm, b_size, b_limit], outputs=[b_status, b_table, b_file])

    with gr.Tab("FSDR Simulation", id="fsdr-sim"):
        gr.HTML('<div class="nbx-muted"><b>Closed loop on the hidden world.</b> Each week: generate the data, collect the rewards of '
                "last week's campaign, train on them (logged propensity), score, recommend for every arm and log it. The control "
                "group (cg) is never contacted, so each arm's uplift is measured against it. Runs in "
                "<code>data/fsdr/simulation/</code>, apart from your runs.</div>")
        with gr.Row():
            sim_start = gr.Textbox(label="First week D (YYYY-MM-DD)", value=date.today().isoformat())
            sim_weeks = gr.Slider(1, 8, value=3, step=1, label="Weeks")
            sim_n = gr.Number(label="Lines", value=2000, precision=0)
            sim_seed = gr.Number(label="Seed", value=config.GENERATOR_SEED, precision=0)
        with gr.Row():
            sim_scen = gr.CheckboxGroup(config.SCENARIOS, value=["baseline"], label="Scenarios")
            sim_arms = gr.CheckboxGroup(["free", "cstr", "llm"], value=["free", "cstr", "llm"], label="Arms (besides the control cg)")
        sim_btn = gr.Button("Run the simulation", variant="primary")
        sim_html = gr.HTML()
        gr.Markdown("---")
        gr.HTML('<div class="nbx-muted"><b>Weekly cycle on your runs.</b> Feedback: pick the run of the week <i>after</i> a campaign — '
                "its fulfilments hold the responses. Report: the campaign week (its run date).</div>")
        with gr.Row(equal_height=True):
            fb_run = gr.Dropdown(**run_kw)
            fb_btn = gr.Button("Collect feedback", variant="primary", scale=1)
        with gr.Row(equal_height=True):
            ops_kind = gr.Radio(["data", "training", "scoring", "monitor"], value="data", label="Operations agent (on the run above)", scale=3)
            ops_btn = gr.Button("Run the agent", scale=1)
        with gr.Row(equal_height=True):
            rp_week = gr.Textbox(label="Campaign week (YYYY-MM-DD)", scale=3)
            rp_btn = gr.Button("Weekly report", scale=1)
        with gr.Row(equal_height=True):
            log_arm = gr.Radio(["all", "cg", "free", "cstr", "llm"], value="all", label="Campaign log: arm", scale=3)
            log_limit = gr.Number(label="Rows", value=50, precision=0, scale=1)
            log_btn = gr.Button("Show the campaign log", scale=1)
        cycle_html = gr.HTML()

    up_btn.click(do_upload, inputs=[up_date, up_files, up_force], outputs=[up_md])
    sim_btn.click(do_simulate, inputs=[sim_start, sim_weeks, sim_n, sim_scen, sim_arms, sim_seed], outputs=[sim_html])
    fb_btn.click(do_feedback, inputs=[fb_run], outputs=[cycle_html])
    ops_btn.click(do_ops_agent, inputs=[ops_kind, fb_run], outputs=[cycle_html])
    rp_btn.click(do_report, inputs=[rp_week], outputs=[cycle_html])
    log_btn.click(do_campaign_log, inputs=[log_arm, log_limit], outputs=[cycle_html])
    refresh_btn.click(load_runs, outputs=[data_run, train_run, offers_run, agent_run, fb_run, runs_html])
    gen_btn.click(do_generate, inputs=[gen_date, gen_n, gen_scen, gen_seed], outputs=[gen_md])
    validate_btn.click(do_validate, inputs=[data_run], outputs=[validation_md])
    prev_btn.click(do_preview, inputs=[data_run, prev_table, prev_limit], outputs=[prev_df, prev_info])
    train_btn.click(do_train, inputs=[train_run, t_mode, t_reward, t_passes, t_eps, t_promote], outputs=[train_html])
    score_btn.click(do_score, inputs=[train_run, s_pop, s_policy, s_stale], outputs=[score_html])
    models_btn.click(do_models, outputs=[models_html])
    promote_btn.click(do_promote, inputs=[promote_box], outputs=[models_html])
    rollback_btn.click(do_rollback, outputs=[models_html])
    offers_btn.click(do_recommend, inputs=[offers_run, o_arm, o_map], outputs=[offers_html, offers_file])
    agent_run.change(load_customers, inputs=[agent_run], outputs=[agent_pick])
    agent_btn.click(run_decision, inputs=[agent_run, agent_msisdn], outputs=[pipeline, trace, decision, competitor, message])
    agent_msisdn.submit(run_decision, inputs=[agent_run, agent_msisdn], outputs=[pipeline, trace, decision, competitor, message])
    return {"load_outputs": [data_run, train_run, offers_run, agent_run, fb_run, runs_html]}
