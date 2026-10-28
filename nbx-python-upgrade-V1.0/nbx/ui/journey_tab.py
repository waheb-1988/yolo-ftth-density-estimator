"""FSDR Journey tab: the NBA / NBO decision step by step, for one MSISDN or a batch.

    1 Upload dummy data  -> every contract table, green when it is complete; joins and recomputed calculations
    2 Best action        -> the action and why (the LLM's own words, the bandit evidence, the guards)
    3 Best offers        -> the top 4 offers
    4 Roaming / intl     -> add-on packs
    5 Competitors        -> each offer against the closest competitor plans
    6 Output             -> everything about the line, saved (JSON per line, CSV per batch)

Every button calls the API (/fsdr/journey/*); the switch "Single MSISDN / Batch" applies to steps 2-6.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from nbx.fsdr.pipelines.journey import parse_msisdns
from nbx.ui import theme as ui
from nbx.ui.fsdr_tabs import _get, _post, _section

esc = ui.esc
STATUS = {"green": ("green", "OK"), "yellow": ("amber", "warning"), "red": ("red", "blocked"), "grey": ("", "optional · not provided")}


def _badge(status: str, text: str | None = None) -> str:
    cls, label = STATUS.get(status, ("", status))
    return f'<span class="nbx-badge {cls}">{esc(text or label)}</span>'


def _htable(rows: list[dict], cols: list[str], labels: dict | None = None, html: set[str] = frozenset()) -> str:
    """A table whose `html` columns are already HTML (badges); everything else is escaped."""
    if not rows:
        return '<div class="nbx-muted">Nothing to show.</div>'
    head = "".join(f"<th>{esc((labels or {}).get(c, c))}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{r.get(c, '') if c in html else esc(_fmt(r.get(c, '')))}</td>" for c in cols) + "</tr>"
                   for r in rows)
    return f'<div style="overflow-x:auto"><table class="nbx-table"><tr>{head}</tr>{body}</table></div>'


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else v


def _download(name: str, payload) -> str:
    path = Path(tempfile.gettempdir()) / name
    if isinstance(payload, str) and Path(payload).exists():
        shutil.copyfile(payload, path)
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# 1. Upload
# ---------------------------------------------------------------------------
def _lines(rep: dict) -> list[str]:
    return [v for arm in ("llm", "cstr", "free") for v in (rep.get("lines") or {}).get(arm, [])]


async def _latest() -> dict:
    try:
        return await _get("/fsdr/journey/latest") or {}
    except Exception:  # noqa: BLE001  (API not up yet)
        return {}


async def load_sources():
    """On page load: the dummy-data folders, and the last upload restored so steps 2-6 work after a reload."""
    try:
        srcs = await _get("/fsdr/journey/sources")
    except Exception:  # noqa: BLE001  (API not up yet)
        srcs = []
    names = [s["source"] for s in srcs]
    rep = await _latest()
    lines = _lines(rep)
    note = (f"<p>{_badge('green', 'last upload restored')} <span class='nbx-muted'>run <b>{esc(rep['run'])}</b> from "
            f"<code>{esc(rep.get('source', ''))}</code> · steps 2-6 continue on it; click 1 again to reload the data.</p>"
            if rep.get("run") else "")
    return (gr.update(choices=names, value=names[-1] if names else None),
            gr.update(choices=lines, value=lines[0] if lines else None), rep.get("run", ""), note)


async def do_ingest(source: str):
    if not source:
        raise gr.Error("No dummy-data folder: put one under data/fsdr_dummy/<YYYY-MM-DD>/.")
    r = await _post("/fsdr/journey/ingest", {"source": source})
    counts = {s: sum(1 for t in r["tables"] if t["status"] == s) for s in ("green", "yellow", "red", "grey")}
    head = (f"<p>{_badge('green' if r['same_data'] else 'red', 'same data as the source (SHA-256 identical)' if r['same_data'] else 'COPY DIFFERS')} "
            f"{_badge('green', str(counts['green']) + ' tables OK')} "
            + (f"{_badge('yellow', str(counts['yellow']) + ' warning')} " if counts["yellow"] else "")
            + (f"{_badge('red', str(counts['red']) + ' blocked')} " if counts["red"] else "")
            + (f"{_badge('grey', str(counts['grey']) + ' optional, not provided')}" if counts["grey"] else "")
            + f"</p><p class='nbx-muted'>{r['tables_copied']} of {r['tables_in_contract']} contract tables copied from "
              f"<code>{esc(r['source'])}</code> to <code>{esc(r['destination'])}</code> · validation <b>{esc(r['validation_status'])}</b> · "
              f"{r['seconds']} s</p>")
    tables = [{**t, "badge": _badge(t["status"])} for t in r["tables"]]
    html = _section("1 · Tables (data contract)", head + _htable(
        tables, ["badge", "table", "group", "used_by", "rows", "fields", "missing_fields", "identical_copy", "sha256", "issues"],
        {"badge": "status", "used_by": "used for", "identical_copy": "same bytes", "sha256": "SHA-256 (16)"}, {"badge"}))
    joins = [{**j, "badge": _badge(j["status"]), "match_rate": f"{j['match_rate']:.1%}"} for j in r["joins"]]
    html += _section("2 · Joins the NBA agent relies on", _htable(
        joins, ["badge", "join", "why", "left_keys", "matched", "match_rate"], {"badge": "status"}, {"badge"}))
    if r.get("training"):
        tr, sc = r["training"], r["scoring"]
        html += _section("3 · Trained and scored on this data", _htable([
            {"step": "training", "result": f"model {tr['model']} ({'promoted' if tr['promoted'] else 'not promoted'}: {tr['rule']})",
             "detail": f"{tr['labelled_rows']} labelled lines ({tr['label_source']}), {tr['features']} features"},
            {"step": "scoring", "result": f"model {sc['model']}, {sc['scored_lines']} lines scored",
             "detail": f"sample groups {sc['groups']} (cg = control, not scored) · max PSI {sc['max_psi']}"}], ["step", "result", "detail"]))
        calcs = [{**c, "badge": _badge(c["status"])} for c in r["calculations"]]
        html += _section("4 · Calculations recomputed from the raw tables", _htable(
            calcs, ["badge", "check", "rule", "lines_checked", "mismatches", "example"], {"badge": "status"}, {"badge"}))
    lines = _lines(r)
    return html, gr.update(choices=lines, value=lines[0] if lines else None), r["run"], _download(f"ingestion_{r['run']}.json", r)


# ---------------------------------------------------------------------------
# 2-6. Steps
# ---------------------------------------------------------------------------
def _targets(mode: str, msisdn: str, pasted: str, file) -> list[str]:
    if mode == "Single MSISDN":
        v = (msisdn or "").strip()
        if not v:
            raise gr.Error("Pick or type an MSISDN.")
        return [v]
    text = pasted or ""
    if file:
        text += "\n" + Path(file if isinstance(file, str) else file.name).read_text(encoding="utf-8-sig", errors="replace")
    vals = parse_msisdns(text)
    if not vals:
        raise gr.Error("Batch: upload a .csv / .txt with an 'msisdn' column (or one MSISDN per line), or paste a list.")
    return vals


async def _call(run: str, step: str, mode, msisdn, pasted, file, llm):
    run = run or (await _latest()).get("run")                # page reloaded: continue on the last upload
    if not run:
        raise gr.Error("Step 1 first: upload the data.")
    return await _post(f"/fsdr/journey/{run}/{step}", {"msisdns": _targets(mode, msisdn, pasted, file), "llm": llm or None})


def _batch_view(res: dict, cols: list[str], rows: list[dict]) -> tuple[str, str | None]:
    ok = res["count"] - res["errors"]
    head = (f"<p>{_badge('green', f'{ok} lines done')} " + (_badge('red', f"{res['errors']} errors") if res["errors"] else "")
            + f" <span class='nbx-muted'>{res['seconds']} s</span></p>")
    f = res.get("files", {}).get("csv")
    return head + _htable(rows, cols), (_download(Path(f).name, f) if f else None)


def _mode_badge(r: dict) -> str:
    return _badge("green", f"LLM {r['llm_model']}") if r.get("llm_model") else _badge("yellow", "rule-based (no LLM reachable)")


async def do_action(run, mode, msisdn, pasted, file, llm):
    res = await _call(run, "action", mode, msisdn, pasted, file, llm)
    if mode != "Single MSISDN":
        rows = [{"msisdn": r["msisdn"], "action": r.get("action"), "p": r.get("propensity"), "bandit top": r.get("bandit_top_action"),
                 "source": r.get("policy_source"), "why": " | ".join(r.get("llm_answer", {}).get("thoughts") or []) or r.get("explanation"),
                 "error": r.get("error", "")} for r in res["results"]]
        return _batch_view(res, ["msisdn", "action", "p", "bandit top", "source", "why", "error"], rows)
    r = res["results"][0]
    if r.get("error"):
        raise gr.Error(r["error"])
    a = r["llm_answer"]
    words = "".join(f"<li>{esc(t)}</li>" for t in a["thoughts"]) or "<li class='nbx-muted'>No LLM answer: the agent ran its rule-based path.</li>"
    llm_html = (f"<ul>{words}</ul>"
                + (f"<p><b>LLM chose:</b> {esc(a['llm_action'])} · <b>confidence</b> {esc(_fmt(a['confidence']))} · "
                   f"<b>expected acceptance</b> {esc(_fmt(a['expected_acceptance']))}</p>" if a.get("llm_action") else "")
                + (f"<p><b>Override reason</b> (differs from the bandit): {esc(a['override_reason'])}</p>" if a.get("override_reason") else "")
                + (f"<p><b>Signals used:</b> {esc(', '.join(a['signals_used']))}</p>" if a.get("signals_used") else ""))
    pmf = [{"action": k, "probability": v, "": "◀ chosen" if k == r["action"] else ""} for k, v in
           sorted(r["bandit_pmf"].items(), key=lambda kv: -kv[1])]
    cp = r["contact_policy"]
    html = _section(f"Best action for {r['msisdn']}", f"<p><span class='nbx-badge red' style='font-size:14px'>{esc(r['action'])}</span> "
                    f"{_mode_badge(r)} <span class='nbx-muted'>policy source {esc(r['policy_source'])} · logged propensity "
                    f"{r['propensity']} · bandit top {esc(r['bandit_top_action'])}</span></p>")
    html += _section("Why this action — the LLM's answer", llm_html)
    html += _section("Why this action — the evidence in plain words", f"<p>{esc(r['explanation'])}</p>")
    html += _section("Bandit probabilities (VW, sampled per BR-57)", _htable(pmf, ["action", "probability", ""]))
    html += _section("Business rules on the action", _htable([
        {"rule": "contact allowed (eligibility, 2-day gap, 3 contacts / 28 days, consent)", "result": cp["verdict"]},
        {"rule": "segment policy (BR-39)", "result": f"{r['segment_policy'].get('segment')} -> {r['segment_policy'].get('segment_action')}"},
        {"rule": "guards applied by the code", "result": "; ".join(r["guards"]) or "none"}], ["rule", "result"]))
    html += _section("Model scores of the line", _htable([{"score": k, "value": v} for k, v in r["scores"].items()], ["score", "value"]))
    return html, None


async def do_offers(run, mode, msisdn, pasted, file, llm):
    res = await _call(run, "offers", mode, msisdn, pasted, file, llm)
    if mode != "Single MSISDN":
        rows = [{"msisdn": r["msisdn"], "action": r.get("action"), "error": r.get("error", ""),
                 **{f"offer {i + 1}": f"{o['offer']} ({o['price']:g} OMR)" for i, o in enumerate((r.get("offers") or [])[:4])}}
                for r in res["results"]]
        return _batch_view(res, ["msisdn", "action", "offer 1", "offer 2", "offer 3", "offer 4", "error"], rows)
    r = res["results"][0]
    if r.get("error"):
        raise gr.Error(r["error"])
    html = _section(f"Top {len(r['offers'])} offers for {r['msisdn']} · action {r['action']}",
                    f"<p>{_mode_badge(r)} <span class='nbx-muted'>offer policy {esc(r['offer_policy'])}</span></p>" + _htable(
                        r["offers"], ["offer", "service", "generosity", "validity", "price", "gb", "minutes", "anchor", "p_accept",
                                      "expected_value_omr", "selected_by", "sms_en"],
                        {"anchor": "ARPU anchor", "p_accept": "P(accept)", "expected_value_omr": "value (OMR)"}))
    a = r["llm_answer"]
    words = "".join(f"<li>{esc(t)}</li>" for t in a["thoughts"]) or "<li class='nbx-muted'>No LLM answer: rule-based offers.</li>"
    html += _section("Why these offers — the LLM's answer", f"<ul>{words}</ul>" + (f"<p><b>Rationale:</b> {esc(a['rationale'])}</p>" if a["rationale"] else "")
                     + (f"<p><b>Override reason:</b> {esc(a['override_reason'])}</p>" if a["override_reason"] else ""))
    html += _section("Business rules on the offers", f"<p>{esc(r['explanation'])}</p><p class='nbx-muted'>Rule-based offers: "
                     f"{esc(', '.join(r['rule_based_offers']) or 'none')} · guards: {esc('; '.join(r['guards']) or 'none')}</p>")
    return html, None


async def do_addons(run, mode, msisdn, pasted, file, llm):
    res = await _call(run, "addons", mode, msisdn, pasted, file, llm)
    if mode != "Single MSISDN":
        rows = [{"msisdn": r["msisdn"], "roaming": ", ".join(o["offer"] for o in r.get("roaming") or []),
                 "international": ", ".join(o["offer"] for o in r.get("international") or []) or r.get("international_reason", ""),
                 "error": r.get("error", "")} for r in res["results"]]
        return _batch_view(res, ["msisdn", "roaming", "international", "error"], rows)
    r = res["results"][0]
    if r.get("error"):
        raise gr.Error(r["error"])
    cols = ["offer", "service", "generosity", "validity", "price", "gb", "minutes", "score"]
    html = _section(f"Top roaming packs for {r['msisdn']} (basis: {r.get('roaming_basis')})", _htable(r["roaming"], cols))
    html += _section(f"International packs — {r['international_reason']}", _htable(r["international"], cols))
    html += _section("Rules", f"<p class='nbx-muted'>{esc(r['rules'])}</p>")
    return html, None


async def do_competitors(run, mode, msisdn, pasted, file, llm):
    res = await _call(run, "competitors", mode, msisdn, pasted, file, llm)
    if mode != "Single MSISDN":
        rows = [{"msisdn": r["msisdn"], "verdict": r.get("verdict"), "summary": r.get("summary"),
                 "swaps": "; ".join(f"{s['out']} -> {s['in']}" for s in r.get("swaps_applied") or []), "error": r.get("error", "")}
                for r in res["results"]]
        return _batch_view(res, ["msisdn", "verdict", "summary", "swaps", "error"], rows)
    r = res["results"][0]
    if r.get("error"):
        raise gr.Error(r["error"])
    status = {"competitive": "green", "mixed": "yellow", "uncompetitive": "red"}.get(r["verdict"], "grey")
    html = _section(f"Competitor comparison for {r['msisdn']}", f"<p>{_badge(status, r['verdict'] or 'no offer')} {_mode_badge(r)}</p>"
                    f"<p>{esc(r['summary'])}</p>")
    for c in r["comparisons"]:
        html += _section(f"{c['offer']} · our monthly equivalent {c['our_monthly_equiv']} OMR · {c['position']}",
                         _htable(c["closest"], ["operator", "plan", "monthly_equiv", "data_gb", "monthly_delta", "verdict"],
                                 {"monthly_equiv": "monthly equiv. (OMR)", "monthly_delta": "our price - theirs"}))
    words = "".join(f"<li>{esc(t)}</li>" for t in r["llm_answer"]["thoughts"])
    if words:
        html += _section("The LLM's answer", f"<ul>{words}</ul>")
    if r["swaps_applied"]:
        html += _section("Swaps applied (checked against the hard rules)", _htable(r["swaps_applied"], ["out", "in", "reason"]))
    return html, None


async def do_output(run, mode, msisdn, pasted, file, llm):
    res = await _call(run, "output", mode, msisdn, pasted, file, llm)
    if mode != "Single MSISDN":
        rows = [{"msisdn": r["msisdn"], "action": (r.get("summary") or {}).get("action"), "offer_1": (r.get("summary") or {}).get("offer_1"),
                 "roaming_1": (r.get("summary") or {}).get("roaming_1"), "competitor": (r.get("summary") or {}).get("competitor_verdict"),
                 "file": r.get("file", ""), "error": r.get("error", "")} for r in res["results"]]
        html, f = _batch_view(res, ["msisdn", "action", "offer_1", "roaming_1", "competitor", "file", "error"], rows)
        saved = (f"<p class='nbx-muted'>Saved: {esc(res['files'].get('csv'))} and {esc(res['files'].get('json'))} "
                 "(one row per line with every column; one JSON file per line).</p>")
        return html + saved, f
    r = res["results"][0]
    if r.get("error"):
        raise gr.Error(r["error"])
    s = r["summary"]
    ctx = r["output"]["context"]
    key = [{"field": k, "value": v} for k, v in s.items() if not k.startswith(("feat_", "score_", "anchor_", "base_"))]
    html = _section(f"Output for {r['msisdn']} — saved to {r['file']}", _htable(key, ["field", "value"]))
    html += _section(f"Raw table rows of the line ({len(ctx['raw_tables'])} tables)", "".join(
        f"<details><summary>{esc(t)} · {len(rows)} row(s)</summary>{_htable(rows, list(rows[0]))}</details>"
        for t, rows in ctx["raw_tables"].items()))
    for name, block in ctx["engine"].items():
        if isinstance(block, dict) and block:
            html += f"<details><summary>engine · {esc(name)} · {len(block)} columns</summary>" \
                    f"{_htable([{'column': k, 'value': v} for k, v in block.items()], ['column', 'value'])}</details>"
    return html, _download(Path(r["file"]).name, r["file"])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def build() -> dict:
    with gr.Tab("FSDR Journey", id="fsdr-journey"):
        gr.HTML('<div class="nbx-muted">The NBA / NBO decision step by step. <b>1</b> loads the dummy data (every table of the '
                "data contract, checked), trains and scores on it. <b>2–6</b> run the agents for one MSISDN or a batch "
                "(switch below): each step continues from the previous one and everything is saved under "
                "<code>data/fsdr/&lt;date&gt;/outputs/journey/</code>.</div>")
        run_state = gr.State("")
        with gr.Row(equal_height=True):
            source = gr.Dropdown(label="Dummy data folder (data/fsdr_dummy/<date>)", choices=[], scale=4)
            ingest_btn = gr.Button("1 · Upload data", variant="primary", scale=1)
        ingest_file = gr.File(label="Ingestion report (JSON)", interactive=False)
        ingest_html = gr.HTML()

        with gr.Row(equal_height=True):
            mode = gr.Radio(["Single MSISDN", "Batch"], value="Single MSISDN", label="Mode", scale=2)
            msisdn = gr.Dropdown(label="MSISDN (scored line; pick or type)", choices=[], allow_custom_value=True, scale=3)
            llm = gr.Textbox(label="LLM model (vLLM; empty = default)", placeholder="qwen2.5:3b", scale=2)
        with gr.Row(visible=False) as batch_row:
            batch_file = gr.File(label="Batch: .csv / .txt with an 'msisdn' column (or one per line)", file_types=[".csv", ".txt"])
            batch_text = gr.Textbox(label="… or paste MSISDNs (one per line or comma-separated)", lines=4)
        mode.change(lambda m: gr.update(visible=m == "Batch"), inputs=mode, outputs=batch_row)

        inputs = [run_state, mode, msisdn, batch_text, batch_file, llm]
        steps = [("2 · Best action recommender", do_action), ("3 · Best offer recommender (top 4)", do_offers),
                 ("4 · Roaming & international recommender", do_addons), ("5 · Competitor comparison", do_competitors),
                 ("6 · Full output (save)", do_output)]
        for label, fn in steps:
            with gr.Group():
                btn = gr.Button(label, variant="primary" if label.startswith("2") else "secondary")
                out = gr.HTML()
                dl = gr.File(label="Download", interactive=False)
            btn.click(fn, inputs=inputs, outputs=[out, dl])
        ingest_btn.click(do_ingest, inputs=source, outputs=[ingest_html, msisdn, run_state, ingest_file])
    return {"load": load_sources, "load_outputs": [source, msisdn, run_state, ingest_html]}
