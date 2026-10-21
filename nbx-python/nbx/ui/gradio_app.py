"""NBX Gradio app (Ooredoo theme): landing page, Agentic Best Action Recommender,
Agentic Offer Recommender, Competitor Intelligence, plus the data/ops tabs.
Talks to the FastAPI backend over HTTP; presentation lives in nbx/ui/theme.py."""
from __future__ import annotations

import asyncio
import json

import httpx
import gradio as gr

from nbx.ui import theme as ui
from nbx.core import config

API_BASE = f"http://localhost:{config.FASTAPI_PORT}"
LLM_CHOICES = [m["name"] for m in config.OLLAMA_MODELS]
STORE_CHOICES = config.VECTOR_STORE_KEYS
EMBEDDING_CHOICES = config.EMBEDDING_MODEL_KEYS

STEP_ICON = {"pending": "⏳", "running": "\U0001f504", "done": "✅", "failed": "❌"}
UPLOAD_POLL_SECONDS = 0.7
OLLAMA_GATE_NOTE = (
    "_LLM backend not connected -- start it (`ollama serve`, or `vllm serve` when LLM_BACKEND=vllm) to enable this tab. "
    "Every other tab works fine without it._"
)


async def _get(path: str, **params) -> dict:
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.get(f"{API_BASE}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{API_BASE}{path}", json=body)
        if resp.status_code >= 400:
            detail = resp.json().get("detail", resp.text)
            raise gr.Error(f"{path} failed: {detail}")
        return resp.json()


def _to_table(rows: list[dict], columns: list[str] | None = None) -> dict:
    """Builds a plain {headers, data} dict for gr.Dataframe. Avoids pandas,
    which segfaults in this environment when loaded alongside torch/sentence-transformers."""
    headers = columns or (list(rows[0].keys()) if rows else [])
    data = [[row.get(col) for col in headers] for row in rows]
    return {"headers": headers, "data": data}


# ---------------------------------------------------------------------------
# Upload & Setup helpers
# ---------------------------------------------------------------------------
def _read_file_bytes(file_obj) -> bytes | None:
    """gr.File (default type='filepath') hands back a local temp-file path
    string, or None if nothing was selected."""
    if file_obj is None:
        return None
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        return None
    with open(path, "rb") as f:
        return f.read()


async def _post_files(path: str, files: dict[str, bytes]) -> dict:
    async with httpx.AsyncClient(timeout=180.0) as client:
        multipart = {k: (f"{k}.json", v, "application/json") for k, v in files.items()}
        resp = await client.post(f"{API_BASE}{path}", files=multipart)
        if resp.status_code >= 400:
            detail = resp.json().get("detail", resp.text)
            raise gr.Error(f"{path} failed: {detail}")
        return resp.json()


def _render_upload_status(status: dict) -> str:
    lines = [f"**Overall:** {status['overall_status']}", ""]
    for s in status["steps"]:
        icon = STEP_ICON.get(s["status"], "•")
        line = f"{icon} **{s['display']}**"
        if s["status"] == "done" and s["message"]:
            line += f" — {s['message']}"
        elif s["status"] == "failed" and s["error"]:
            line += f" — {s['error']}"
        elif s["status"] == "running":
            line += " — running..."
        lines.append(line)
    return "\n\n".join(lines)


def _gating_from_health(health: dict) -> tuple:
    ok = health["ollama_connected"]
    return gr.update(interactive=ok), gr.update(value="" if ok else OLLAMA_GATE_NOTE, visible=not ok)


_NO_GATE_CHANGE = (gr.update(), gr.update())  # Benchmark button + its note


# ---------------------------------------------------------------------------
# Tab callbacks
# ---------------------------------------------------------------------------
async def load_customer_choices():
    try:
        rows = await _get("/customers")
    except httpx.HTTPError:
        return gr.update(), gr.update()
    rows.sort(key=lambda r: r["churn"], reverse=True)
    choices = [(ui.describe(r), r["msisdn"]) for r in rows]
    return gr.update(choices=choices, value=None), gr.update(choices=choices, value=None)


def parse_msisdns(text) -> tuple[list[int], list[str]]:
    """'15, 7 22\n31' -> ([15, 7, 22, 31], []). Order kept, duplicates dropped,
    non-numeric tokens returned separately so they can be reported."""
    import re

    ids, bad = [], []
    for tok in re.split(r"[\s,;]+", str(text or "").strip()):
        if not tok:
            continue
        digits = tok.lstrip("+")
        if digits.isdigit():
            if int(digits) not in ids:
                ids.append(int(digits))
        else:
            bad.append(tok)
    return ids, bad


async def _customer_360(msisdn: int) -> dict | None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{API_BASE}/customers/{msisdn}/360")
        return resp.json() if resp.status_code == 200 else None


async def run_agentic(msisdn_text, scope: str, use_last_nba: bool = False):
    """Shows the customer card, then streams one council run (GET /agentic/stream)
    and re-renders after every event."""
    ids, _ = parse_msisdns(msisdn_text)
    if len(ids) != 1:
        raise gr.Error("Type one MSISDN (use the batch section below for a list).")
    msisdn = ids[0]
    try:
        customer = await _customer_360(msisdn)
    except httpx.HTTPError as e:
        raise gr.Error(f"Could not reach API at {API_BASE}: {e}")
    if customer is None:
        raise gr.Error(f"No customer with MSISDN {msisdn}.")
    profile = ui.render_profile(customer)
    events: list[dict] = []
    empty = ""
    yield profile, ui.render_pipeline([], True), '<div class="nbx-muted">Starting the agents…</div>', empty, empty, empty, empty, empty
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            async with client.stream("GET", f"{API_BASE}/agentic/stream", params={"msisdn": int(msisdn), "scope": scope, "use_last_nba": use_last_nba}) as resp:
                if resp.status_code >= 400:
                    raise gr.Error(f"Backend refused the run: {(await resp.aread()).decode()[:300]}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    events.append(ev)
                    if ev["type"] == "error":
                        raise gr.Error(ev.get("detail", "Agent run failed"))
                    if ev["type"] == "run_end":
                        d = ev["decision"]
                        yield (profile, ui.render_pipeline(events, False), ui.render_trace(events), ui.render_action(d),
                               ui.render_offers(d) if scope != "nba" else empty, ui.render_why(d), ui.render_message(d), ui.render_stats(d))
                        return
                    if ev["type"] in ("tool_call", "tool_result") and len(events) % 2:
                        continue  # halve re-renders during tool bursts
                    yield profile, ui.render_pipeline(events, True), ui.render_trace(events), empty, empty, empty, empty, empty
    except httpx.HTTPError as e:
        raise gr.Error(f"Could not reach API at {API_BASE}: {e}")


async def run_best_action(msisdn):
    async for out in run_agentic(msisdn, "nba"):
        yield out


async def run_offer(msisdn):
    """Step 1 (Best Action agent) then Step 2 (Offer agent) in one run."""
    async for out in run_agentic(msisdn, "nbo"):
        yield out


async def run_offer_from_last_action(msisdn):
    """Step 2 only: offers for the action just decided in this tab (the backend
    reuses that NBA decision for 30 minutes; otherwise it re-runs Step 1)."""
    async for out in run_agentic(msisdn, "nbo", use_last_nba=True):
        yield out


BATCH_LIMIT = 50
BATCH_CONCURRENCY = 2  # matches the LLM server's parallel slots (llama.cpp -np 2)


def _batch_row(msisdn: int, customer: dict, d: dict) -> dict:
    expl = d.get("explanation") or {}
    lead = (expl.get("offers") or [{}])[0]
    narrative = expl.get("narrative", "") or d.get("nba_rationale", "")
    return {
        "msisdn": msisdn,
        "description": ui.describe(customer),
        "action": d["action"],
        "channel": f"{d.get('channel')} · {d.get('send_window')} · {' / '.join(d['message'].get('languages') or [])}",
        "languages": " + ".join(d["message"].get("languages") or []),
        "sms_en": d["message"].get("sms_en", ""),
        "sms_ar": d["message"].get("sms_ar", ""),
        "sms_native": d["message"].get("sms_native", ""),
        "native_language": d["message"].get("native_language") or "",
        "offers": [o["offer"] for o in d["offers"]],
        "ev": f"{d['offers'][0]['expected_value_omr']} OMR/mo" if d["offers"] else "-",
        "position": (lead.get("competition") or {}).get("position", "-"),
        "critic": d["critic"]["verdict"] + (" · revised" if d["critic"]["revised"] else ""),
        "why": (narrative[:280] + "…") if len(narrative) > 280 else narrative,
        "narrative": narrative,
        "time": f"{d['stats']['elapsed_ms'] / 1000:.0f}s",
    }


def _write_batch_csv(rows: list[dict], scope: str) -> str:
    import csv
    import tempfile

    cols = ["msisdn", "description", "action", "channel", "languages", "offers", "ev", "position", "critic", "narrative",
            "sms_en", "sms_ar", "native_language", "sms_native", "time", "error"]
    f = tempfile.NamedTemporaryFile("w", suffix=f"_nbx_{scope}_batch.csv", delete=False, newline="", encoding="utf-8-sig")
    with f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "offers": " | ".join(r.get("offers", []))})
    return f.name


async def run_batch(text, scope: str):
    """Recommends for a list of MSISDNs: BATCH_CONCURRENCY runs at a time,
    the table fills in as each customer finishes; CSV at the end."""
    ids, bad = parse_msisdns(text)
    if not ids:
        raise gr.Error("Paste at least one MSISDN (separated by commas, spaces or new lines).")
    note = ""
    if len(ids) > BATCH_LIMIT:
        note = f" Only the first {BATCH_LIMIT} are processed in the UI; use `python -m nbx.scripts.batch_score` for full runs."
        ids = ids[:BATCH_LIMIT]
    rows: dict[int, dict] = {m: {"msisdn": m, "pending": "queued"} for m in ids}
    for tok in bad:
        rows[tok] = {"msisdn": tok, "error": "not a number"}
    order = list(rows)

    def view(done: int):
        md = f"**{done} / {len(ids)}** customers done.{' Ignored: ' + ', '.join(bad) + '.' if bad else ''}{note}"
        return md, ui.render_batch([rows[k] for k in order], scope)

    yield (*view(0), None)
    sem = asyncio.Semaphore(BATCH_CONCURRENCY)

    async def one(msisdn: int):
        async with sem:
            customer = await _customer_360(msisdn)
            if customer is None:
                return msisdn, {"msisdn": msisdn, "error": "unknown MSISDN"}
            rows[msisdn] = {"msisdn": msisdn, "description": ui.describe(customer), "pending": "agents are working…"}
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(f"{API_BASE}/agentic/decide", json={"msisdn": msisdn, "scope": scope})
            if resp.status_code >= 400:
                return msisdn, {"msisdn": msisdn, "error": resp.text[:200]}
            return msisdn, _batch_row(msisdn, customer, resp.json()["decision"])

    tasks = [asyncio.create_task(one(m)) for m in ids]
    done = 0
    try:
        for fut in asyncio.as_completed(tasks):
            try:
                msisdn, row = await fut
            except httpx.HTTPError as e:
                raise gr.Error(f"Could not reach API at {API_BASE}: {e}")
            rows[msisdn] = row
            done += 1
            yield (*view(done), None)
    finally:
        for t in tasks:
            t.cancel()
    yield (*view(done), _write_batch_csv([rows[k] for k in order], scope))


async def run_batch_action(text):
    async for out in run_batch(text, "nba"):
        yield out


async def run_batch_offer(text):
    async for out in run_batch(text, "nbo"):
        yield out


async def load_landing():
    try:
        summary, comp, health = await asyncio.gather(_get("/portfolio/summary"), _get("/competitors/offers"), _get("/health"))
    except httpx.HTTPError:
        return ui.landing_html(None, 0, "backend offline"), ui.header_html("Backend offline")
    llm = ", ".join(health.get("loaded_models") or []) or "fallback mode"
    status = f"{health.get('llm_backend', '')} · {llm}" if health.get("ollama_connected") else "LLM offline · deterministic fallback"
    return ui.landing_html(summary, len(comp), llm), ui.header_html(status)


# --- Competitor intelligence -------------------------------------------------
async def comp_load(operator: str, segment: str):
    params = {}
    if operator != "All":
        params["operator"] = operator
    if segment != "All":
        params["segment"] = segment
    rows = await _get("/competitors/offers", **params)
    rows.sort(key=lambda r: (r["operator"], r.get("monthly_equiv_omr") or 0))
    snap = rows[0]["scraped_at"] if rows else "never"
    by_op = {}
    for r in rows:
        by_op[r["operator"]] = by_op.get(r["operator"], 0) + 1
    summary = (f"**{len(rows)} competitor plans** ({', '.join(f'{k}: {v}' for k, v in by_op.items()) or 'none'}) · "
               f"snapshot `{snap}` · prices as published, incl./excl. VAT as stated on each site")
    body = "".join(
        f"<tr><td>{ui.esc(r['operator'])}</td><td><b>{ui.esc(r['plan'])}</b></td><td>{ui.esc(r['category'])}</td>"
        f"<td>{r['price_omr']} OMR</td><td>{ui.esc(r['validity_days'])} d</td>"
        f"<td>{ui.esc(r['data_gb'] if r['data_gb'] is not None else 'unlimited / n.a.')}</td><td>{ui.esc(r.get('minutes') or '-')}</td>"
        f"<td>{ui.esc(r.get('monthly_equiv_omr'))} OMR</td><td>{ui.esc(r.get('price_per_gb_omr') or '-')}</td>"
        f"<td><a href='{ui.esc(r['source_url'])}' target='_blank' style='color:#ff5470'>source</a></td></tr>" for r in rows)
    table = (f'<div class="nbx-section" style="max-height:520px;overflow-y:auto"><table class="nbx-table"><tr><th>Operator</th><th>Plan</th>'
             f"<th>Type</th><th>Price</th><th>Validity</th><th>Data (GB)</th><th>Minutes</th><th>Monthly equiv.</th><th>OMR / GB</th>"
             f"<th></th></tr>{body}</table></div>")
    return summary, table


async def comp_our_offers():
    offers = await _get("/offers")
    return gr.update(choices=[(f"{o['program_rule']} · {o['category']} · {o['price_omr']} OMR / {o['validity_days']} d", o["program_rule"])
                              for o in offers], value="DT_30D_12R_20GB")


async def comp_compare(code: str):
    if not code:
        return ""
    r = await _get(f"/competitors/compare/{code}")
    rows = "".join(
        f"<tr><td>{ui.esc(c['operator'])}</td><td>{ui.esc(c['plan'])}</td><td>{c['price_omr']} OMR / {c['validity_days']} d</td>"
        f"<td>{ui.esc(c['data_gb'] if c['data_gb'] is not None else 'unlimited')} GB</td><td>{c['monthly_equiv_omr']} OMR</td>"
        f"<td>{c['monthly_delta_omr']:+.2f} OMR</td><td><b>{ui.esc(c['verdict'])}</b></td></tr>" for c in r["closest"])
    return (f'<div class="nbx-section"><h3>{ui.esc(code)} vs the market</h3>'
            f'<div style="margin-bottom:8px"><span class="nbx-badge red">{ui.esc(r["position"])}</span>'
            f'<span class="nbx-badge">ours: {r["our_monthly_equiv_omr"]} OMR / month</span>'
            f'<span class="nbx-badge">ours per GB: {r["our_price_per_gb_omr"] or "n/a"} OMR</span></div>'
            f'<table class="nbx-table"><tr><th>Operator</th><th>Plan</th><th>Price</th><th>Data</th><th>Monthly equiv.</th>'
            f'<th>Our delta</th><th>Ours is</th></tr>{rows}</table></div>')


async def comp_search(q: str):
    hits = await _get("/competitors/search", q=q, top_k=6)
    body = "".join(f"<tr><td>{h['score']:.2f}</td><td>{ui.esc(h['operator'])}</td><td>{ui.esc(h['text'][:420])}</td>"
                   f"<td><a href='{ui.esc(h['source_url'])}' target='_blank' style='color:#ff5470'>source</a></td></tr>" for h in hits)
    return (f'<div class="nbx-section"><table class="nbx-table"><tr><th>Score</th><th>Operator</th><th>Passage</th><th></th></tr>'
            f"{body}</table></div>")


async def comp_refresh():
    try:
        r = await _post("/competitors/refresh", {})
    except httpx.HTTPError as e:
        raise gr.Error(f"Could not reach API at {API_BASE}: {e}")
    errs = "".join(f"\n- ⚠ {e['url']}: {e['error']}" for e in r["errors"])
    return f"Scraped **{r['offers']} plans** and **{r['passages']} passages** ({r['indexed']} documents indexed) at `{r['scraped_at']}`.{errs}"


async def do_cohort_offers(msisdn: float):
    try:
        result = await _get(f"/graph/cohort-offers/{int(msisdn)}")
    except httpx.HTTPError as e:
        raise gr.Error(f"Could not reach API at {API_BASE}: {e}")
    return _to_table(result, ["program_rule", "graph_score", "peer_count", "reason"])


async def do_similar_customers(msisdn: float):
    try:
        result = await _get(f"/graph/similar-customers/{int(msisdn)}")
    except httpx.HTTPError as e:
        raise gr.Error(f"Could not reach API at {API_BASE}: {e}")
    return _to_table(result, ["msisdn", "shared_segment", "shared_offer_count"])


async def do_graph_stats():
    try:
        result = await _get("/graph/stats")
    except httpx.HTTPError as e:
        raise gr.Error(f"Could not reach API at {API_BASE}: {e}")
    rows = [{"kind": k, "count": v} for k, v in result["by_kind"].items()]
    summary = f"**Nodes:** {result['nodes']}  \n**Edges:** {result['edges']}"
    return summary, _to_table(rows, ["kind", "count"])


async def do_customer_lookup(msisdn: float):
    try:
        result = await _get(f"/customer/{int(msisdn)}")
    except httpx.HTTPStatusError:
        raise gr.Error(f"No customer found for msisdn={int(msisdn)}")
    except httpx.HTTPError as e:
        raise gr.Error(f"Could not reach API at {API_BASE}: {e}")

    profile_table = {"headers": ["field", "value"], "data": [[k, v] for k, v in result["customer"].items()]}
    history_table = _to_table(result["purchase_history"])
    return profile_table, history_table


async def do_benchmark(msisdn: float, store: str):
    result = await _post("/benchmark", {"msisdn": int(msisdn), "vectorstore": store})
    rows = [
        {
            "llm": r["llm"],
            "detected_need": r["detected_need"],
            "avg_final_score": r["avg_final_score"],
            "pipeline_time_ms": r["pipeline_time_ms"],
            "valid_json": r["valid_json"],
        }
        for r in result["results"]
    ]
    rows.sort(key=lambda r: r["avg_final_score"], reverse=True)
    summary = f"**Winner:** {result['winner']}  \n**Ranking:** {' > '.join(result['ranking'])}"
    return summary, _to_table(rows)


async def do_compare(query: str, top_k: float):
    result = await _post("/vectorstore/compare", {"query": query, "top_k": int(top_k)})
    rows = []
    for row in result["rows"]:
        rows.append(
            {
                "store": row["store"],
                "status": row["status"],
                "time_ms": row["time_ms"],
                "top_result": row["top_result"],
                "results_found": len(row["results"]),
            }
        )
    return _to_table(rows)


async def do_switch_store(store: str):
    result = await _post("/vectorstore/switch", {"store": store})
    return f"Switched to **{result['store']}**  \nStats: {result['stats']}"


async def refresh_health():
    health = await _get("/health")
    stores = await _get("/vectorstores")
    models = await _get("/models")
    embeddings = await _get("/embeddings")

    health_md = (
        f"**Status:** {health['status']}  \n"
        f"**LLM backend:** {health.get('llm_backend', 'ollama')} ({health.get('llm_backend_url', '')})  \n"
        f"**LLM backend connected:** {health['ollama_connected']}  \n"
        f"**Loaded models:** {', '.join(health['loaded_models']) or '(none)'}"
    )
    gating = _gating_from_health(health)  # Benchmark tab
    return (health_md, _to_table(stores), _to_table(models), _to_table(embeddings), *gating)


async def do_seed():
    result = await _post("/seed", {})
    return (
        f"Seeded {result['customers']} customers, {result['offers']} offers, "
        f"{result['purchases']} purchases.  \nIndexed: {result['indexed']}"
    )


async def do_index_embedding(embedding_model: str):
    result = await _post("/seed/embedding", {"embedding_model": embedding_model})
    return f"Indexed **{result['embedding_model']}** into all stores.  \nIndexed: {result['indexed']}"


async def _poll_job(job_id: str, status_path: str = "/upload/status"):
    """Polls GET {status_path}/{job_id} until the job finishes, yielding a
    (status_md, job_id, retry_btn_update, failed_step, *gating_updates) tuple
    each time -- Gradio streams each yield to the browser live. Shared by
    the NBO ("Upload & Setup") and NBA ("Upload NBA Data") tabs -- same job
    shape, different endpoint."""
    while True:
        try:
            status = await _get(f"{status_path}/{job_id}")
        except httpx.HTTPError as e:
            yield (f"Could not reach API at {API_BASE}: {e}", job_id, gr.update(visible=True), None, *_NO_GATE_CHANGE)
            return

        md = _render_upload_status(status)
        overall = status["overall_status"]
        failed_step = next((s["name"] for s in status["steps"] if s["status"] == "failed"), None)

        if overall == "done":
            health = await _get("/health")
            gating = _gating_from_health(health)  # Benchmark tab
            yield (md, job_id, gr.update(visible=False), None, *gating)
            return
        if overall == "failed":
            yield (md, job_id, gr.update(visible=True), failed_step, *_NO_GATE_CHANGE)
            return

        yield (md, job_id, gr.update(visible=False), None, *_NO_GATE_CHANGE)
        await asyncio.sleep(UPLOAD_POLL_SECONDS)


async def do_upload_start(customers_file, offers_file, purchases_file, segments_file):
    files = {
        "customers": _read_file_bytes(customers_file),
        "offers": _read_file_bytes(offers_file),
        "purchases": _read_file_bytes(purchases_file),
        "segments": _read_file_bytes(segments_file),
    }
    missing = [k for k, v in files.items() if not v]
    if missing:
        yield (
            f"_Select all 4 files before starting -- missing: {', '.join(f + '.json' for f in missing)}_",
            None, gr.update(visible=False), None, *_NO_GATE_CHANGE,
        )
        return

    try:
        result = await _post_files("/upload", files)
    except httpx.HTTPError as e:
        yield (f"Could not reach API at {API_BASE}: {e}", None, gr.update(visible=False), None, *_NO_GATE_CHANGE)
        return

    async for update in _poll_job(result["job_id"]):
        yield update


async def do_upload_quick():
    """No drag-and-drop: runs the exact same ingestion pipeline against the
    4 files already sitting in data/ (POST /upload/quick), for quickly
    re-validating/re-indexing what's already on this machine."""
    try:
        result = await _post("/upload/quick", {})
    except httpx.HTTPError as e:
        yield (f"Could not reach API at {API_BASE}: {e}", None, gr.update(visible=False), None, *_NO_GATE_CHANGE)
        return

    async for update in _poll_job(result["job_id"]):
        yield update


async def do_upload_retry(job_id, failed_step, customers_file, offers_file, purchases_file, segments_file):
    if not job_id:
        yield ("_No job to retry._", None, gr.update(visible=False), None, *_NO_GATE_CHANGE)
        return

    files = None
    if failed_step == "upload_validate":
        files = {
            "customers": _read_file_bytes(customers_file),
            "offers": _read_file_bytes(offers_file),
            "purchases": _read_file_bytes(purchases_file),
            "segments": _read_file_bytes(segments_file),
        }
        missing = [k for k, v in files.items() if not v]
        if missing:
            yield (
                f"_Retrying the Upload & validate step needs all 4 files reselected -- "
                f"missing: {', '.join(f + '.json' for f in missing)}_",
                job_id, gr.update(visible=True), failed_step, *_NO_GATE_CHANGE,
            )
            return

    try:
        if files:
            await _post_files(f"/upload/retry/{job_id}", files)
        else:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(f"{API_BASE}/upload/retry/{job_id}")
                if resp.status_code >= 400:
                    detail = resp.json().get("detail", resp.text)
                    raise gr.Error(f"/upload/retry failed: {detail}")
    except httpx.HTTPError as e:
        yield (f"Could not reach API at {API_BASE}: {e}", job_id, gr.update(visible=True), failed_step, *_NO_GATE_CHANGE)
        return

    async for update in _poll_job(job_id):
        yield update


async def do_nba_upload_start(churn_f, reactivation_f, appetency_f, fragility_f, rev_decrease_f, activity_f, targeting_f):
    files = {
        "churn": _read_file_bytes(churn_f),
        "reactivation": _read_file_bytes(reactivation_f),
        "appetency": _read_file_bytes(appetency_f),
        "fragility": _read_file_bytes(fragility_f),
        "revenue_decrease": _read_file_bytes(rev_decrease_f),
        "activity_usage": _read_file_bytes(activity_f),
        "targeting": _read_file_bytes(targeting_f),
    }
    missing = [k for k, v in files.items() if not v]
    if missing:
        yield (
            f"_Select all 7 files before starting -- missing: {', '.join(f + '.json' for f in missing)}_",
            None, gr.update(visible=False), None, *_NO_GATE_CHANGE,
        )
        return

    try:
        result = await _post_files("/nba/upload", files)
    except httpx.HTTPError as e:
        yield (f"Could not reach API at {API_BASE}: {e}", None, gr.update(visible=False), None, *_NO_GATE_CHANGE)
        return

    async for update in _poll_job(result["job_id"], status_path="/nba/upload/status"):
        yield update


async def do_nba_upload_retry(job_id, failed_step, churn_f, reactivation_f, appetency_f, fragility_f, rev_decrease_f, activity_f, targeting_f):
    if not job_id:
        yield ("_No job to retry._", None, gr.update(visible=False), None, *_NO_GATE_CHANGE)
        return

    files = None
    if failed_step == "upload_validate":
        files = {
            "churn": _read_file_bytes(churn_f),
            "reactivation": _read_file_bytes(reactivation_f),
            "appetency": _read_file_bytes(appetency_f),
            "fragility": _read_file_bytes(fragility_f),
            "revenue_decrease": _read_file_bytes(rev_decrease_f),
            "activity_usage": _read_file_bytes(activity_f),
            "targeting": _read_file_bytes(targeting_f),
        }
        missing = [k for k, v in files.items() if not v]
        if missing:
            yield (
                f"_Retrying the Upload & validate step needs all 7 files reselected -- "
                f"missing: {', '.join(f + '.json' for f in missing)}_",
                job_id, gr.update(visible=True), failed_step, *_NO_GATE_CHANGE,
            )
            return

    try:
        if files:
            await _post_files(f"/nba/upload/retry/{job_id}", files)
        else:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(f"{API_BASE}/nba/upload/retry/{job_id}")
                if resp.status_code >= 400:
                    detail = resp.json().get("detail", resp.text)
                    raise gr.Error(f"/nba/upload/retry failed: {detail}")
    except httpx.HTTPError as e:
        yield (f"Could not reach API at {API_BASE}: {e}", job_id, gr.update(visible=True), failed_step, *_NO_GATE_CHANGE)
        return

    async for update in _poll_job(job_id, status_path="/nba/upload/status"):
        yield update


async def do_nba_upload_quick():
    """No drag-and-drop: runs the exact same NBA ingestion pipeline against
    the 7 files already sitting in data/ (POST /nba/upload/quick)."""
    try:
        result = await _post("/nba/upload/quick", {})
    except httpx.HTTPError as e:
        yield (f"Could not reach API at {API_BASE}: {e}", None, gr.update(visible=False), None, *_NO_GATE_CHANGE)
        return

    async for update in _poll_job(result["job_id"], status_path="/nba/upload/status"):
        yield update


async def do_load_rules():
    try:
        rules = await _get("/rules")
    except httpx.HTTPError as e:
        return f"_Could not reach API at {API_BASE}: {e}_", ""

    nbo_md = "\n\n".join(
        f"**{r['rule_name']}**\n\n{r['description']}" for r in rules["nbo_rules"]
    ) or "_No NBO rules registered._"
    nbo_md += f"\n\n---\n*Affordability band: **{rules['affordability_band_omr']} OMR** above estimated average spend, tightest preference tier.*"

    pol = rules["nba_policy"]
    seg_rows = "\n".join(f"| `{seg}` | {action} |" for seg, action in pol["segment_action_map"].items())
    prop_rows = "\n".join(f"| {action} | {p:.4f} |" for action, p in pol["action_propensity"].items())
    override_rows = "\n".join(f"| {action} | `{need}` |" for action, need in rules["action_need_override"].items())

    nba_md = (
        f"**Bandit policy** — `{pol['vw_args']}`\n\n"
        f"**Excluded segments** (never reach the bandit): {', '.join(f'`{s}`' for s in pol['excluded_segments'])}\n\n"
        f"**Non-push actions** (offers returned as context only, not a sales push): {', '.join(f'`{a}`' for a in pol['non_push_actions'])}\n\n"
        f"**Reward window:** {pol['reward_window_days']} days\n\n"
        f"---\n**Segment → action policy**\n\n| Segment | Action |\n|---|---|\n{seg_rows}\n\n"
        f"---\n**Action propensity (base rate)**\n\n| Action | Propensity |\n|---|---|\n{prop_rows}\n\n"
        f"---\n**Action → NBO need override** (steers which offers NBO looks for)\n\n"
        f"| Bandit/LLM action | NBO need forced to |\n|---|---|\n{override_rows}"
    )
    return nbo_md, nba_md


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
RULES_CARD_CSS = """
.rules-card { border: 1px solid #2a2a2e; border-radius: 14px; padding: 16px 20px; background: #17171a; }
"""


def _agentic_tab(title_html: str, button_label: str, picker_label: str, run_fn, batch_fn, example_list: str, step2_fn=None):
    gr.HTML(title_html)
    with gr.Row(equal_height=True):
        msisdn = gr.Textbox(label="MSISDN", placeholder="e.g. 15", scale=2)
        picker = gr.Dropdown(label=picker_label, choices=[], scale=4, allow_custom_value=False)
        run_btn = gr.Button(button_label, variant="primary", scale=1)
    picker.change(lambda v: gr.update(value=str(v) if v is not None else ""), inputs=[picker], outputs=[msisdn])
    profile = gr.HTML()
    pipeline = gr.HTML()
    with gr.Row():
        with gr.Column(scale=5):
            trace = gr.HTML()
        with gr.Column(scale=7):
            action = gr.HTML()
            offers = gr.HTML()
            stats = gr.HTML()
    why = gr.HTML()
    message = gr.HTML()
    outputs = [profile, pipeline, trace, action, offers, why, message, stats]
    run_btn.click(run_fn, inputs=[msisdn], outputs=outputs)
    msisdn.submit(run_fn, inputs=[msisdn], outputs=outputs)
    if step2_fn is not None:
        step2_btn = gr.Button("Step 2 → Recommend offers for this action", variant="primary")
        step2_btn.click(step2_fn, inputs=[msisdn], outputs=outputs)

    with gr.Accordion("Batch: recommend for a list of MSISDNs", open=False):
        gr.HTML('<div class="nbx-muted">Paste numbers separated by commas, spaces or new lines. Two customers run at a time; '
                "the table fills in as each one finishes, then a CSV is ready to download.</div>")
        with gr.Row(equal_height=True):
            batch_text = gr.Textbox(label="MSISDN list", value=example_list, lines=3, scale=4)
            batch_btn = gr.Button("Recommend for all", variant="primary", scale=1)
        batch_status = gr.Markdown()
        batch_table = gr.HTML()
        batch_file = gr.File(label="Download results (CSV)", interactive=False)
        batch_btn.click(batch_fn, inputs=[batch_text], outputs=[batch_status, batch_table, batch_file])
    return picker


with gr.Blocks(title="NBX · Agentic Decisioning for Ooredoo Oman", theme=ui.THEME, css=ui.CSS + RULES_CARD_CSS, js=ui.FORCE_DARK_JS) as demo:
    header = gr.HTML(ui.header_html())

    with gr.Tabs(selected="home") as tabs:
        with gr.Tab("Home", id="home"):
            landing = gr.HTML(ui.landing_html(None, 0, "…"))
            with gr.Row():
                go_action = gr.Button("Open Agentic Best Action Recommender", variant="primary")
                go_offer = gr.Button("Open Agentic Offer Recommender", variant="primary")
                go_comp = gr.Button("Competitor Intelligence")

        with gr.Tab("Agentic Best Action Recommender", id="action"):
            act_picker = _agentic_tab(
                '<div class="nbx-muted"><b>Step 1.</b> The Best Action agent investigates the customer with tools (risk scores, usage trends, '
                "contextual bandit, contact policy, competitor prices), chooses one action, and a critic reviews it. "
                "Below the decision: why this action, which signals drove it, and why the alternatives were rejected. "
                "Then use <b>Step 2</b> to get the offers for that action.</div>",
                "Recommend best action", "…or pick a customer (sorted by churn risk)", run_best_action, run_batch_action, "15, 7, 1, 31",
                step2_fn=run_offer_from_last_action)

        with gr.Tab("Agentic Offer Recommender", id="offer"):
            off_picker = _agentic_tab(
                '<div class="nbx-muted"><b>Step 1 → Step 2.</b> The Best Action agent first decides the action; the Offer agent then '
                "builds on that decision: it searches the catalog, collaborative filtering and the cohort graph for offers that serve "
                "the action, checks eligibility, simulates expected value and benchmarks each candidate against Omantel and Vodafone Oman. "
                "Below the offers: why each one, with the closest competitor plans.</div>",
                "Recommend offers", "…or pick a customer (sorted by churn risk)", run_offer, run_batch_offer, "7, 15, 22")

        with gr.Tab("Competitor Intelligence", id="competitors"):
            gr.HTML('<div class="nbx-muted">Public plan pages of Omantel and Vodafone Oman, scraped politely '
                    "(robots.txt checked, one request every 1.5 s), normalized to OMR / GB / days, and indexed in their own "
                    "vector collection. The agents use this snapshot when they recommend.</div>")
            with gr.Row():
                comp_operator = gr.Dropdown(["All", "Omantel", "Vodafone Oman"], value="All", label="Operator")
                comp_segment = gr.Dropdown(["All", "mass", "students", "visitors"], value="mass", label="Segment")
                comp_refresh_btn = gr.Button("Re-scrape now (~30 s)")
            comp_summary = gr.Markdown()
            comp_table = gr.HTML()
            gr.Markdown("### Benchmark one of our offers")
            with gr.Row():
                comp_offer = gr.Dropdown(label="Our offer", choices=[], scale=4)
                comp_cmp_btn = gr.Button("Compare", variant="primary", scale=1)
            comp_cmp_html = gr.HTML()
            gr.Markdown("### Semantic search over competitor pages")
            with gr.Row():
                comp_q = gr.Textbox(label="Query", value="unlimited social media pass", scale=4)
                comp_q_btn = gr.Button("Search", scale=1)
            comp_hits = gr.HTML()
            comp_refresh_md = gr.Markdown()

            comp_operator.change(comp_load, inputs=[comp_operator, comp_segment], outputs=[comp_summary, comp_table])
            comp_segment.change(comp_load, inputs=[comp_operator, comp_segment], outputs=[comp_summary, comp_table])
            comp_cmp_btn.click(comp_compare, inputs=[comp_offer], outputs=[comp_cmp_html])
            comp_q_btn.click(comp_search, inputs=[comp_q], outputs=[comp_hits])
            comp_refresh_btn.click(comp_refresh, outputs=[comp_refresh_md]).then(
                comp_load, inputs=[comp_operator, comp_segment], outputs=[comp_summary, comp_table])

        with gr.Tab("Business Rules", id="rules"):
            gr.Markdown("Read live from the running code (`GET /rules`).")
            rules_refresh_btn = gr.Button("Refresh")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Offer rules — `nbx/agents/rules_engine.py`")
                    nbo_rules_md = gr.Markdown(elem_classes=["rules-card"])
                with gr.Column():
                    gr.Markdown("### Action policy — bandit + `nbx/agents/final_agent.py`")
                    nba_rules_md = gr.Markdown(elem_classes=["rules-card"])

        with gr.Tab("Graph Insights", id="graph"):
            gr.Markdown("Cohort purchase graph over customers / offers / segments: what customers like this one actually bought.")
            graph_msisdn = gr.Number(label="MSISDN", value=15, precision=0)
            with gr.Row():
                cohort_btn = gr.Button("Cohort offers", variant="primary")
                similar_btn = gr.Button("Similar customers")
                graph_stats_btn = gr.Button("Graph stats")
            cohort_table = gr.Dataframe(label="Offers bought by this customer's cohort", interactive=False)
            similar_table = gr.Dataframe(label="Customers sharing this segment", interactive=False)
            graph_stats_md = gr.Markdown()
            graph_stats_table = gr.Dataframe(label="Nodes by kind", interactive=False)
            cohort_btn.click(do_cohort_offers, inputs=[graph_msisdn], outputs=[cohort_table])
            similar_btn.click(do_similar_customers, inputs=[graph_msisdn], outputs=[similar_table])
            graph_stats_btn.click(do_graph_stats, outputs=[graph_stats_md, graph_stats_table])

        with gr.Tab("Customer lookup", id="lookup"):
            cust_msisdn = gr.Number(label="MSISDN", value=15, precision=0)
            cust_btn = gr.Button("Look up customer")
            cust_profile = gr.Dataframe(label="Profile", interactive=False)
            cust_history = gr.Dataframe(label="Purchase history", interactive=False)
            cust_btn.click(do_customer_lookup, inputs=[cust_msisdn], outputs=[cust_profile, cust_history])

        with gr.Tab("Benchmark LLMs", id="bench"):
            with gr.Row():
                bench_msisdn = gr.Number(label="MSISDN", value=15, precision=0)
                bench_store = gr.Dropdown(choices=STORE_CHOICES, value=config.DEFAULT_VECTORSTORE, label="Vector store")
            bench_btn = gr.Button("Run benchmark across installed models", variant="primary")
            bench_gate_note = gr.Markdown(visible=False)
            bench_summary = gr.Markdown()
            bench_table = gr.Dataframe(label="Results", interactive=False)
            bench_btn.click(do_benchmark, inputs=[bench_msisdn, bench_store], outputs=[bench_summary, bench_table])

        with gr.Tab("Vector store compare", id="vectors"):
            with gr.Row():
                cmp_query = gr.Textbox(label="Query", value="international roaming offers")
                cmp_topk = gr.Number(label="Top K", value=5, precision=0)
            cmp_btn = gr.Button("Compare all 5 stores", variant="primary")
            cmp_table = gr.Dataframe(label="Comparison", interactive=False)
            cmp_btn.click(do_compare, inputs=[cmp_query, cmp_topk], outputs=[cmp_table])
            gr.Markdown("---")
            switch_store = gr.Dropdown(choices=STORE_CHOICES, value=config.DEFAULT_VECTORSTORE, label="Set active store")
            switch_btn = gr.Button("Switch active vector store")
            switch_result = gr.Markdown()
            switch_btn.click(do_switch_store, inputs=[switch_store], outputs=[switch_result])

        with gr.Tab("Upload & Setup", id="upload"):
            gr.Markdown(
                "Upload your own `customers.json` / `offers.json` / `purchases.json` / `segments.json` "
                "and run the ingestion pipeline as a visible sequence of steps. This replaces the seeded data."
            )
            with gr.Row():
                up_customers = gr.File(label="customers.json", file_types=[".json"])
                up_offers = gr.File(label="offers.json", file_types=[".json"])
                up_purchases = gr.File(label="purchases.json", file_types=[".json"])
                up_segments = gr.File(label="segments.json", file_types=[".json"])
            with gr.Row():
                up_start_btn = gr.Button("Start ingestion", variant="primary")
                up_quick_btn = gr.Button("⚡ Quick load from data/ (no drag-and-drop)")
            up_status_md = gr.Markdown("_No upload started yet._")
            up_retry_btn = gr.Button("Retry from failed step", variant="stop", visible=False)
            up_job_state = gr.State(value=None)
            up_failed_step_state = gr.State(value=None)

        with gr.Tab("Upload NBA Data", id="upload-nba"):
            gr.Markdown(
                "Upload the 7 NBA source files (see `core.config.NBA_SOURCE_FILE_ATTRS`) and retrain the contextual "
                "bandit as a visible sequence of steps. This replaces the NBA data on disk."
            )
            with gr.Row():
                nba_up_churn = gr.File(label="churn.json", file_types=[".json"])
                nba_up_reactivation = gr.File(label="reactivation.json", file_types=[".json"])
                nba_up_appetency = gr.File(label="appetency.json", file_types=[".json"])
                nba_up_fragility = gr.File(label="fragility.json", file_types=[".json"])
            with gr.Row():
                nba_up_rev_decrease = gr.File(label="revenue_decrease.json", file_types=[".json"])
                nba_up_activity = gr.File(label="activity_usage.json", file_types=[".json"])
                nba_up_targeting = gr.File(label="targeting.json", file_types=[".json"])
            with gr.Row():
                nba_up_start_btn = gr.Button("Start ingestion", variant="primary")
                nba_up_quick_btn = gr.Button("⚡ Quick load from data/ (no drag-and-drop)")
            nba_up_status_md = gr.Markdown("_No upload started yet._")
            nba_up_retry_btn = gr.Button("Retry from failed step", variant="stop", visible=False)
            nba_up_job_state = gr.State(value=None)
            nba_up_failed_step_state = gr.State(value=None)

        with gr.Tab("System status", id="status"):
            health_btn = gr.Button("Refresh status", variant="primary")
            health_md = gr.Markdown()
            stores_table = gr.Dataframe(label="Vector stores", interactive=False)
            models_table = gr.Dataframe(label="LLMs", interactive=False)
            embeddings_table = gr.Dataframe(label="Embedding models", interactive=False)
            health_gate_outputs = [bench_btn, bench_gate_note]
            health_btn.click(refresh_health, outputs=[health_md, stores_table, models_table, embeddings_table, *health_gate_outputs])
            gr.Markdown("---")
            seed_btn = gr.Button("Re-seed data + re-index all stores")
            seed_result = gr.Markdown()
            seed_btn.click(do_seed, outputs=[seed_result])
            gr.Markdown("---\nIndex an additional embedding model without touching the others:")
            with gr.Row():
                index_embed_choice = gr.Dropdown(choices=EMBEDDING_CHOICES, value=config.DEFAULT_EMBEDDING_MODEL, label="Embedding model")
                index_embed_btn = gr.Button("Index this embedding model")
            index_embed_result = gr.Markdown()
            index_embed_btn.click(do_index_embedding, inputs=[index_embed_choice], outputs=[index_embed_result])

    go_action.click(lambda: gr.Tabs(selected="action"), outputs=tabs)
    go_offer.click(lambda: gr.Tabs(selected="offer"), outputs=tabs)
    go_comp.click(lambda: gr.Tabs(selected="competitors"), outputs=tabs)

    demo.load(load_landing, outputs=[landing, header])
    demo.load(load_customer_choices, outputs=[act_picker, off_picker])
    demo.load(comp_load, inputs=[comp_operator, comp_segment], outputs=[comp_summary, comp_table])
    demo.load(comp_our_offers, outputs=[comp_offer])
    demo.load(refresh_health, outputs=[health_md, stores_table, models_table, embeddings_table, *health_gate_outputs])
    demo.load(do_load_rules, outputs=[nbo_rules_md, nba_rules_md])

    up_upload_outputs = [up_status_md, up_job_state, up_retry_btn, up_failed_step_state, bench_btn, bench_gate_note]
    up_start_btn.click(do_upload_start, inputs=[up_customers, up_offers, up_purchases, up_segments], outputs=up_upload_outputs)
    up_quick_btn.click(do_upload_quick, outputs=up_upload_outputs)
    up_retry_btn.click(
        do_upload_retry,
        inputs=[up_job_state, up_failed_step_state, up_customers, up_offers, up_purchases, up_segments],
        outputs=up_upload_outputs,
    )

    nba_up_upload_outputs = [nba_up_status_md, nba_up_job_state, nba_up_retry_btn, nba_up_failed_step_state, bench_btn, bench_gate_note]
    nba_up_files = [nba_up_churn, nba_up_reactivation, nba_up_appetency, nba_up_fragility,
                    nba_up_rev_decrease, nba_up_activity, nba_up_targeting]
    nba_up_start_btn.click(do_nba_upload_start, inputs=nba_up_files, outputs=nba_up_upload_outputs)
    nba_up_quick_btn.click(do_nba_upload_quick, outputs=nba_up_upload_outputs)
    nba_up_retry_btn.click(do_nba_upload_retry, inputs=[nba_up_job_state, nba_up_failed_step_state, *nba_up_files],
                           outputs=nba_up_upload_outputs)

    rules_refresh_btn.click(do_load_rules, outputs=[nbo_rules_md, nba_rules_md])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=config.GRADIO_PORT)
