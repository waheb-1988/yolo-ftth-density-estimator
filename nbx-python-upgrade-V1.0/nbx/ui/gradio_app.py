"""FSDR web UI (Gradio): Home, the step-by-step Journey (nbx/ui/journey_tab.py), the 5 FSDR tabs (nbx/ui/fsdr_tabs.py)
and Competitors.

Every button calls the FastAPI backend over HTTP (no business logic here).
"""
from __future__ import annotations

import gradio as gr
import httpx

from nbx.core import config
from nbx.ui import fsdr_tabs, journey_tab
from nbx.ui import theme as ui

API_BASE = fsdr_tabs.API_BASE


async def _health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            return (await c.get(f"{API_BASE}/health")).json()
    except Exception:  # noqa: BLE001  (API not up yet)
        return {"status": "API not reachable", "llm_backend": "?", "loaded_models": []}


async def load_header():
    h = await _health()
    llm = h["llm_backend"] + (f" · {h['loaded_models'][0]}" if h.get("loaded_models") else "")
    return ui.header_html(f"API {h['status']} · LLM {llm}"), ui.landing_html(llm)


async def comp_load(operator: str, segment: str):
    params = {k: v for k, v in (("operator", operator), ("segment", segment)) if v and v != "All"}
    rows = await fsdr_tabs._get("/competitors/offers", **params)
    cols = ["operator", "plan", "category", "segment", "price_omr", "validity_days", "data_gb", "monthly_equiv_omr", "price_per_gb_omr"]
    summary = f"**{len(rows)} competitor plans** in the snapshot" + (f" (scraped {rows[0].get('scraped_at', '')[:10]})" if rows else "")
    return summary, fsdr_tabs._table(rows, cols)


async def comp_search(q: str, operator: str):
    if not q.strip():
        raise gr.Error("Type a few words, e.g. 'unlimited data weekly'.")
    rows = await fsdr_tabs._get("/competitors/search", q=q, top_k=8, **({"operator": operator} if operator and operator != "All" else {}))
    return fsdr_tabs._table([{**r, "text": r["text"][:300]} for r in rows], ["operator", "kind", "score", "text"])


async def comp_compare(btl: str):
    if not btl.strip():
        raise gr.Error("Type an FSDR offer code, e.g. BTL_DATA_L2_30D_14.4GB.")
    r = await fsdr_tabs._get(f"/competitors/compare/{btl.strip()}")
    head = fsdr_tabs._kv({"our monthly equivalent (OMR)": r["our_monthly_equiv_omr"], "our price per GB": r["our_price_per_gb_omr"],
                          "position": r["position"]})
    rows = fsdr_tabs._table(r["closest"], ["operator", "plan", "price_omr", "validity_days", "data_gb", "monthly_equiv_omr",
                                           "monthly_delta_omr", "verdict"])
    return fsdr_tabs._section("Comparison", head) + fsdr_tabs._section("Closest competitor plans", rows)


with gr.Blocks(title=f"FSDR Recommender · {config.OPERATOR_NAME}", theme=ui.THEME, css=ui.CSS, js=ui.FORCE_DARK_JS) as demo:
    header = gr.HTML(ui.header_html())
    with gr.Tabs(selected="home"):
        with gr.Tab("Home", id="home"):
            landing = gr.HTML(ui.landing_html("…"))
        journey_ui = journey_tab.build()
        fsdr_ui = fsdr_tabs.build()
        with gr.Tab("Competitors", id="competitors"):
            gr.HTML('<div class="nbx-muted">Anonymised snapshot of the two competitors\' public plans (COMPETITOR_A / '
                    "COMPETITOR_B). The competitor agent compares every recommended offer with the closest plan of the same "
                    "validity band.</div>")
            with gr.Row(equal_height=True):
                comp_operator = gr.Dropdown(["All", *config.COMPETITOR_LABELS], value="All", label="Operator", scale=2)
                comp_segment = gr.Dropdown(["All", "mass", "visitors", "students"], value="All", label="Segment", scale=2)
                comp_btn = gr.Button("Show plans", variant="primary", scale=1)
            comp_summary = gr.Markdown()
            comp_table = gr.HTML()
            with gr.Row(equal_height=True):
                comp_q = gr.Textbox(label="Search the snapshot", placeholder="unlimited data weekly", scale=4)
                comp_q_btn = gr.Button("Search", scale=1)
            comp_q_out = gr.HTML()
            with gr.Row(equal_height=True):
                comp_offer = gr.Textbox(label="Compare one FSDR offer", placeholder="BTL_DATA_L2_30D_14.4GB", scale=4)
                comp_offer_btn = gr.Button("Compare", scale=1)
            comp_offer_out = gr.HTML()

    comp_btn.click(comp_load, inputs=[comp_operator, comp_segment], outputs=[comp_summary, comp_table])
    comp_q_btn.click(comp_search, inputs=[comp_q, comp_operator], outputs=[comp_q_out])
    comp_offer_btn.click(comp_compare, inputs=[comp_offer], outputs=[comp_offer_out])
    demo.load(load_header, outputs=[header, landing])
    demo.load(fsdr_tabs.load_runs, outputs=fsdr_ui["load_outputs"])
    demo.load(journey_ui["load"], outputs=journey_ui["load_outputs"])


if __name__ == "__main__":
    demo.queue().launch(server_name=config.UI_HOST, server_port=config.GRADIO_PORT)
