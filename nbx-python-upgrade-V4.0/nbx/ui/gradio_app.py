"""FSDR web UI (Gradio): Home, the step-by-step Journey (nbx/ui/journey_tab.py), the 5 FSDR tabs (nbx/ui/fsdr_tabs.py)
and the training / scoring tab of the two targets (nbx/ui/groups_tab.py).

Every button calls the FastAPI backend over HTTP (no business logic here).
"""
from __future__ import annotations

import re

import gradio as gr
import httpx

from nbx.core import config
from nbx.ui import fsdr_tabs, groups_tab, journey_tab
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


with gr.Blocks(title=f"FSDR Recommender · {config.OPERATOR_NAME}", theme=ui.THEME, css=ui.CSS, js=ui.FORCE_DARK_JS) as demo:
    header = gr.HTML(ui.header_html())
    with gr.Tabs(selected="home"):
        with gr.Tab("Home", id="home"):
            landing = gr.HTML(ui.landing_html("…"))
        journey_ui = journey_tab.build()
        fsdr_ui = fsdr_tabs.build()
        groups_tab.build()

    demo.load(load_header, outputs=[header, landing])
    demo.load(fsdr_tabs.load_runs, outputs=fsdr_ui["load_outputs"])
    demo.load(journey_ui["load"], outputs=journey_ui["load_outputs"])


# V4.0 local only: Gradio's page template asks the browser to preconnect to Google Fonts and to load
# iframe-resizer from cdnjs (only needed when the UI is embedded in an iframe). Both are stripped, so the
# browser loads nothing from outside this machine (the fonts are Gradio's bundled LocalFont / system fonts).
_EXTERNAL_TAGS = re.compile(rb"<link[^>]*?fonts\.(?:googleapis|gstatic)\.com[^>]*?>"
                            rb"|<script[^>]*?cdnjs\.cloudflare\.com[^>]*?>\s*</script>", re.S)


def strip_external(html: bytes) -> bytes:
    return _EXTERNAL_TAGS.sub(b"", html)


def local_app():
    """The UI served by our own FastAPI app, with the external tags removed from every HTML page."""
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def _no_external(request: Request, call_next):
        resp = await call_next(request)
        if not resp.headers.get("content-type", "").startswith("text/html"):
            return resp
        body = strip_external(b"".join([chunk async for chunk in resp.body_iterator]))
        headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}
        return Response(body, status_code=resp.status_code, headers=headers)

    return gr.mount_gradio_app(app, demo.queue(), path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(local_app(), host=config.UI_HOST, port=config.GRADIO_PORT, log_level="warning")
    print(f"Stopped. (UI was on http://{config.UI_HOST}:{config.GRADIO_PORT})")
