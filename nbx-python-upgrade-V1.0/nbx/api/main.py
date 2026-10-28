"""FSDR API: health, the FSDR recommender (/fsdr/*, nbx/api/fsdr.py) and competitor intelligence (/competitors/*).

Security (review finding #5): CORS is limited to FSDR_CORS_ORIGINS (default: the local UI), and with
FSDR_API_KEY set every /fsdr/* and /competitors/* route needs the `X-API-Key` header.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nbx.api import fsdr as fsdr_api
from nbx.api.security import require_api_key
from nbx.core import competitors, config
from nbx.llms import llm_client

app = FastAPI(title="FSDR Recommender API", version="1.0.0")
app.include_router(fsdr_api.router)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key", "X-Admin-Key"])


@app.middleware("http")
async def timing(request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(int((time.time() - start) * 1000))
    return response


@app.get("/health")
async def health():
    """Open (no key): backend, model, and whether the API is protected."""
    ok = await llm_client.is_available()
    return {"status": "ok" if ok else "degraded", "llm_backend": llm_client.backend_name(),
            "llm_backend_url": llm_client.backend_url(), "loaded_models": await llm_client.list_installed_models() if ok else [],
            "api_key_required": bool(config.API_KEY), "operator": config.OPERATOR_NAME}


@app.get("/competitors/offers", dependencies=[Depends(require_api_key)])
async def competitor_offers(operator: str | None = None, segment: str | None = None):
    rows = competitors.load_offers()
    if operator:
        rows = [r for r in rows if r["operator"].lower() == operator.lower()]
    if segment:
        rows = [r for r in rows if r.get("segment") == segment]
    return rows


@app.get("/competitors/search", dependencies=[Depends(require_api_key)])
async def competitor_search(q: str, top_k: int = 5, operator: str | None = None):
    return competitors.search(q, top_k=top_k, operator=operator)


@app.get("/competitors/compare/{btl_name}", dependencies=[Depends(require_api_key)])
async def competitor_compare(btl_name: str, run: str | None = None):
    """One FSDR catalogue offer against the closest plan of each competitor (latest run's catalogue by default)."""
    from nbx.fsdr.agents.tools import _as_nbx_offer
    from nbx.fsdr.datasource import JsonSource
    from nbx.fsdr.offer_rules import load_catalog

    runs = sorted(p for p in fsdr_api.config.FSDR_DATA_DIR.glob("20??-??-??") if p.is_dir())
    folder = (fsdr_api.config.FSDR_DATA_DIR / run) if run else (runs[-1] if runs else None)
    if folder is None or not folder.is_dir():
        raise HTTPException(status_code=404, detail="no FSDR run folder: generate or upload data first")
    cat = load_catalog(JsonSource(folder)).set_index("BTL_NAME", drop=False)
    if btl_name not in cat.index:
        raise HTTPException(status_code=404, detail=f"unknown offer {btl_name}")
    return competitors.compare_offer(_as_nbx_offer(cat.loc[btl_name]), per_operator=2)


@app.post("/competitors/refresh", dependencies=[Depends(require_api_key)])
async def competitor_refresh():
    """Re-scrapes the pages of the local sources.json (one request every 1.5 s) and saves the snapshot."""
    return await asyncio.to_thread(competitors.refresh)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("nbx.api.main:app", host="127.0.0.1", port=config.FASTAPI_PORT, reload=True)
