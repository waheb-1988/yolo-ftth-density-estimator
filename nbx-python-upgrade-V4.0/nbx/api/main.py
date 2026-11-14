"""FSDR API: health and the FSDR recommender (/fsdr/*, nbx/api/fsdr.py).

Security (review finding #5): CORS is limited to FSDR_CORS_ORIGINS (default: the local UI), and with
FSDR_API_KEY set every /fsdr/* route needs the `X-API-Key` header.
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from nbx.api import fsdr as fsdr_api
from nbx.core import config
from nbx.llms import llm_client

# V4.0 local only: the Swagger UI loads its JS / CSS from a CDN; the default /docs is a local page instead
# (nbx/api/local_docs.py). FSDR_SWAGGER_CDN=true restores Swagger UI.
SWAGGER_CDN = os.getenv("FSDR_SWAGGER_CDN", "false").lower() == "true" or not config.LOCAL_ONLY
app = FastAPI(title="FSDR Recommender API", version="4.0.0",
              docs_url="/docs" if SWAGGER_CDN else None, redoc_url="/redoc" if SWAGGER_CDN else None)
if not SWAGGER_CDN:
    from nbx.api.local_docs import PAGE as _DOCS_PAGE

    @app.get("/docs", include_in_schema=False)
    async def local_docs():
        return HTMLResponse(_DOCS_PAGE)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("nbx.api.main:app", host="127.0.0.1", port=config.FASTAPI_PORT, reload=True)
