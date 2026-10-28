"""API security (review finding #5): an API key on every data / decision route, an admin key for actions that
bypass a code rule (forced model promotion). Keys come from .env (FSDR_API_KEY, FSDR_ADMIN_KEY) and are
compared in constant time."""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from nbx.core import config


def _matches(given: str | None, expected: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode(), expected.encode())


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """With FSDR_API_KEY set, the header `X-API-Key` must match it (401 otherwise). Unset = open (local dev)."""
    if config.API_KEY and not _matches(x_api_key, config.API_KEY):
        raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")


def require_admin(x_admin_key: str | None) -> None:
    """Admin-only actions (forced promotion of a model the IPS rule refused)."""
    if not config.ADMIN_KEY:
        raise HTTPException(status_code=403, detail="forced promotion is disabled: set FSDR_ADMIN_KEY to allow it")
    if not _matches(x_admin_key, config.ADMIN_KEY):
        raise HTTPException(status_code=403, detail="missing or wrong X-Admin-Key")
