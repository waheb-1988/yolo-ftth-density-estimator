"""Data minimisation for LLM prompts (V1.0, review #8).

The agents never need the MSISDN itself: prompts and tool outputs carry a stable pseudonym, a salted HMAC of the
number, so the same customer keeps the same reference across steps and runs, and the number cannot be read back
from a prompt or an LLM server log. Set FSDR_PII_SALT in .env (any long random string) and keep it private.
"""
from __future__ import annotations

import hashlib
import hmac
import os

_DEFAULT_SALT = "fsdr-dev-salt-change-me"


def pseudonym(value: str | int | None) -> str | None:
    if value is None or value == "":
        return None
    salt = os.getenv("FSDR_PII_SALT", _DEFAULT_SALT).encode()
    return "C-" + hmac.new(salt, str(value).encode(), hashlib.sha256).hexdigest()[:12]
