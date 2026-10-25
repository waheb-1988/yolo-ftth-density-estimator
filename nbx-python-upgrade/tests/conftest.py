"""Shared test setup.

Every test starts on the offline LLM backend, whatever the developer's .env
says, so the suite needs Python libraries only. Tests that simulate a live
LLM server switch to "ollama" themselves and monkeypatch the client, so the
Ollama code path stays enabled in tests (OLLAMA_ENABLED is off in the app).
"""
from __future__ import annotations

import pytest

from nbx.core import config


@pytest.fixture(autouse=True)
def offline_backend(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "offline")
    monkeypatch.setattr(config, "OLLAMA_ENABLED", True)
