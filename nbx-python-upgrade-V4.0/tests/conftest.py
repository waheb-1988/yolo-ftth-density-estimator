"""Shared test setup.

Every test starts on the offline LLM backend, whatever the developer's .env
says, so the suite needs Python libraries only. Tests that simulate a live
vLLM server switch backend themselves and monkeypatch the client.
"""
from __future__ import annotations

import os

# The suite tests the full engine: both targets at 25k each (the V4.0 deployment default is Target 2 only at 50k + 50k,
# tested explicitly in tests/fsdr/test_v4_split_targets.py). Set before any nbx.fsdr import reads them.
os.environ.setdefault("FSDR_TARGETS", "target1,target2")
os.environ.setdefault("FSDR_T2_CONSTRAINT_SIZE", "25000")
os.environ.setdefault("FSDR_T2_FREE_SIZE", "25000")
os.environ.setdefault("FSDR_CONTROL_SIZE", "20000")

import pytest  # noqa: E402

from nbx.core import config  # noqa: E402


@pytest.fixture(autouse=True)
def offline_backend(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "offline")
