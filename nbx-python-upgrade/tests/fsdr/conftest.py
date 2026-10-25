"""Shared FSDR fixtures: one small generated run (in memory + written as JSON), reused by every test."""
from __future__ import annotations

from datetime import date

import pytest

from nbx.fsdr.datasource import SyntheticSource

RUN_DATE = date(2026, 10, 21)          # a Wednesday: T = 2026-10-15
N = 400


@pytest.fixture(scope="session")
def small_source() -> SyntheticSource:
    return SyntheticSource(n=N, run_date=RUN_DATE, scenarios="baseline", seed=11)


@pytest.fixture(scope="session")
def small_folder(small_source, tmp_path_factory):
    return small_source.write(tmp_path_factory.mktemp("fsdr") / RUN_DATE.isoformat())


@pytest.fixture(scope="session")
def scored_run(tmp_path_factory):
    """A 1,200-line run folder that has been trained and scored (model registry in a temp folder)."""
    from nbx.fsdr.pipelines.score import run_scoring
    from nbx.fsdr.pipelines.train import run_training

    base = tmp_path_factory.mktemp("fsdr scored")          # a space in the path on purpose (VW model loading)
    folder = SyntheticSource(n=1200, run_date=RUN_DATE, scenarios="baseline", seed=13).write(base / RUN_DATE.isoformat())
    root = base / "models"
    train = run_training(folder, root=root)
    score = run_scoring(folder, root=root)
    return {"folder": folder, "root": root, "train": train, "score": score}


@pytest.fixture(scope="session")
def workspace(scored_run):
    from nbx.fsdr.agents.tools import FsdrWorkspace

    return FsdrWorkspace(scored_run["folder"])
