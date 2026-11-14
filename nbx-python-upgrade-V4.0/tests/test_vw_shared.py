"""A3: the FSDR bandit goes through the shared Vowpal Wabbit layer (nbx/core/vw_format.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nbx.core import vw_format


def test_fsdr_bandit_trains_through_the_shared_loops(monkeypatch):
    calls = []
    real = vw_format.learn_simulation

    def spy(*args, **kw):
        calls.append(len(args[1]))
        return real(*args, **kw)

    monkeypatch.setattr(vw_format, "learn_simulation", spy)
    from nbx.fsdr import bandit
    feats = ["PROBA_CHURN"]
    df = pd.DataFrame({"PROBA_CHURN": [0.1, 0.9] * 10, "NATIONALITY_GROUP": "Oman", "ACTION_IDX": [0, 2] * 10,
                       "COST": -1.0, "PROPENSITY": 0.5, "REWARD": 1})
    bandit.train(df, feats, mode="simulation", sim_iterations=50)
    assert len(calls) == 1
    assert not hasattr(bandit, "_sample_pmf")


def test_sample_action_returns_the_drawn_probability():
    i, p = vw_format.sample_action([0.1, 0.0, 0.9], np.random.default_rng(1))
    assert i in (0, 2) and p == (0.1 if i == 0 else 0.9)
