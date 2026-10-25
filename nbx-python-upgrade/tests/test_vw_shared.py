"""A3: one Vowpal Wabbit layer -- the NBX demo bandit and the FSDR bandit use the same format and loops."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nbx.core import config, nba_bandit, vw_format


def test_both_bandits_train_through_the_shared_loops(monkeypatch):
    calls = []
    real = vw_format.learn_simulation

    def spy(*args, **kw):
        calls.append(len(args[1]))
        return real(*args, **kw)

    monkeypatch.setattr(vw_format, "learn_simulation", spy)
    nba_bandit.get_bandit(force_rebuild=True)                       # NBX demo bandit
    from nbx.fsdr import bandit
    from nbx.fsdr import config as fcfg
    feats = ["PROBA_CHURN"]
    df = pd.DataFrame({"PROBA_CHURN": [0.1, 0.9] * 10, "NATIONALITY_GROUP": "Oman", "ACTION_IDX": [0, 2] * 10,
                       "COST": -1.0, "PROPENSITY": 0.5, "REWARD": 1})
    bandit.train(df, feats, mode="simulation", sim_iterations=50)  # FSDR bandit
    assert len(calls) == 2
    assert not hasattr(nba_bandit, "_sample_pmf") and not hasattr(nba_bandit, "_build_example")


def test_demo_bandit_writes_numbers_as_name_value_pairs():
    ctx = {f: 0.25 for f in config.NBA_CONTEXT_FEATURES}
    ctx["nationality_group"] = "Oman"
    line = nba_bandit._shared_line(ctx)
    assert line.startswith("shared |User nationality_group=Oman ") and "proba_churn:0.25" in line
    assert "proba_churn=" not in line


def test_sample_action_returns_the_drawn_probability():
    i, p = vw_format.sample_action([0.1, 0.0, 0.9], np.random.default_rng(1))
    assert i in (0, 2) and p == (0.1 if i == 0 else 0.9)
