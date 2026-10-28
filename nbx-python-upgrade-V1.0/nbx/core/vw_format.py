"""Vowpal Wabbit contextual-bandit (cb_adf) text format and learning loops, written once -- used by
nbx/fsdr/bandit.py.

Fixes the bugs of the original production code (docs/fsdr_reference/README_NBA_pipelines.md):
  - numeric features are written `name:value` (a real number), never
    `name=value` (VW reads that as a category);
  - action names become tokens without spaces (CROSS_SELL_DATA);
  - the action list is passed in fixed order and never shuffled: the
    probability vector VW returns is in that order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import vowpalwabbit


def action_token(action: str) -> str:
    return action.replace(" ", "_")


def action_lines(actions: list[str]) -> list[str]:
    return [f"|Action product={action_token(a)}" for a in actions]


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def shared_lines(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> list[str]:
    """One 'shared |User ...' line per row. Zero-valued numeric features are omitted (VW treats absent as 0)."""
    s = pd.Series("shared |User", index=df.index)
    for c in categorical:
        val = df[c].astype(str).str.replace(r"\s+", "_", regex=True) if c in df.columns else pd.Series("NA", index=df.index)
        s = s + f" {c.lower()}=" + val
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float) if c in df.columns else pd.Series(0.0, index=df.index)
        tok = " " + c.lower() + ":" + v.map(_fmt)
        s = s + tok.where(v != 0, "")
    return s.tolist()


def make_example(shared: str, actions: list[str], label: tuple[int, float, float] | None = None) -> list[str]:
    """label = (action_index, cost, probability) for a training example, None for prediction."""
    lines = [shared]
    for i, al in enumerate(action_lines(actions)):
        if label is not None and i == label[0]:
            lines.append(f"0:{label[1]}:{label[2]:.6f} {al}")
        else:
            lines.append(al)
    return lines


def predict_pmf(vw: vowpalwabbit.Workspace, shared: str, actions: list[str]) -> np.ndarray:
    ex = vw.parse(make_example(shared, actions), vowpalwabbit.LabelType.CONTEXTUAL_BANDIT)
    pmf = np.asarray(vw.predict(ex), dtype=float)
    vw.finish_example(ex)
    return pmf / pmf.sum()


def learn_one(vw: vowpalwabbit.Workspace, shared: str, actions: list[str], action_idx: int, cost: float, prob: float) -> None:
    ex = vw.parse(make_example(shared, actions, (action_idx, cost, prob)), vowpalwabbit.LabelType.CONTEXTUAL_BANDIT)
    vw.learn(ex)
    vw.finish_example(ex)


def sample_action(pmf, rng: np.random.Generator) -> tuple[int, float]:
    """One action index drawn from the PMF (production sample_custom_pmf) and its probability, which is the
    propensity of the next learn()."""
    p = np.asarray(pmf, dtype=float)
    p = p / p.sum()
    i = int(rng.choice(len(p), p=p))
    return i, float(p[i])


def learn_logged(vw: vowpalwabbit.Workspace, shared: list[str], actions: list[str], action_idx, cost, prob,
                 passes: int, rng: np.random.Generator) -> None:
    """Logged mode: every row with its real action, cost and (logged) propensity; rows permuted each pass."""
    for _ in range(passes):
        for i in rng.permutation(len(shared)):
            learn_one(vw, shared[i], actions, int(action_idx[i]), float(cost[i]), float(prob[i]))


def learn_simulation(vw: vowpalwabbit.Workspace, shared: list[str], actions: list[str], true_idx, iterations: int,
                     rng: np.random.Generator) -> dict:
    """Production's original training: draw a random taker, sample an action from the current PMF, cost -1 if
    it is the taker's logged action else 0, learn with the PMF probability of the sampled action."""
    counts = np.zeros(len(actions), dtype=int)
    cum = 0.0
    n = iterations if len(shared) else 0
    for _ in range(n):
        j = int(rng.integers(len(shared)))
        a, p = sample_action(predict_pmf(vw, shared[j], actions), rng)
        c = -1.0 if a == int(true_idx[j]) else 0.0
        learn_one(vw, shared[j], actions, a, c, p)
        counts[a] += 1
        cum += c
    return {"iterations": iterations, "rows": len(shared), "match_rate": round(-cum / n, 4) if n else 0.0,
            "action_counts": {a: int(k) for a, k in zip(actions, counts)}}
