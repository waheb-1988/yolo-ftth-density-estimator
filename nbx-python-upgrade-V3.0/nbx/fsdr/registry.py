"""Model registry: one versioned folder per trained bandit, a champion, rollback.

    data/fsdr/state/models/
    ├── registry.json            {"champion": "v0003", "history": [...], "models": {version: summary}}
    ├── v0001/nba_cb_adf.vw + nba_meta.json
    └── v0002/...

Promotion is code-enforced (the training agent can decide "keep" or "ask human", never force a worse model in):
  paired_ci (V1.0 default, review #3)    challenger and champion are evaluated on the same holdout (paired bootstrap
                                         of the SNIPS gain, deployed sampling policy): refused only when the 95 % upper
                                         bound of the gain is < -PROMOTION_MARGIN (significantly worse); significantly
                                         better or inconclusive -> promoted (the fresher model)
  point (V0)                             holdout IPS >= the champion's minus PROMOTION_TOLERANCE
The first model is promoted (no champion yet). Weeks whose labels are not usable for off-policy estimates
(legacy targeting, deterministic propensity) fall back to the point rule and say so.

Writes go through a lock file (review #16), so two trainings never interleave their registry updates.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import vowpalwabbit

from nbx.fsdr import bandit, config


def _root(root: Path | None) -> Path:
    return Path(root) if root else config.FSDR_STATE_DIR / "models"


def _index(root: Path) -> dict:
    p = root / "registry.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"champion": None, "history": [], "models": {}}


def _write(root: Path, idx: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "registry.json.tmp"
    tmp.write_text(json.dumps(idx, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, root / "registry.json")                  # atomic: a reader never sees half a file


LOCK_TIMEOUT_S = 60.0
LOCK_STALE_S = 600.0


@contextmanager
def _locked(root: Path):
    """Exclusive lock on the registry (O_EXCL lock file). A lock older than LOCK_STALE_S is a crashed writer's."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "registry.lock"
    deadline = time.time() + LOCK_TIMEOUT_S
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                raise TimeoutError(f"model registry {root} is locked by another process ({lock})")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _summary(meta: dict) -> dict:
    keys = ("trained_at", "run_date", "anchor_date", "train_mode", "reward_definition", "epsilon", "n_train",
            "holdout_take_rate", "holdout_ips_take_rate", "holdout_snips_take_rate", "holdout_snips_ci",
            "holdout_ips_greedy_take_rate", "ope_valid", "vs_champion", "data_dir")
    return {k: meta.get(k) for k in keys}


def register(vw: vowpalwabbit.Workspace, meta: dict, root: Path | None = None) -> str:
    root = _root(root)
    with _locked(root):
        idx = _index(root)
        version = f"v{len(idx['models']) + 1:04d}"
        meta = {**meta, "version": version}
        bandit.save(vw, meta, root / version)
        idx["models"][version] = _summary(meta)
        _write(root, idx)
    return version


def champion(root: Path | None = None) -> str | None:
    return _index(_root(root))["champion"]


def models(root: Path | None = None) -> dict:
    return _index(_root(root))


def meta(version: str, root: Path | None = None) -> dict:
    return json.loads((_root(root) / version / bandit.META_FILE).read_text(encoding="utf-8"))


def can_promote(version: str, root: Path | None = None) -> tuple[bool, str]:
    """Code-enforced promotion rule (the agent cannot bypass it)."""
    idx = _index(_root(root))
    if version not in idx["models"]:
        return False, f"unknown version {version}"
    current = idx["champion"]
    if current is None:
        return True, "no champion yet"
    m = idx["models"][version]
    vs = m.get("vs_champion") or {}
    if config.PROMOTION_RULE == "paired_ci":
        if not m.get("ope_valid", True):
            note = " (point rule: labels not usable for off-policy estimates)"
        elif vs.get("champion") != current or vs.get("ci", [None])[0] is None:
            note = " (point rule: champion not evaluated on this holdout)"
        else:
            (lb, ub), d = vs["ci"], vs["delta"]
            head = f"SNIPS gain vs {current} {d:+.4f}, 95 % CI [{lb:+.4f}, {ub:+.4f}] on {vs['n']} rows: "
            if ub < -config.PROMOTION_MARGIN - 1e-12:
                return False, head + "significantly worse"
            return True, head + ("significantly better" if lb > 0 else "inconclusive, the fresher model is promoted")
    else:
        note = ""
    new_ips = m.get("holdout_ips_take_rate")
    old_ips = idx["models"][current].get("holdout_ips_take_rate")
    if new_ips is None or old_ips is None:
        return False, "missing holdout metrics"
    if new_ips + 1e-12 >= old_ips - config.PROMOTION_TOLERANCE:
        return True, f"IPS {new_ips:.4f} >= champion {current} {old_ips:.4f}{note}"
    return False, f"IPS {new_ips:.4f} < champion {current} {old_ips:.4f}{note}"


def promote(version: str, root: Path | None = None, reason: str = "", force: bool = False) -> dict:
    root = _root(root)
    with _locked(root):
        ok, why = can_promote(version, root)
        if not ok and not force:
            raise ValueError(f"promotion refused: {why}")
        idx = _index(root)
        idx["history"].append({"at": datetime.now().isoformat(timespec="seconds"), "from": idx["champion"], "to": version,
                               "reason": reason or why, "forced": bool(force and not ok)})
        idx["champion"] = version
        _write(root, idx)
    return idx


def rollback(root: Path | None = None) -> dict:
    """Back to the previous champion."""
    root = _root(root)
    with _locked(root):
        idx = _index(root)
        prev = next((h["from"] for h in reversed(idx["history"]) if h["to"] == idx["champion"] and h["from"]), None)
        if prev is None:
            raise ValueError("nothing to roll back to")
        idx["history"].append({"at": datetime.now().isoformat(timespec="seconds"), "from": idx["champion"], "to": prev,
                               "reason": "rollback", "forced": False})
        idx["champion"] = prev
        _write(root, idx)
    return idx


def load(version: str | None = None, root: Path | None = None, features: list[str] | None = None):
    """(vw, meta) of `version` or the champion; refuses an incompatible model."""
    root = _root(root)
    version = version or champion(root)
    if version is None:
        raise FileNotFoundError("no champion model: run fsdr-train first")
    return bandit.load(root / version, features)
