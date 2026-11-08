"""Campaign log and rewards: what was sent to whom, and what came back (prompt §6, audit M1..M4).

    nba_campaign_log.jsonl   one row per (run_date, arm, line): action, offers (code, price, validity),
                             policy_source, propensity, bandit PMF, model version, LLM model,
                             justification id, the line's ARPU anchors. Append-only.
    nba_rewards.jsonl        one row per logged line once the reward window D+7 is known: the three reward
                             definitions (any_btl / same_service / exact_offer), revenue, last BTL date.
                             Append-only.

Both files live in <data root>/state/campaign/, next to the dated run folders, so one log follows the
weekly cycle. Appending never rewrites a row: a key (run_date, arm, msisdn) already present is skipped,
so re-running a step is safe.

Uses: training labels with the logged propensity (H5), last-campaign features (M3), the 28-day contact
cap (L4), the anchor growth guard (M4), and the weekly report per arm.

V1.0 (review #15): the parsed rows and the key set of each file are cached on (size, mtime), and an append only
parses the bytes added since, so appending is no longer a full re-read of a growing file (quadratic over time).
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import config

LOG_FILE = "nba_campaign_log.jsonl"
REWARDS_FILE = "nba_rewards.jsonl"
KEY = ("run_date", "arm", "msisdn")
LOG_FIELDS = ["run_date", "arm", "msisdn", "action", "offers", "policy_source", "propensity", "bandit_pmf",
              "model_version", "llm_model", "justification_id", "anchors", "logged_at"]


def root_of(data_root: str | Path | None = None) -> Path:
    """<data root>/state/campaign (default data root: config.FSDR_DATA_DIR)."""
    return Path(data_root or config.FSDR_DATA_DIR).resolve() / config.FSDR_STATE_DIR.name / config.CAMPAIGN_DIR


def root_for(data_dir: str | Path) -> Path:
    """The log of the run folder <data root>/<D>."""
    return root_of(Path(data_dir).resolve().parent)


_CACHE: dict[str, dict] = {}          # resolved path -> {"size", "mtime", "rows", "keys"}


def _key(r: dict) -> tuple:
    return tuple(str(r.get(k)) for k in KEY)


def _load(path: Path) -> dict:
    """Rows + keys of a JSONL file; re-parses only the bytes appended since the cached state."""
    p = str(path.resolve())
    if not path.exists():
        _CACHE.pop(p, None)
        return {"size": 0, "mtime": 0, "rows": [], "keys": set()}
    st = path.stat()
    c = _CACHE.get(p)
    if c and c["size"] == st.st_size and c["mtime"] == st.st_mtime_ns:
        return c
    start = c["size"] if c and st.st_size > c["size"] else 0       # grown: read the tail; else (rewritten) all
    rows, keys = (list(c["rows"]), set(c["keys"])) if start else ([], set())
    with path.open("rb") as fh:
        fh.seek(start)
        for line in fh.read().decode("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                keys.add(_key(r))
    c = _CACHE[p] = {"size": st.st_size, "mtime": st.st_mtime_ns, "rows": rows, "keys": keys}
    return c


def _read_jsonl(path: Path) -> list[dict]:
    return list(_load(path)["rows"])


def _append(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set(_load(path)["keys"])
    new = []
    for r in rows:
        k = _key(r)
        if k not in seen:
            seen.add(k)
            new.append(r)
    with path.open("a", encoding="utf-8") as fh:
        for r in new:
            fh.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
    return len(new)


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return str(v)


def append_log(root: Path, rows: list[dict]) -> int:
    """Adds campaign rows (missing fields become null); returns how many were new."""
    stamp = datetime.now().isoformat(timespec="seconds")
    return _append(Path(root) / LOG_FILE, [{**{f: None for f in LOG_FIELDS}, "logged_at": stamp, **r} for r in rows])


def append_rewards(root: Path, rows: list[dict]) -> int:
    return _append(Path(root) / REWARDS_FILE, rows)


def read_log(root: Path, start: date | None = None, end: date | None = None, arm: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(_read_jsonl(Path(root) / LOG_FILE), columns=LOG_FIELDS)
    return _filter(df, start, end, arm)


def read_rewards(root: Path, start: date | None = None, end: date | None = None, arm: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(_read_jsonl(Path(root) / REWARDS_FILE))
    return _filter(df, start, end, arm) if len(df) else df


def _filter(df: pd.DataFrame, start, end, arm) -> pd.DataFrame:
    if not len(df):
        return df
    d = pd.to_datetime(df["run_date"]).dt.date
    keep = pd.Series(True, index=df.index)
    if start:
        keep &= d >= start
    if end:
        keep &= d <= end
    if arm:
        keep &= df["arm"] == arm
    return df[keep].reset_index(drop=True)


def justification_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Rows from each arm
# ---------------------------------------------------------------------------
def _pmf(row: pd.Series) -> dict:
    from nbx.core.vw_format import action_token
    return {a: round(float(row.get(f"P_{action_token(a)}", np.nan)), 6) for a in config.ACTIONS
            if pd.notna(row.get(f"P_{action_token(a)}", np.nan))}


def _anchors(anchors: pd.DataFrame | None, val: str) -> dict | None:
    if anchors is None or val not in anchors.index:
        return None
    return {"monthly": round(float(anchors.at[val, "AVG_REV_MONTHLY"]), 6), "weekly": round(float(anchors.at[val, "AVG_REV_WEEKLY"]), 6)}


def rows_from_recos(recos: pd.DataFrame, scored: pd.DataFrame, arm: str, run_date: date,
                    anchors: pd.DataFrame | None = None) -> list[dict]:
    """cstr: the bandit's sampled action and its PMF probability; free: the ranker picks the offers (BR-60),
    the scored action is logged with its probability so the row stays usable for off-policy learning."""
    flag = {"free": "free", "cstr": "with_constraint"}[arm]
    pop = scored[scored["TG_FLAG"] == flag].copy()
    pop["ACCESS_METHOD_VAL"] = pop["ACCESS_METHOD_VAL"].astype(str)
    by_val = {v: g for v, g in recos.groupby("ACCESS_METHOD_VAL")} if len(recos) else {}
    rows = []
    for r in pop.itertuples(index=False):
        s = pd.Series(r._asdict())
        val = s["ACCESS_METHOD_VAL"]
        g = by_val.get(val)
        offers = [] if g is None else [{"code": o.BTL_NAME, "price": float(o.PRICE), "validity": int(o.VALIDITY)}
                                       for o in g.itertuples(index=False)]
        rows.append({"run_date": run_date.isoformat(), "arm": arm, "msisdn": val, "action": s["ACTION"], "offers": offers,
                     "policy_source": "bandit" if arm == "cstr" else "ranker", "propensity": round(float(s["PROBA"]), 6),
                     "bandit_pmf": _pmf(s), "model_version": s.get("MODEL_VERSION"), "llm_model": None,
                     "justification_id": None, "anchors": _anchors(anchors, val)})
    return rows


def rows_from_decisions(decisions: list[dict], anchors: pd.DataFrame | None = None) -> list[dict]:
    """llm arm: one row per agent decision (recommender_agent.decide)."""
    rows = []
    for d in decisions:
        just = d.get("justification") or {}
        rows.append({"run_date": d["run_date"], "arm": "llm", "msisdn": str(d["msisdn"]), "action": d["action"],
                     "offers": [{"code": o["offer"], "price": float(o["price"]), "validity": int(o["validity"])} for o in d.get("offers", [])],
                     "policy_source": d["policy_source"], "propensity": d["propensity"], "bandit_pmf": d.get("bandit_pmf"),
                     "model_version": d.get("model_version"), "llm_model": d.get("model"),
                     "justification_id": justification_id(d["msisdn"], d["run_date"], json.dumps(just, sort_keys=True)),
                     "anchors": _anchors(anchors, str(d["msisdn"]))})
    return rows


def rows_control(sample_vals: list[str], run_date: date) -> list[dict]:
    """cg: logged as not contacted, so the report can compare every arm with the control."""
    return [{"run_date": run_date.isoformat(), "arm": "cg", "msisdn": str(v), "action": config.NO_CONTACT, "offers": [],
             "policy_source": "control", "propensity": 1.0} for v in sample_vals]


# ---------------------------------------------------------------------------
# Rewards (feedback at D+7) and last-campaign features
# ---------------------------------------------------------------------------
def compute_rewards(log: pd.DataFrame, fulfilments: pd.DataFrame, catalog: pd.DataFrame) -> list[dict]:
    """Rewards of logged rows from the fulfilments of the reward window that follows them (BR-23, BR-40).
    fulfilments: ACCESS_METHOD_VAL, PRODUCTNAME, DATE, SERVICE (labels.load_fulfilments)."""
    price = dict(zip(catalog["BTL_NAME"], catalog["PRICE"]))
    service = dict(zip(catalog["BTL_NAME"], catalog["SERVICE"]))
    by_val = {v: g for v, g in fulfilments.groupby("ACCESS_METHOD_VAL")} if len(fulfilments) else {}
    rows = []
    for r in log.itertuples(index=False):
        val, sent = str(r.msisdn), [o["code"] for o in (r.offers or [])]
        f = by_val.get(val)
        bought = [] if f is None else list(f["PRODUCTNAME"])
        sent_services = {service.get(c) for c in sent}
        rows.append({"run_date": r.run_date, "arm": r.arm, "msisdn": val, "action": r.action,
                     "propensity": r.propensity, "policy_source": r.policy_source,
                     "reward_any_btl": int(bool(bought)),
                     "reward_same_service": int(any(service.get(b) in sent_services for b in bought)),
                     "reward_exact_offer": int(any(b in sent for b in bought)),
                     "revenue": round(float(sum(price.get(b, 0.0) for b in bought)), 3),
                     "last_btl_date": None if f is None else max(f["DATE"]).isoformat()})
    return rows


def last_campaign_features(root: Path | None, vals: pd.Series, run_date: date) -> pd.DataFrame:
    """M3, for every line: PREV_ACTION (last logged action, 'NONE' if never contacted), PREV_OFFER_TAKEN
    (exact offer taken after that contact), DAYS_SINCE_LAST_BTL (capped at the 270-day history),
    NB_CONTACTS_LAST_4W (contacts logged in D-28..D-1, control rows excluded)."""
    cap = -config.INTERACTION_WINDOW[0]
    out = pd.DataFrame({"ACCESS_METHOD_VAL": vals.astype(str).to_numpy(), "PREV_ACTION": config.PREV_ACTION_NONE,
                        "PREV_OFFER_TAKEN": 0, "DAYS_SINCE_LAST_BTL": float(cap), "NB_CONTACTS_LAST_4W": 0.0})
    if root is None:
        return out
    log = read_log(root, end=run_date - timedelta(days=1))
    if not len(log):
        return out
    contacts = log[(log["arm"] != "cg") & (log["action"] != config.NO_CONTACT)].copy()
    contacts["d"] = pd.to_datetime(contacts["run_date"]).dt.date
    last = contacts.sort_values("d").drop_duplicates("msisdn", keep="last").set_index("msisdn")
    window = contacts[contacts["d"] >= run_date - timedelta(days=config.CONTACT_CAP_DAYS)]
    n4w = window.groupby("msisdn").size()
    rew = read_rewards(root, end=run_date - timedelta(days=1))
    taken, last_btl = {}, {}
    if len(rew):
        rew = rew.assign(d=pd.to_datetime(rew["run_date"]).dt.date).sort_values("d")
        taken = rew[rew["arm"] != "cg"].drop_duplicates("msisdn", keep="last").set_index("msisdn")["reward_exact_offer"].to_dict()
        lb = rew.dropna(subset=["last_btl_date"])
        last_btl = lb.drop_duplicates("msisdn", keep="last").set_index("msisdn")["last_btl_date"].to_dict()
    v = out["ACCESS_METHOD_VAL"]
    out["PREV_ACTION"] = v.map(last["action"]).fillna(config.PREV_ACTION_NONE).str.replace(" ", "_")
    out["PREV_OFFER_TAKEN"] = v.map(taken).fillna(0).astype(int)
    days = v.map(last_btl).map(lambda s: (run_date - date.fromisoformat(s)).days if isinstance(s, str) else cap)
    out["DAYS_SINCE_LAST_BTL"] = days.clip(upper=cap).astype(float)
    out["NB_CONTACTS_LAST_4W"] = v.map(n4w).fillna(0).astype(float)
    return out


def previous_anchors(root: Path | None, run_date: date) -> dict[str, dict]:
    """Latest anchors logged before run_date, per line (for the M4 growth guard)."""
    if root is None:
        return {}
    log = read_log(root, end=run_date - timedelta(days=1)).dropna(subset=["anchors"])
    if not len(log):
        return {}
    log = log.assign(d=pd.to_datetime(log["run_date"]).dt.date).sort_values("d").drop_duplicates("msisdn", keep="last")
    return dict(zip(log["msisdn"].astype(str), log["anchors"]))
