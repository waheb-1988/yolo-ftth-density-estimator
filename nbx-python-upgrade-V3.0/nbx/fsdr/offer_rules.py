"""Offer business rules -- port of production Codes 3 (constraint arm) and 4 (free arm).

  - catalogue, conflicts, eligibility                       BR-63 .. BR-65, conflicting_offers_h
  - ARPU anchor = mean + std of periods, data 90 % / voice 10 %     BR-68 .. BR-77
  - action -> offer families and generosity (normal | all_L2)      BR-66, free arm BR-67
  - 3-pass price-band selection, at most 2 offers per validity      BR-78 .. BR-82
  - NEW: two conflicting offers are never sent together

Fixes vs the original: CROSS SELL VOICE gets its own family filter (the
original repeated "CROSS SELL DATA"), and conflicts are enforced.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.calendar import MONTHS, WEEKS
from nbx.fsdr.datasource import DataSource
from nbx.fsdr.features import norm_val, upper

ANCHOR_BY_SERVICE = {"COMBO": "TOTAL", "DATA": "DATA", "VOICE": "VOICE"}   # BR-77


# ---------------------------------------------------------------------------
# Catalogue and conflicts
# ---------------------------------------------------------------------------
def load_catalog(src: DataSource) -> pd.DataFrame:
    """btl_catalog with numeric price / validity / gb / min (production stores them as strings)."""
    df = src.read("btl_catalog").copy()
    df.columns = [str(c).upper().replace(" ", "_") for c in df.columns]
    for c in ("PRICE", "GB", "MIN", "PRICE_DATA", "PRICE_VOICE"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["VALIDITY"] = pd.to_numeric(df["VALIDITY"], errors="coerce").astype("Int64")
    for c in ("SERVICE", "GENEROSITY"):
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["BTL_NAME"] = df["BTL_NAME"].astype(str).str.strip()
    return df.drop_duplicates("BTL_NAME").reset_index(drop=True)


def catalog_version(catalog: pd.DataFrame) -> str:
    key = "|".join(f"{r.BTL_NAME}:{r.PRICE}:{r.VALIDITY}" for r in catalog.sort_values("BTL_NAME").itertuples())
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def load_conflicts(src: DataSource) -> set[frozenset]:
    if not src.has("conflicting_offers_h"):
        return set()
    df = src.read("conflicting_offers_h")
    return {frozenset((str(a).strip(), str(b).strip())) for a, b in zip(df["Offer_A"], df["Offer_B"]) if a and b}


def conflicts_with(name: str, chosen: list[str], conflicts: set[frozenset]) -> bool:
    return any(frozenset((name, c)) in conflicts for c in chosen)


# ---------------------------------------------------------------------------
# ARPU anchors (BR-68 .. BR-77)
# ---------------------------------------------------------------------------
def arpu_anchors(src: DataSource) -> pd.DataFrame:
    """One row per line: revenue per period = subscriptions + total PAYG (BR-68),
    anchor = mean + std (BR-71, pandas ddof=1), data = 0.9 x anchor, voice = 0.1 x anchor."""
    sub = upper(src.read("subscriptions_tg")) if src.has("subscriptions_tg") else pd.DataFrame(columns=["ACCESS_METHOD_VAL"])
    use = upper(src.read("usage_data_voice_payg_tg")) if src.has("usage_data_voice_payg_tg") else pd.DataFrame(columns=["ACCESS_METHOD_VAL"])
    m_cols = [f"TOTAL_SUB_AMT_{m}" for m in MONTHS] + [f"TOTAL_SUB_AMT_{w}" for w in WEEKS]
    p_cols = [f"PG_TOTAL_AMOUNT_{p}" for p in MONTHS + WEEKS]
    sub = sub[["ACCESS_METHOD_VAL"] + [c for c in m_cols if c in sub.columns]].copy()
    use = use[["ACCESS_METHOD_VAL"] + [c for c in p_cols if c in use.columns]].copy()
    for df in (sub, use):
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    sub = sub.dropna(subset=["ACCESS_METHOD_VAL"]).groupby("ACCESS_METHOD_VAL").sum()
    use = use.dropna(subset=["ACCESS_METHOD_VAL"]).groupby("ACCESS_METHOD_VAL").sum()
    out = sub.join(use, how="outer").fillna(0.0)                                       # BR-76 missing = 0
    for c in m_cols + p_cols:
        if c not in out.columns:
            out[c] = 0.0
    for p in MONTHS + WEEKS:
        out[f"REV_{p}"] = out[f"TOTAL_SUB_AMT_{p}"] + out[f"PG_TOTAL_AMOUNT_{p}"]
    for name, periods in (("MONTHLY", MONTHS), ("WEEKLY", WEEKS)):
        cols = [f"REV_{p}" for p in periods]
        out[f"MAX_REV_{name}"] = out[cols].max(axis=1)                                  # BR-72 (not used)
        out[f"STD_REV_{name}"] = out[cols].std(axis=1, ddof=1).fillna(0.0)
        out[f"AVG_REV_{name}"] = out[cols].mean(axis=1) + out[f"STD_REV_{name}"]        # BR-71
        out[f"AVG_REV_DATA_{name}"] = config.ARPU_DATA_SHARE * out[f"AVG_REV_{name}"]   # BR-74
        out[f"AVG_REV_VOICE_{name}"] = config.ARPU_VOICE_SHARE * out[f"AVG_REV_{name}"]  # BR-75
    keep = [f"REV_{p}" for p in MONTHS + WEEKS] + [c for c in out.columns if c.startswith(("AVG_REV", "STD_REV", "MAX_REV"))]
    return out[keep]


def cap_anchor_growth(anchors: pd.DataFrame, previous: dict[str, dict], growth: float) -> pd.DataFrame:
    """M4 price-escalation guard: AVG_REV_* at most (1 + growth) x the anchor logged for the line last time
    (campaign purchases raise next week's spend, so anchors -- and prices -- could climb week after week).
    The data / voice anchors follow (BR-74/75). Lines never logged keep their anchor."""
    out = anchors.copy()
    out["ANCHOR_CAPPED"] = False
    for period, key in (("MONTHLY", "monthly"), ("WEEKLY", "weekly")):
        prev = pd.Series({v: a.get(key) for v, a in previous.items() if a and a.get(key) is not None}, dtype=float)
        prev = prev.reindex(out.index)
        cap = prev * (1.0 + growth)
        over = prev.notna() & (out[f"AVG_REV_{period}"] > cap)
        out.loc[over, f"AVG_REV_{period}"] = cap[over]
        out["ANCHOR_CAPPED"] |= over
        out[f"AVG_REV_DATA_{period}"] = config.ARPU_DATA_SHARE * out[f"AVG_REV_{period}"]
        out[f"AVG_REV_VOICE_{period}"] = config.ARPU_VOICE_SHARE * out[f"AVG_REV_{period}"]
    return out


def anchor_column(service: str, validity: int) -> str:
    """BR-77: COMBO vs total ARPU, DATA vs data ARPU, VOICE vs voice ARPU; weekly offers use weekly anchors.
    Services outside BR-77 (the roaming / international add-on families) use the total anchor, as the add-on
    engine does -- an unknown service must never crash an agent tool."""
    kind = ANCHOR_BY_SERVICE.get(str(service).upper(), "TOTAL")
    period = "WEEKLY" if int(validity) == 7 else "MONTHLY"
    return f"AVG_REV_{period}" if kind == "TOTAL" else f"AVG_REV_{kind}_{period}"


def anchor_value(anchors: pd.DataFrame, val: str, service: str, validity: int) -> float:
    if val not in anchors.index:
        return 0.0
    return float(anchors.at[val, anchor_column(service, validity)])


# ---------------------------------------------------------------------------
# Families (BR-66, BR-67) and price margins (BR-78)
# ---------------------------------------------------------------------------
def families(action: str | None, arm: str, mapping: str | None = None) -> list[tuple[str, str]]:
    """[(service, generosity) for type A, type B]. cstr: from the action; free: L2 COMBO + DATA."""
    if arm == "free":
        return [(s, config.FREE_ARM_GENEROSITY) for s in config.FREE_ARM_SERVICES]
    if action not in config.ACTION_FAMILIES:
        raise ValueError(f"Unknown action {action!r}")
    svc_a, svc_b = config.ACTION_FAMILIES[action]                 # CROSS SELL VOICE has its own entry (typo fixed)
    mapping = mapping or config.GENEROSITY_MAPPING
    if mapping == "all_L2":
        return [(svc_a, "L2"), (svc_b, "L2")]
    gen_a, gen_b = config.ACTION_GENEROSITY_NORMAL[action]
    return [(svc_a, gen_a), (svc_b, gen_b)]


def k_margin(action: str | None, arm: str) -> float:
    return config.PRICE_BAND_K["free" if arm == "free" else action]


def family_items(catalog: pd.DataFrame, fams: list[tuple[str, str]], validity: int) -> pd.DataFrame:
    mask = np.zeros(len(catalog), bool)
    for svc, gen in fams:
        mask |= (catalog["SERVICE"] == svc) & (catalog["GENEROSITY"] == gen)
    return catalog[mask & (catalog["VALIDITY"] == validity)]


# ---------------------------------------------------------------------------
# 3-pass selection (BR-78 .. BR-82)
# ---------------------------------------------------------------------------
@dataclass
class Pick:
    btl_name: str
    service: str
    generosity: str
    validity: int
    price: float
    score: float
    rank: int
    pass_no: int
    anchor: float
    anchor_type: str
    p_accept: float | None = None        # V1.0 value policy: acceptance model's P(accept); None for the price bands

    def as_dict(self) -> dict:
        return asdict(self)


def select_offers(cands: pd.DataFrame, anchors: pd.DataFrame, val: str, fams: list[tuple[str, str]], k: float,
                  validity: int, conflicts: set[frozenset], already: list[str] | None = None) -> list[Pick]:
    """At most one offer per family (so <= 2) for one customer and one validity.

    cands: ranker candidates for this customer (BTL_NAME, SERVICE, GENEROSITY, VALIDITY, PRICE, SCORE, RANK).
    pass 1: anchor < price < anchor + k, sorted score desc, price desc        (BR-78, BR-81 since 9 Oct)
    pass 2: the missing type only, price > anchor, no cap, score desc, price asc (BR-79)
    pass 3: no offer yet: both types, price > anchor, no cap, score desc, price asc (BR-80)
    A customer whose anchor is above every price gets nothing (BR-82)."""
    chosen: list[Pick] = []
    taken = list(already or [])
    c = cands[cands["VALIDITY"] == validity]
    # numpy arrays once per call: pick() runs up to 4 times per customer (review #15: pandas filtering per pick was
    # ~4 ms, i.e. hours for a 1.7 M base); same filters, same order (score desc, then price; ties keep input order)
    names, svcs, gens = c["BTL_NAME"].to_numpy(), c["SERVICE"].to_numpy(), c["GENEROSITY"].to_numpy()
    price, score, rank_ = c["PRICE"].to_numpy(float), c["SCORE"].to_numpy(float), c["RANK"].to_numpy()

    def pick(svc: str, gen: str, lo: float, hi: float | None, asc_price: bool, pass_no: int) -> Pick | None:
        m = (svcs == svc) & (gens == gen) & (price > lo)
        if hi is not None:
            m &= price < hi
        idx = np.flatnonzero(m)
        for j in idx[np.lexsort((price[idx] if asc_price else -price[idx], -score[idx]))]:
            if not conflicts_with(names[j], taken, conflicts):
                return Pick(names[j], svc, gen, int(validity), float(price[j]), float(score[j]), int(rank_[j]), pass_no,
                            round(lo, 6), anchor_column(svc, validity))   # 6 dp: 12.9995 must not print as 13.0
        return None

    anchors_f = [anchor_value(anchors, val, svc, validity) for svc, _ in fams]
    first = []
    for (svc, gen), a in zip(fams, anchors_f):          # sequential: the 2nd pick must see the 1st (conflicts)
        p = pick(svc, gen, a, a + k, False, 1)
        first.append(p)
        if p:
            chosen.append(p)
            taken.append(p.btl_name)
    found = [p is not None for p in first]
    if any(found) and not all(found):                       # pass 2: missing type only
        i = found.index(False)
        svc, gen = fams[i]
        p = pick(svc, gen, anchors_f[i], None, True, 2)
        if p:
            chosen.append(p)
            taken.append(p.btl_name)
    elif not any(found):                                    # pass 3: nothing yet
        for (svc, gen), a in zip(fams, anchors_f):
            p = pick(svc, gen, a, None, True, 3)
            if p:
                chosen.append(p)
                taken.append(p.btl_name)
    return chosen[: config.MAX_OFFERS_PER_VALIDITY]


# ---------------------------------------------------------------------------
# Hard rules (code-enforced; the LLM cannot override them)
# ---------------------------------------------------------------------------
def check_offers(names: list[str], catalog: pd.DataFrame, conflicts: set[frozenset]) -> list[str]:
    """Violations for a proposed list of offers (empty list = all hard rules pass)."""
    v: list[str] = []
    cat = catalog.set_index("BTL_NAME")
    if len(names) != len(set(names)):
        v.append("duplicate offer in the list")
    known = [n for n in names if n in cat.index]
    for n in names:
        if n not in cat.index:
            v.append(f"{n}: not in the catalogue (BR-63)")
        elif not cat.at[n, "PRICE"] > 0:
            v.append(f"{n}: price must be > 0")
    if len(names) > config.MAX_OFFERS_TOTAL:
        v.append(f"{len(names)} offers: at most {config.MAX_OFFERS_TOTAL} in total (BR-82)")
    for validity in config.VALIDITIES:
        k = sum(1 for n in known if int(cat.at[n, "VALIDITY"]) == validity)
        if k > config.MAX_OFFERS_PER_VALIDITY:
            v.append(f"{k} offers with validity {validity}: at most {config.MAX_OFFERS_PER_VALIDITY} (BR-82)")
    for i, a in enumerate(known):
        for b in known[i + 1:]:
            if frozenset((a, b)) in conflicts:
                v.append(f"{a} and {b} conflict (conflicting_offers_h)")
    return v


def customer_eligibility(base_row: dict | None, segment: str | None = None) -> list[str]:
    """Reasons a line may not be contacted (BR-02, BR-10, BR-36); empty = eligible."""
    if base_row is None:
        return ["not in the eligible prepaid base"]
    reasons = []
    val = str(base_row.get("ACCESS_METHOD_VAL") or "")
    if not (len(val) == 8 and val.isdigit() and val > config.MSISDN_MIN_EXCLUSIVE):
        reasons.append("MSISDN outside the prepaid range (BR-02)")
    if base_row.get("CURRENT_PLAN") in config.EXCLUDED_PLANS:
        reasons.append("internal / employee plan (BR-10)")
    if segment and segment.strip().lower() in config.EXCLUDED_SEGMENTS:
        reasons.append("Nursering segment, tenure < 45 days (BR-36)")
    return reasons
