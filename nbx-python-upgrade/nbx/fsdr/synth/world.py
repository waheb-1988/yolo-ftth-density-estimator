"""Hidden customer world for the dummy data and the closed-loop simulation.

Every customer has a hidden state (churn risk, data / voice appetite, price
sensitivity, preferred validity, spend level, usage type). From it the world
produces:
  - a daily behaviour history (calls, minutes, MB, PAYG, subscriptions) --
    the generator aggregates it with the business-rule windows;
  - the hidden signals that drive the 11 model scores (scores are noisy
    readings of the truth, not pure noise);
  - a hidden response function P(accept | customer, action, offer, price).

Daily values come from a random stream keyed on (seed, calendar day, chunk),
so two generation dates of the same world agree on every shared day: W2 of
date D equals W1 of date D-7 by construction.

Nothing here is visible to the pipelines; they only see the JSON tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nbx.fsdr import config

USAGE_TYPES = np.array(["voice_only", "data_only", "dual", "inactive"])
NATIONALITIES = ["Oman", "India", "Bangladesh", "Pakistan", "Egypt", "Philippines", "Sri Lanka", "Nepal",
                 "Jordan", "Sudan", "Yemen", "Syria", "Indonesia", "United Kingdom", "Tanzania"]
NATIONALITY_P = [0.45, 0.22, 0.12, 0.09, 0.025, 0.02, 0.015, 0.012, 0.01, 0.008, 0.008, 0.006, 0.005, 0.004, 0.007]
REGULAR_PLANS = ["FSDR Prepaid Basic", "FSDR Prepaid Plus", "FSDR Prepaid Smart", "FSDR Hala Flex", "FSDR Visitor SIM",
                 "FSDR Shababiah", "FSDR Prepaid Data"]
REGULAR_PLAN_P = [0.30, 0.22, 0.16, 0.12, 0.05, 0.08, 0.07]
EXCLUDED_PLAN_SHARE = 0.02
DV_SEGMENTS = ["DV_SMARTPHONE_HIGH", "DV_SMARTPHONE_MID", "DV_SMARTPHONE_LOW", "DV_FEATURE_PHONE", "DV_ROUTER"]
REGIONS = ["Muscat", "Dhofar", "Musandam", "Al Buraimi", "Ad Dakhiliyah", "North Batinah", "South Batinah",
           "South Sharqiyah", "North Sharqiyah", "Ad Dhahirah", "Al Wusta"]          # the 11 governorates
WAVE_REGIONS = {"North Batinah"}                    # churn_wave hits this region
GENEROSITY_BONUS = {"L0": 0.0, "L1": 0.35, "L2": 0.7}
ORGANIC_WEEKLY_BTL_RATE = 0.03                      # BTL bought without contact (control arm baseline)
RESPONSE_BASE_LOGIT = -5.6                          # tuned: ~10 % take rate per targeted customer (4 offers)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _unique_ids(rng: np.random.Generator, n: int, start: int, max_step: int) -> np.ndarray:
    """n unique integers >= start in random order (cheap, no huge permutation)."""
    ids = start + np.cumsum(rng.integers(1, max_step, n))
    rng.shuffle(ids)
    return ids


# ---------------------------------------------------------------------------
# Offer catalogue
# ---------------------------------------------------------------------------
_PRICE_LADDER = {  # (service, validity) -> {generosity: [prices]}; generosity monotonic with price
    ("DATA", 30): {"L0": [2.0, 3.0, 4.0], "L1": [5.0, 6.0, 7.0], "L2": [8.0, 10.0, 12.0]},
    ("VOICE", 30): {"L0": [1.5, 2.5, 3.5], "L1": [4.0, 5.0, 6.0], "L2": [7.0, 8.5, 10.0]},
    ("COMBO", 30): {"L0": [3.0, 4.5, 6.0], "L1": [7.0, 8.5, 10.0], "L2": [11.0, 13.0, 15.0]},
    ("DATA", 7): {"L0": [0.5, 0.8, 1.0], "L1": [1.2, 1.5, 1.8], "L2": [2.0, 2.5, 3.0]},
    ("VOICE", 7): {"L0": [0.5, 0.7, 0.9], "L1": [1.0, 1.3, 1.6], "L2": [1.8, 2.2, 2.6]},
    ("COMBO", 7): {"L0": [0.8, 1.2, 1.5], "L1": [1.8, 2.2, 2.6], "L2": [3.0, 3.5, 4.0]},
    # add-on families for the batch flow (roaming data, international minutes)
    ("ROAMING", 30): {"L0": [5.0], "L1": [9.0], "L2": [14.0]},
    ("ROAMING", 7): {"L0": [2.0], "L1": [3.5], "L2": [5.0]},
    ("INTERNATIONAL", 30): {"L0": [2.0], "L1": [4.0], "L2": [7.0]},
    ("INTERNATIONAL", 7): {"L0": [0.7], "L1": [1.3], "L2": [2.2]},
}
_GB_PER_OMR = {"L0": 0.8, "L1": 1.2, "L2": 1.8}
_MIN_PER_OMR = {"L0": 25, "L1": 35, "L2": 50}
_GB_SHARE = {"DATA": 1.0, "COMBO": 0.6, "ROAMING": 0.25}           # services that carry data
_MIN_SHARE = {"VOICE": 1.0, "COMBO": 0.5, "INTERNATIONAL": 0.4}     # services that carry minutes


def build_catalog(new_offers: bool = False) -> pd.DataFrame:
    """~54 BTL offers (+3 'new' ones, never bought yet, for the new_offers scenario). Typed frame;
    the generator writes it with the production string types."""
    rows = []
    for (service, validity), ladder in _PRICE_LADDER.items():
        for gen, prices in ladder.items():
            for price in prices:
                rows.append(_offer_row(service, validity, gen, price, is_new=False))
    if new_offers:
        for service, validity, gen, price in [("DATA", 30, "L2", 9.0), ("COMBO", 30, "L2", 12.0), ("COMBO", 7, "L1", 2.4)]:
            rows.append(_offer_row(service, validity, gen, price, is_new=True))
    df = pd.DataFrame(rows).drop_duplicates("btl_name").reset_index(drop=True)
    return df


def _offer_row(service: str, validity: int, gen: str, price: float, is_new: bool) -> dict:
    gb = round(price * _GB_PER_OMR[gen] * _GB_SHARE[service], 1) if service in _GB_SHARE else 0.0
    mins = int(price * _MIN_PER_OMR[gen] * _MIN_SHARE[service]) if service in _MIN_SHARE else 0
    parts = [f"BTL_{service}_{gen}_{validity}D"]
    if gb:
        parts.append(f"{gb:g}GB")
    if mins:
        parts.append(f"{mins}MIN")
    if is_new:
        parts.append("NEW")
    price_data = price if service == "DATA" else (round(price * 0.7, 3) if service == "COMBO" else 0.0)
    return {"btl_name": "_".join(parts), "type": f"BTL_{service}", "generosity": gen, "gb": gb, "min": mins,
            "price": price, "price_data": price_data, "price_voice": round(price - price_data, 3),
            "service": service, "validity": validity, "is_new": is_new}


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
@dataclass
class World:
    seed: int
    scenarios: list[str]
    as_of: date                       # scenario drifts end here (ramp over the previous 4 weeks)
    c: pd.DataFrame                   # one row per customer: ids, demographics, hidden state
    catalog: pd.DataFrame
    params: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.c)

    def has(self, scenario: str) -> bool:
        return scenario in self.scenarios

    # -- creation ------------------------------------------------------------
    @classmethod
    def create(cls, n: int, seed: int = config.GENERATOR_SEED, scenarios: list[str] | None = None,
               as_of: date | None = None, arms: list[str] | None = None) -> "World":
        scenarios = [s for s in (scenarios or ["baseline"]) if s]
        unknown = set(scenarios) - set(config.SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown scenario(s) {sorted(unknown)}; valid: {config.SCENARIOS}")
        if not 1 <= n <= config.GENERATOR_MAX_N:
            raise ValueError(f"n must be between 1 and {config.GENERATOR_MAX_N}")
        as_of = as_of or date.today()
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))

        nat = rng.choice(NATIONALITIES, n, p=np.array(NATIONALITY_P) / sum(NATIONALITY_P))
        omani = nat == "Oman"
        u = rng.random(n)
        tenure = np.where(u < 0.06, rng.integers(1, 45, n),
                          np.where(u < 0.11, rng.integers(45, 91, n),
                                   np.clip(90 + rng.lognormal(np.log(700), 0.9, n), 91, 3650))).astype(int)
        usage_p = np.where(omani[:, None], [[0.22, 0.20, 0.53, 0.05]], [[0.38, 0.10, 0.47, 0.05]])
        usage = USAGE_TYPES[(usage_p.cumsum(axis=1) < rng.random(n)[:, None]).sum(axis=1).clip(0, 3)]
        plan = rng.choice(REGULAR_PLANS, n, p=REGULAR_PLAN_P)
        excl = rng.random(n) < EXCLUDED_PLAN_SHARE
        plan = np.where(excl, rng.choice(config.EXCLUDED_PLANS, n), plan)
        spend = np.clip(rng.lognormal(np.log(4.5), 0.75, n), 0.3, 60.0)          # OMR / month, heavy-tailed
        churn = rng.beta(1.5, 6.0, n)
        churn = np.clip(churn + np.where(tenure < 60, 0.18, 0.0) + np.where(usage == "inactive", 0.35, 0.0), 0, 0.97)
        data_app = rng.beta(2.0, 4.0, n)
        data_app = np.where((usage == "voice_only") & (rng.random(n) < 0.45), np.clip(data_app + 0.4, 0, 1), data_app)
        voice_app = rng.beta(2.0, 4.0, n)
        voice_app = np.where((usage == "data_only") & (rng.random(n) < 0.45), np.clip(voice_app + 0.4, 0, 1), voice_app)
        price_sens = rng.beta(2.0, 3.0, n) * 2.0
        if "price_sensitive" in scenarios:
            price_sens = price_sens * 2.5
        churned = rng.random(n) < churn * 0.35                                   # some lines stop being active
        churn_day = np.where(churned, rng.integers(-40, 25, n), 10_000)          # offset (days) from as_of

        region = rng.choice(REGIONS, n)
        c = pd.DataFrame({
            "access_method_id": _unique_ids(rng, n, 1_000_000, 60),
            "msisdn": _unique_ids(rng, n, 70_000_000, 100).astype(str),
            "account_id": _unique_ids(rng, n, 30_000_000, 50),
            "party_id": _unique_ids(rng, n, 5_000_000, 50),
            "subscription_id": (4_000_000_000 + _unique_ids(rng, n, 0, 40)).astype(str),
            "tenure": tenure,
            "nationality": nat,
            "gender": np.where(omani, rng.choice(["M", "F"], n), rng.choice(["M", "F"], n, p=[0.8, 0.2])),
            "age": np.clip(rng.normal(34, 10, n), 18, 75).astype(int),
            "plan": plan,
            "usage_type": usage,
            "region": region,
            "wilayat": [f"WILAYAT_{REGIONS.index(r) + 1:02d}{k}" for r, k in zip(region, rng.integers(1, 6, n))],
            "area": [f"AREA_{x:04d}" for x in rng.integers(1, 400, n)],
            "dv_segment": rng.choice(DV_SEGMENTS, n, p=[0.25, 0.35, 0.22, 0.12, 0.06]),
            "spend": spend,
            "churn_risk": churn,
            "data_appetite": data_app,
            "voice_appetite": voice_app,
            "price_sensitivity": price_sens,
            "preferred_validity": np.where(rng.random(n) < 0.55, 7, 30),
            "onnet_share": rng.uniform(0.4, 0.8, n),
            "intl_share": np.where(omani, rng.uniform(0.0, 0.04, n), rng.uniform(0.05, 0.3, n)),
            "payg_share": rng.uniform(0.1, 0.6, n),
            "churn_day": churn_day,
            "contacts_last_4w": rng.poisson(0.8, n),
            # travels abroad (drives roaming purchases): more often expats and high spenders
            "traveller": rng.random(n) < np.clip(np.where(omani, 0.05, 0.12) + 0.02 * np.clip(spend / 10, 0, 2), 0, 0.5),
            "wave": np.isin(region, list(WAVE_REGIONS)) if "churn_wave" in scenarios else np.zeros(n, bool),
            "growth": ((usage == "voice_only") & (data_app > 0.5)) if "data_growth" in scenarios else np.zeros(n, bool),
        })
        arms = [a for a in config.ARMS if a in (arms or config.ARMS)]
        if not arms:
            raise ValueError(f"arms must be a subset of {config.ARMS}")
        params = {"fatigue": 0.35 if "campaign_fatigue" in scenarios else 0.05, "arms": arms}
        return cls(seed, scenarios, as_of, c, build_catalog("new_offers" in scenarios), params)

    # -- time-varying state ----------------------------------------------------
    def ramp(self, day: date) -> float:
        """0 before the scenario drift starts, 1 at as_of (linear over SCENARIO_RAMP_DAYS)."""
        start = self.as_of - timedelta(days=config.SCENARIO_RAMP_DAYS)
        return float(np.clip((day - start).days / config.SCENARIO_RAMP_DAYS, 0.0, 1.0))

    def churn_risk_at(self, day: date, idx=slice(None)) -> np.ndarray:
        base = self.c["churn_risk"].to_numpy()[idx]
        wave = self.c["wave"].to_numpy()[idx]
        return np.clip(base + wave * 0.55 * self.ramp(day) * (1 - base), 0, 0.99)

    def data_appetite_at(self, day: date, idx=slice(None)) -> np.ndarray:
        base = self.c["data_appetite"].to_numpy()[idx]
        return np.clip(base + self.c["growth"].to_numpy()[idx] * 0.3 * self.ramp(day), 0, 1)

    # -- daily behaviour -------------------------------------------------------
    def daily(self, day: date, idx: slice, chunk_id: int) -> dict[str, np.ndarray]:
        """One day of behaviour for customers idx. Deterministic in (seed, day, chunk_id)."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2, day.toordinal(), chunk_id]))
        c = self.c.iloc[idx]
        m = len(c)
        usage = c["usage_type"].to_numpy()
        spend = c["spend"].to_numpy()
        churn_eff = self.churn_risk_at(day, idx)
        off = (day - self.as_of).days
        gone = off >= c["churn_day"].to_numpy()
        base_act = np.select([usage == "dual", usage == "voice_only", usage == "data_only"], [0.88, 0.82, 0.82], 0.08)
        p_act = np.where(gone, 0.03, base_act * (1.0 - 0.45 * churn_eff))
        active = rng.random(m) < p_act

        voice_user = usage != "data_only"
        growth_on = c["growth"].to_numpy() & (rng.random(m) < 0.9 * self.ramp(day))
        data_user = (usage == "data_only") | (usage == "dual") | growth_on | ((usage == "inactive") & (rng.random(m) < 0.3))

        level = np.sqrt(spend / 4.5)
        calls = rng.poisson(4.0 * level) * active * voice_user
        mins_total = np.where(calls > 0, rng.gamma(np.maximum(1.5 * calls, 1e-9), 1.6), 0.0)
        intl = mins_total * c["intl_share"].to_numpy()
        local_min = mins_total - intl
        local_calls = np.rint(calls * (1 - c["intl_share"].to_numpy())).astype(int)
        onnet = c["onnet_share"].to_numpy()
        onnet_calls = np.rint(local_calls * onnet).astype(int)

        mb_per_session = 35.0 * level * (1.0 + 0.8 * self.data_appetite_at(day, idx))
        sessions = rng.poisson(np.where(growth_on & (usage == "voice_only"), 2.0 + 4.0 * self.ramp(day), 6.0) * level) * active * data_user
        mb = np.where(sessions > 0, rng.gamma(np.maximum(0.8 * sessions, 1e-9), mb_per_session), 0.0)

        payg = c["payg_share"].to_numpy()
        pg_voice = local_min * 0.012 * payg + intl * 0.08
        pg_data = mb * 0.0015 * payg
        pg_other = np.where(active & (rng.random(m) < 0.1), 0.05, 0.0)
        pg_neg = np.where(active & (rng.random(m) < 0.005), -rng.uniform(0.05, 0.5, m), 0.0)

        p_sub = np.clip(spend * (1 - 0.5 * payg) / 3.2 / 30.0, 0, 0.9) * np.where(gone, 0.05, 1.0)
        buys = rng.random(m) < p_sub
        amt = np.where(buys, np.maximum(0.3, np.round(spend * rng.uniform(0.15, 0.5, m) * 2) / 2), 0.0)
        cls_u = rng.random(m)
        p_data_cls = np.where(data_user, 0.6, 0.1)
        is_data = buys & (cls_u < p_data_cls)
        is_voice = buys & ~is_data & (cls_u < p_data_cls + 0.3)

        return {
            "voice_calls": calls.astype(float),
            "voice_min_rounded": mins_total * 1.07,
            "local_calls": local_calls.astype(float),
            "local_min": local_min,
            "onnet_calls": onnet_calls.astype(float),
            "onnet_min": local_min * onnet,
            "offnet_calls": (local_calls - onnet_calls).astype(float),
            "offnet_min": local_min * (1 - onnet),
            "data_sessions": sessions.astype(float),
            "data_mb_rounded": mb * 1.03,
            "data_mb": mb,
            "pg_voice": pg_voice,
            "pg_data": pg_data,
            "pg_pos": pg_voice + pg_data + pg_other,                 # positive charges only (BR-33 _tg)
            "pg_all": pg_voice + pg_data + pg_other + pg_neg,        # every charge (activities)
            "sub_amt": amt,                                          # SUBSCRIPTION_AMT (BR-19 _tg)
            "sub_tot_amt": amt * 1.05,                               # SUBSCRIPTION_TOT_AMT incl. tax (BR-19 NBA)
            "sub_amt_data": np.where(is_data, amt, 0.0),
            "sub_amt_voice": np.where(is_voice, amt, 0.0),
            "sub_count": buys.astype(float),
            "voice_day": (calls > 0).astype(float),
            "data_day": (sessions > 0).astype(float),
            "any_day": ((calls > 0) | (sessions > 0) | (np.abs(pg_other + pg_neg) > 0)).astype(float),
        }

    # -- hidden signals behind the model scores --------------------------------
    def signal(self, name: str, idx=slice(None), day: date | None = None) -> np.ndarray:
        """True propensity in [0, 1] that model `name` tries to estimate on `day` (default as_of): the scores of
        week D read the world at D, so a scenario drift shows up week after week."""
        day = day or self.as_of
        c = self.c.iloc[idx]
        churn = self.churn_risk_at(day, idx)
        gone = (day - self.as_of).days >= c["churn_day"].to_numpy()
        tenure = c["tenure"].to_numpy()
        usage = c["usage_type"].to_numpy()
        if name == "churn":
            return np.clip(churn + 0.25 * gone + np.where(tenure < 60, 0.08, 0.0), 0, 0.99)
        if name == "churn_1d":
            return np.clip(0.8 * churn * ~gone + 0.05, 0, 0.99)
        if name in ("reactivation_30", "reactivation_15"):
            k = 1.0 if name == "reactivation_30" else 0.75
            low = gone | (usage == "inactive")
            return np.clip(k * np.where(low, 0.75 * (1 - c["churn_risk"].to_numpy()), 0.08), 0, 0.99)
        if name == "data_appetite":
            return self.data_appetite_at(day, idx)
        if name == "voice_appetite":
            return c["voice_appetite"].to_numpy()
        if name == "revenue_decrease":
            return np.clip(0.6 * churn + 0.3 * c["wave"].to_numpy() * self.ramp(day) + 0.05, 0, 0.99)
        if name == "fragility_data":
            return np.clip(np.where(usage == "voice_only", 0.05, 0.5 * churn + 0.3 * (1 - c["data_appetite"].to_numpy())), 0, 0.99)
        if name == "fragility_voice":
            return np.clip(np.where(usage == "data_only", 0.05, 0.5 * churn + 0.3 * (1 - c["voice_appetite"].to_numpy())), 0, 0.99)
        # unknown signal (a model added through config only): a noisy mix of the main signals
        return np.clip(0.5 * churn + 0.5 * self.data_appetite_at(day, idx), 0, 0.99)

    def score(self, name: str, rng: np.random.Generator, idx=slice(None), day: date | None = None) -> np.ndarray:
        """Noisy model score = sigmoid(logit(signal) + noise)."""
        s = self.signal(name, idx, day)
        return _sigmoid(_logit(s) + rng.normal(0.0, 0.6, len(s)))

    # -- hidden response function ---------------------------------------------
    def accept_probability(self, idx: np.ndarray, action: np.ndarray, service: np.ndarray, generosity: np.ndarray,
                           price: np.ndarray, validity: np.ndarray, day: date | None = None) -> np.ndarray:
        """P(accept offer | customer, action, offer, price vs spend) for aligned arrays."""
        day = day or self.as_of
        c = self.c.iloc[idx]
        churn = self.churn_risk_at(day, idx)
        da = self.data_appetite_at(day, idx)
        va = c["voice_appetite"].to_numpy()
        usage = c["usage_type"].to_numpy()
        voice_only, data_only, dual = usage == "voice_only", usage == "data_only", usage == "dual"
        spend = c["spend"].to_numpy()

        fit = np.select(
            [action == "RETAIN", action == "UPSELL DATA", action == "CROSS SELL DATA", action == "UPSELL VOICE",
             action == "CROSS SELL VOICE", action == "UPSELL", action == "MAINTAIN"],
            [2.5 * churn, 2.0 * da * voice_only, 1.5 * da * ~data_only * (da > 0.3), 2.0 * va * data_only,
             1.5 * va * ~voice_only * (va > 0.3), 1.2 * np.clip(spend / 10, 0, 1), 0.8 * (1 - churn) * dual],
            0.0)
        service_fit = np.select([service == "DATA", service == "VOICE"], [da, va], 0.5 * (da + va))
        gen = np.vectorize(GENEROSITY_BONUS.get)(generosity) if len(generosity) else np.zeros(0)
        anchor = np.where(validity == 7, spend / 4.3, spend)
        gap = np.maximum(0.0, price - anchor) / np.maximum(anchor, 0.5)
        logit = (RESPONSE_BASE_LOGIT + 1.6 * fit + 1.2 * service_fit + gen
                 - 1.2 * c["price_sensitivity"].to_numpy() * gap
                 + 0.3 * (validity == c["preferred_validity"].to_numpy())
                 - self.params["fatigue"] * c["contacts_last_4w"].to_numpy())
        return _sigmoid(logit)

    def organic_probability(self, idx: np.ndarray) -> np.ndarray:
        """Weekly chance of buying a BTL with no contact (control arm baseline)."""
        spend = self.c["spend"].to_numpy()[idx]
        return np.clip(ORGANIC_WEEKLY_BTL_RATE + 0.02 * np.clip(spend / 10, 0, 1), 0, 0.2)
