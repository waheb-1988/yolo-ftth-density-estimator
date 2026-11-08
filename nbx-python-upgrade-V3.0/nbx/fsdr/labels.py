"""Training labels: logged action, reward and propensity per targeted line (BR-21 .. BR-45).

  action      segment -> action (BR-39), case-insensitive, Nursering excluded (BR-36),
              rows without segment dropped (BR-37), latest targeting per line (BR-42)
  reward      three definitions, all computed (plan Q4):
                any_btl       took ANY catalogue BTL in the reward window (BR-40, production)
                same_service  took a BTL of a service that was offered
                exact_offer   took one of the offers that were sent
              Any fulfilment in the reward window D-7..D-1 counts (BR-23, BR-40), as in
              production; FSDR_REWARD_AFTER_TARGETING=true also requires it on or after date1.
  cost        -reward (VW minimises cost, BR-41); V1.0 uplift reward (review #4): -(reward - baseline), the
              baseline = the control group's organic take rate predicted from the features (uplift_adjust)
  propensity  logged propensity if the campaign log has one; else, for the legacy targeting tables,
              1.0 (V1.0 review #1: the legacy policy is deterministic by segment, BR-39) or this week's
              action share (FSDR_TARGETING_PROPENSITY=share, V0), else the historical prior (BR-45), clipped
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from nbx.fsdr import calendar as cal
from nbx.fsdr import config
from nbx.fsdr.datasource import DataSource
from nbx.fsdr.features import norm_id, norm_val
from nbx.fsdr.offer_rules import load_catalog

ACTION_INDEX = {a: i for i, a in enumerate(config.ACTIONS)}


def access_method_map(src: DataSource) -> pd.DataFrame:
    """ACCESS_METHOD_ID -> ACCESS_METHOD_VAL, latest activation kept. MSISDN is ACCESS_METHOD_VAL as is
    (production SQL: `ACCESS_METHOD_VAL as MSISDN`); BR-12's prefix strip happens upstream, in the SQL."""
    am = src.read("access_method_id_auto")
    df = pd.DataFrame({"ACCESS_METHOD_ID": norm_id(am["ACCESS_METHOD_ID"]),
                       "ACCESS_METHOD_VAL": norm_val(am["MSISDN"]),
                       "ACTIVATION_DATE": am["ACTIVATION_DATE"].astype(str)})
    return (df.dropna(subset=["ACCESS_METHOD_ID", "ACCESS_METHOD_VAL"]).sort_values("ACTIVATION_DATE")
              .drop_duplicates("ACCESS_METHOD_ID", keep="last")[["ACCESS_METHOD_ID", "ACCESS_METHOD_VAL"]])


def segment_action(segment) -> str | None:
    if segment is None or (isinstance(segment, float) and np.isnan(segment)):
        return None
    return config.SEGMENT_TO_ACTION.get(str(segment).strip().lower())


def load_fulfilments(src: DataSource, run_date: date) -> pd.DataFrame:
    """Catalogue BTLs fulfilled with success in the reward window (BR-23, BR-25, BR-40, BR-63)."""
    cat = load_catalog(src)
    f = src.read("subs_btl_auto")
    df = pd.DataFrame({"ACCESS_METHOD_VAL": norm_val(f["access_method_val"]),
                       "PRODUCTNAME": f["PRODUCTNAME"].astype(str).str.strip(),
                       "DATE": pd.to_datetime(f["FULFILMENTSUCCESSDATE_TIME"], errors="coerce").dt.date})
    w = cal.campaign_windows(run_date)["reward"]
    df = df.dropna(subset=["ACCESS_METHOD_VAL", "DATE"])
    df = df[(df["DATE"] >= w.start) & (df["DATE"] <= w.end) & df["PRODUCTNAME"].isin(set(cat["BTL_NAME"]))]
    return df.merge(cat[["BTL_NAME", "SERVICE"]], left_on="PRODUCTNAME", right_on="BTL_NAME", how="left")


def build_labels(src: DataSource, feat: pd.DataFrame, run_date: date, reward_def: str = config.REWARD_DEFINITION,
                 reward_after_targeting: bool = config.REWARD_AFTER_TARGETING,
                 logged_propensity: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    if reward_def not in config.REWARD_DEFINITIONS:
        raise ValueError(f"reward_def must be one of {config.REWARD_DEFINITIONS}")
    cat = load_catalog(src).set_index("BTL_NAME")
    raw = src.read("targeting_dataiku_auto")
    offer_cols = [c for c in raw.columns if str(c).lower().startswith("destination_offer_name_")]
    tg = pd.DataFrame({"DATE1": pd.to_datetime(raw["date1"], errors="coerce").dt.date,
                       "ACCESS_METHOD_ID": norm_id(raw["access_method_id"]), "SEGMENT": raw["segment"]})
    tg["OFFERS"] = [[o for o in row if o] for row in raw[offer_cols].itertuples(index=False)] if offer_cols else [[] for _ in range(len(tg))]
    tg["DATE1"] = tg["DATE1"].map(lambda d: d if isinstance(d, date) else cal.campaign_windows(run_date)["targeting"].start)
    n0 = len(tg)
    tg = tg.merge(access_method_map(src), on="ACCESS_METHOD_ID", how="left")
    no_segment = int(tg["SEGMENT"].isna().sum())
    tg = tg.dropna(subset=["SEGMENT"])                                                              # BR-37
    seg = tg["SEGMENT"].astype(str).str.strip().str.lower()
    excluded = int(seg.isin(config.EXCLUDED_SEGMENTS).sum())
    tg = tg[~seg.isin(config.EXCLUDED_SEGMENTS)]                                                    # BR-36
    tg["ACTION"] = tg["SEGMENT"].map(segment_action)                                                # BR-39
    unmapped = tg.loc[tg["ACTION"].isna(), "SEGMENT"].value_counts().to_dict()
    no_msisdn = int(tg["ACCESS_METHOD_VAL"].isna().sum())
    tg = tg.dropna(subset=["ACTION", "ACCESS_METHOD_VAL"])
    tg = tg.sort_values("DATE1").drop_duplicates("ACCESS_METHOD_VAL", keep="last")                   # BR-42

    # rewards (all three definitions)
    ful = load_fulfilments(src, run_date)
    t = tg[["ACCESS_METHOD_VAL", "DATE1", "OFFERS"]].merge(ful, on="ACCESS_METHOD_VAL", how="left")
    if reward_after_targeting:
        t = t[t["DATE"].isna() | (t["DATE"] >= t["DATE1"])]
    took = t.dropna(subset=["PRODUCTNAME"])
    any_btl = set(took["ACCESS_METHOD_VAL"])
    offered_services = {v: {cat.at[o, "SERVICE"] for o in offers if o in cat.index} for v, offers in zip(tg["ACCESS_METHOD_VAL"], tg["OFFERS"])}
    offered = dict(zip(tg["ACCESS_METHOD_VAL"], tg["OFFERS"].map(set)))
    same_service = {r.ACCESS_METHOD_VAL for r in took.itertuples() if r.SERVICE in offered_services.get(r.ACCESS_METHOD_VAL, set())}
    exact = {r.ACCESS_METHOD_VAL for r in took.itertuples() if r.PRODUCTNAME in offered.get(r.ACCESS_METHOD_VAL, set())}
    tg["REWARD_ANY_BTL"] = tg["ACCESS_METHOD_VAL"].isin(any_btl).astype(int)
    tg["REWARD_SAME_SERVICE"] = tg["ACCESS_METHOD_VAL"].isin(same_service).astype(int)
    tg["REWARD_EXACT_OFFER"] = tg["ACCESS_METHOD_VAL"].isin(exact).astype(int)
    tg["REWARD"] = tg[f"REWARD_{reward_def.upper()}"]
    tg["COST"] = -tg["REWARD"]                                                                     # BR-41

    # propensity: logged > legacy policy (deterministic, or V0's share) > prior (BR-44/45)
    if config.TARGETING_PROPENSITY == "deterministic":
        prop = pd.Series(1.0, index=tg.index)
    else:
        share = tg["ACTION"].value_counts(normalize=True)
        prop = tg["ACTION"].map(share).fillna(tg["ACTION"].map(config.ACTION_PRIOR))
    if logged_propensity is not None and len(logged_propensity):
        lp = tg["ACCESS_METHOD_VAL"].map(dict(zip(logged_propensity["ACCESS_METHOD_VAL"], logged_propensity["PROPENSITY"])))
        prop = lp.fillna(prop)
    tg["PROPENSITY"] = prop.clip(*config.PROPENSITY_CLIP)

    train_all = tg.drop(columns=["OFFERS"]).merge(feat, on="ACCESS_METHOD_VAL", how="inner")
    train_all["ACTION_IDX"] = train_all["ACTION"].map(ACTION_INDEX)
    by_action = train_all.groupby("ACTION")["REWARD"].agg(["count", "mean"]).round(4)
    report = {
        "targeting_rows": n0, "no_segment_dropped": no_segment, "nursering_excluded": excluded, "unmapped_segments": unmapped,
        "no_msisdn_dropped": no_msisdn, "labelled_rows": len(train_all), "reward_definition": reward_def,
        "reward_after_targeting": reward_after_targeting, "targeting_propensity": config.TARGETING_PROPENSITY,
        "take_rate": {d: round(float(train_all[f"REWARD_{d.upper()}"].mean()), 4) if len(train_all) else 0.0 for d in config.REWARD_DEFINITIONS},
        "by_action": {a: {"rows": int(r["count"]), "take_rate": float(r["mean"])} for a, r in by_action.iterrows()},
    }
    return train_all, report


def build_labels_from_log(src: DataSource, feat: pd.DataFrame, run_date: date, log: pd.DataFrame,
                          reward_def: str = config.REWARD_DEFINITION) -> tuple[pd.DataFrame, dict]:
    """Labels from last week's campaign log (H5): the action actually sent, the reward from the fulfilments of
    D-7..D-1, and the LOGGED propensity of that action (not a share). Control and no-contact rows are skipped."""
    from nbx.fsdr.campaign_log import compute_rewards

    if reward_def not in config.REWARD_DEFINITIONS:
        raise ValueError(f"reward_def must be one of {config.REWARD_DEFINITIONS}")
    sent = log[(log["arm"] != "cg") & log["action"].isin(config.ACTIONS)].copy()
    sent["msisdn"] = sent["msisdn"].astype(str)
    sent = sent.sort_values("run_date").drop_duplicates("msisdn", keep="last")
    rew = pd.DataFrame(compute_rewards(sent, load_fulfilments(src, run_date), load_catalog(src)))
    if not len(rew):
        return pd.DataFrame(), {"label_source": "campaign_log", "labelled_rows": 0}
    tg = pd.DataFrame({"ACCESS_METHOD_VAL": rew["msisdn"], "ACTION": rew["action"], "ARM": rew["arm"],
                       "REWARD_ANY_BTL": rew["reward_any_btl"], "REWARD_SAME_SERVICE": rew["reward_same_service"],
                       "REWARD_EXACT_OFFER": rew["reward_exact_offer"]})
    tg["REWARD"] = tg[f"REWARD_{reward_def.upper()}"]
    tg["COST"] = -tg["REWARD"]                                                                     # BR-41
    tg["PROPENSITY"] = pd.to_numeric(rew["propensity"], errors="coerce").fillna(
        tg["ACTION"].map(config.ACTION_PRIOR)).clip(*config.PROPENSITY_CLIP)
    train_all = tg.merge(feat, on="ACCESS_METHOD_VAL", how="inner")
    train_all["ACTION_IDX"] = train_all["ACTION"].map(ACTION_INDEX)
    by_action = train_all.groupby("ACTION")["REWARD"].agg(["count", "mean"]).round(4)
    report = {
        "label_source": "campaign_log", "campaign_dates": sorted(set(log["run_date"])), "logged_rows": int(len(sent)),
        "labelled_rows": len(train_all), "reward_definition": reward_def,
        "rows_per_arm": train_all["ARM"].value_counts().to_dict(),
        "take_rate": {d: round(float(train_all[f"REWARD_{d.upper()}"].mean()), 4) if len(train_all) else 0.0 for d in config.REWARD_DEFINITIONS},
        "by_action": {a: {"rows": int(r["count"]), "take_rate": float(r["mean"])} for a, r in by_action.iterrows()},
    }
    return train_all.drop(columns=["ARM"]), report

def uplift_adjust(train_all: pd.DataFrame, src: DataSource, feat: pd.DataFrame, run_date: date, log: pd.DataFrame,
                  features: list[str], reward_def: str = config.REWARD_DEFINITION) -> tuple[pd.DataFrame, dict]:
    """V1.0 review #4 (T-learner, Kunzel et al. 2019): model m0(x) = P(reward | x, not contacted) on the control
    rows (cg) of the same campaign, then cost = -(reward - m0(x)). The action-independent organic part of the
    reward no longer drives the bandit; the report gives the estimated lift per action."""
    from nbx.fsdr.campaign_log import compute_rewards
    from nbx.fsdr.stats import LogReg

    cg = log[log["arm"] == "cg"].copy() if len(log) else log
    if not len(cg) or not len(train_all):
        return train_all, {"applied": False, "reason": "no control rows in the campaign log"}
    cg["msisdn"] = cg["msisdn"].astype(str)
    cg = cg.sort_values("run_date").drop_duplicates("msisdn", keep="last")
    rew = pd.DataFrame(compute_rewards(cg, load_fulfilments(src, run_date), load_catalog(src)))
    ctrl = pd.DataFrame({"ACCESS_METHOD_VAL": rew["msisdn"], "Y": rew[f"reward_{reward_def}"]}).merge(feat, on="ACCESS_METHOD_VAL")
    if len(ctrl) < 50:
        return train_all, {"applied": False, "reason": f"too few control rows ({len(ctrl)})"}
    out = train_all.copy()
    if ctrl["Y"].sum() >= 20:                     # enough control takers for a model of the baseline
        m0 = LogReg(l2=20.0).fit(ctrl[features].to_numpy(float), ctrl["Y"].to_numpy(), features)
        out["BASELINE"], kind = m0.predict(out[features].to_numpy(float)), "logistic"
    else:                                         # else the control group's take rate (smoothed towards 0)
        out["BASELINE"], kind = float(ctrl["Y"].sum() / (len(ctrl) + 10)), "constant"
    out["COST"] = -(out["REWARD"] - out["BASELINE"])
    lift = (out["REWARD"] - out["BASELINE"]).groupby(out["ACTION"]).mean().round(4).to_dict()
    return out, {"applied": True, "baseline_model": kind, "control_rows": int(len(ctrl)), "control_take_rate": round(float(ctrl["Y"].mean()), 4),
                 "mean_baseline": round(float(out["BASELINE"].mean()), 4),
                 "mean_lift": round(float((out["REWARD"] - out["BASELINE"]).mean()), 4), "lift_by_action": lift}
