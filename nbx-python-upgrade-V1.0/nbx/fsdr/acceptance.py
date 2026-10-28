"""Offer acceptance model and value arbitration (V1.0, review #10 / #14; Olaya et al. 2023, Lemmens & Gupta 2020).

    P(accept | customer, action, offer) = logistic regression on
        offer       service, generosity level, validity, log price, price vs the line's own revenue
                    (ratio and gap above it)
        customer    churn / data / voice scores, revenue, 270-day purchase history (share per service and
                    validity, count)
        fit         score x service (data appetite x data offer, ...), action x score (RETAIN x churn, ...)
    trained on last week's campaign: every offer sent (campaign log, or the legacy targeting tables before the
    first logged campaign), label = that exact offer fulfilled in the reward window (BR-23).

The inputs are offer ATTRIBUTES, not offer ids, so an offer never sold has a score too (cold start). The value
policy ranks each family's catalogue offers on P(accept) x price x OFFER_MARGIN; offer_rules still decides which
families are allowed (BR-66/67), how many offers (BR-82) and which pairs conflict.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.datasource import DataSource
from nbx.fsdr.stats import LogReg

GEN_LEVEL = {"L0": 0.0, "L1": 1.0, "L2": 2.0}
COLUMNS = (["SVC_DATA", "SVC_VOICE", "SVC_COMBO", "GEN", "WEEKLY", "LOG_PRICE", "LOG_PRICE_RATIO", "GAP_ABOVE",
            "FIT_DATA", "FIT_VOICE", "HIST_SERVICE", "HIST_VALIDITY", "HIST_N", "CHURN", "APP_DATA", "APP_VOICE",
            "LOG_REV_M", "RETAIN_X_CHURN", "DATA_ACTION_X_APP", "VOICE_ACTION_X_APP"]
           + [f"ACT_{a.replace(' ', '_')}" for a in config.ACTIONS])
DATA_ACTIONS = {"UPSELL DATA", "CROSS SELL DATA"}
VOICE_ACTIONS = {"UPSELL VOICE", "CROSS SELL VOICE"}


def customer_table(src: DataSource, anchors: pd.DataFrame, interactions: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """One row per line: the scores, mean revenue per period and the purchase-history shares the model reads."""
    from nbx.fsdr.features import load_model_scores

    scores, _ = load_model_scores(src)
    cust = pd.DataFrame(index=pd.Index(anchors.index.astype(str), name="ACCESS_METHOD_VAL"))
    s = scores.set_index("ACCESS_METHOD_VAL") if len(scores) else pd.DataFrame()
    for col, src_col in (("CHURN", "PROBA_CHURN"), ("APP_DATA", "PROBA_APP_DATA"), ("APP_VOICE", "PROBA_APP_VOICE")):
        cust[col] = s[src_col].reindex(cust.index).fillna(0.01).to_numpy() if src_col in s.columns else 0.01
    cust["REV_M"] = (anchors["AVG_REV_MONTHLY"] - anchors["STD_REV_MONTHLY"]).clip(lower=0).to_numpy()
    cust["REV_W"] = (anchors["AVG_REV_WEEKLY"] - anchors["STD_REV_WEEKLY"]).clip(lower=0).to_numpy()
    h = interactions.merge(catalog[["BTL_NAME", "SERVICE", "VALIDITY"]], on="BTL_NAME", how="left")
    if len(h):
        n = h.groupby("ACCESS_METHOD_VAL")["WEIGHT"].sum()
        by_svc = h.pivot_table(index="ACCESS_METHOD_VAL", columns="SERVICE", values="WEIGHT", aggfunc="sum", fill_value=0)
        weekly = h[h["VALIDITY"] == 7].groupby("ACCESS_METHOD_VAL")["WEIGHT"].sum()
        cust["HIST_N"] = n.reindex(cust.index).fillna(0).to_numpy()
        for svc in config.SERVICES:
            share = (by_svc[svc] / n) if svc in by_svc.columns else pd.Series(dtype=float)
            cust[f"HIST_{svc}"] = share.reindex(cust.index).fillna(0).to_numpy()
        cust["HIST_WEEKLY"] = (weekly / n).reindex(cust.index).fillna(0).to_numpy()
    else:
        cust["HIST_N"] = 0.0
        for svc in config.SERVICES:
            cust[f"HIST_{svc}"] = 0.0
        cust["HIST_WEEKLY"] = 0.0
    return cust


def design(cust: pd.DataFrame, offers: pd.DataFrame, actions) -> np.ndarray:
    """Rows aligned: cust (customer rows, may repeat), offers (SERVICE, GENEROSITY, VALIDITY, PRICE), actions."""
    svc = offers["SERVICE"].astype(str).to_numpy()
    weekly = (offers["VALIDITY"].astype(int).to_numpy() == 7).astype(float)
    price = offers["PRICE"].astype(float).to_numpy()
    rev = np.where(weekly == 1, cust["REV_W"].to_numpy(), cust["REV_M"].to_numpy())
    anchor = np.maximum(rev, 0.3)
    d, v, c = (svc == "DATA").astype(float), (svc == "VOICE").astype(float), (svc == "COMBO").astype(float)
    app_d, app_v, churn = cust["APP_DATA"].to_numpy(), cust["APP_VOICE"].to_numpy(), cust["CHURN"].to_numpy()
    hist_svc = np.select([d == 1, v == 1, c == 1], [cust["HIST_DATA"].to_numpy(), cust["HIST_VOICE"].to_numpy(),
                                                    cust["HIST_COMBO"].to_numpy()], 0.0)
    hist_val = np.where(weekly == 1, cust["HIST_WEEKLY"].to_numpy(), 1 - cust["HIST_WEEKLY"].to_numpy())
    acts = np.asarray(["" if a is None else str(a) for a in np.broadcast_to(np.asarray(actions, dtype=object), (len(offers),))])
    cols = [d, v, c, offers["GENEROSITY"].map(GEN_LEVEL).fillna(1.0).to_numpy(float), weekly, np.log1p(price),
            np.log(np.maximum(price, 0.05) / anchor), np.maximum(0.0, price - anchor) / anchor,
            app_d * (d + 0.5 * c), app_v * (v + 0.5 * c), hist_svc, hist_val, np.log1p(cust["HIST_N"].to_numpy()),
            churn, app_d, app_v, np.log1p(cust["REV_M"].to_numpy()),
            (acts == "RETAIN") * churn, np.isin(acts, list(DATA_ACTIONS)) * app_d, np.isin(acts, list(VOICE_ACTIONS)) * app_v]
    cols += [(acts == a).astype(float) for a in config.ACTIONS]
    return np.column_stack(cols)


def _sent_offers(src: DataSource, log_root: Path | None, run_date: date) -> tuple[pd.DataFrame, str]:
    """(line, action, offer) of last week's campaign: the campaign log when it has it, else the legacy targeting."""
    from nbx.fsdr import campaign_log
    from nbx.fsdr.labels import access_method_map, segment_action

    lo, hi = config.TARGETING_WINDOW
    if log_root is not None:
        log = campaign_log.read_log(log_root, run_date + timedelta(days=lo), run_date + timedelta(days=hi))
        log = log[(log["arm"] != "cg")] if len(log) else log
        rows = [(str(r.msisdn), r.action, o["code"]) for r in log.itertuples(index=False) for o in (r.offers or [])]
        if rows:
            return pd.DataFrame(rows, columns=["ACCESS_METHOD_VAL", "ACTION", "BTL_NAME"]), "campaign_log"
    from nbx.fsdr.features import norm_id

    raw = src.read("targeting_dataiku_auto")
    offer_cols = [c for c in raw.columns if str(c).lower().startswith("destination_offer_name_")]
    tg = pd.DataFrame({"ACCESS_METHOD_ID": norm_id(raw["access_method_id"]), "ACTION": raw["segment"].map(segment_action),
                       "OFFERS": [[o for o in row if isinstance(o, str) and o] for row in raw[offer_cols].itertuples(index=False)]})
    tg = tg.merge(access_method_map(src), on="ACCESS_METHOD_ID", how="left").dropna(subset=["ACCESS_METHOD_VAL"])
    rows = [(v, a, o) for v, a, offers in zip(tg["ACCESS_METHOD_VAL"], tg["ACTION"], tg["OFFERS"]) for o in offers]
    return pd.DataFrame(rows, columns=["ACCESS_METHOD_VAL", "ACTION", "BTL_NAME"]).drop_duplicates(), "targeting"


class AcceptanceModel:
    def __init__(self, model: LogReg, cust: pd.DataFrame, report: dict):
        self.model, self.cust, self.report = model, cust, report

    @classmethod
    def train(cls, src: DataSource, catalog: pd.DataFrame, anchors: pd.DataFrame, interactions: pd.DataFrame,
              log_root: Path | None, run_date: date) -> tuple["AcceptanceModel | None", dict]:
        from nbx.fsdr.labels import load_fulfilments

        cust = customer_table(src, anchors, interactions, catalog)
        sent, source = _sent_offers(src, log_root, run_date)
        cat = catalog.set_index("BTL_NAME")
        sent = sent[sent["BTL_NAME"].isin(cat.index) & sent["ACCESS_METHOD_VAL"].isin(cust.index)]
        if not len(sent):
            return None, {"trained": False, "source": source, "reason": "no offer sent last week"}
        ful = load_fulfilments(src, run_date)
        bought = set(zip(ful["ACCESS_METHOD_VAL"].astype(str), ful["PRODUCTNAME"]))
        y = np.array([(v, o) in bought for v, o in zip(sent["ACCESS_METHOD_VAL"], sent["BTL_NAME"])], float)
        if y.sum() < config.ACCEPTANCE_MIN_POSITIVES:
            return None, {"trained": False, "source": source, "rows": int(len(sent)), "positives": int(y.sum()),
                          "reason": f"fewer than {config.ACCEPTANCE_MIN_POSITIVES} accepted offers"}
        offers = cat.loc[sent["BTL_NAME"], ["SERVICE", "GENEROSITY", "VALIDITY", "PRICE"]].reset_index(drop=True)
        X = design(cust.loc[sent["ACCESS_METHOD_VAL"]].reset_index(drop=True), offers, sent["ACTION"].to_numpy(dtype=object))
        model = LogReg(l2=5.0).fit(X, y, COLUMNS)
        p = model.predict(X)
        report = {"trained": True, "source": source, "rows": int(len(sent)), "positives": int(y.sum()),
                  "acceptance_rate": round(float(y.mean()), 4), "mean_predicted": round(float(p.mean()), 4),
                  "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
                  "weights": model.weights()}
        return cls(model, cust, report), report

    def p_accept(self, val: str, action: str | None, offers: pd.DataFrame) -> np.ndarray:
        """P(accept) of each offer row (SERVICE, GENEROSITY, VALIDITY, PRICE) for one line."""
        return self.p_accept_many([val], action, offers)[0] if len(offers) else np.zeros(0)

    def p_accept_many(self, vals: list[str], action: str | None, offers: pd.DataFrame) -> np.ndarray:
        """(lines x offers) P(accept); a line missing from the pricing tables is scored as the median customer."""
        if not len(offers) or not len(vals):
            return np.zeros((len(vals), len(offers)))
        if not hasattr(self, "_median"):
            self._median = self.cust.median(numeric_only=True)
        cust = self.cust.reindex([str(v) for v in vals])
        cust = cust.fillna(self._median) if cust.isna().to_numpy().any() else cust
        n, m = len(vals), len(offers)
        c = pd.DataFrame(np.repeat(cust.to_numpy(float), m, axis=0), columns=cust.columns)
        o = pd.DataFrame({k: np.tile(offers[k].to_numpy(), n) for k in ("SERVICE", "GENEROSITY", "VALIDITY", "PRICE")})
        return self.model.predict(design(c, o, action)).reshape(n, m)
