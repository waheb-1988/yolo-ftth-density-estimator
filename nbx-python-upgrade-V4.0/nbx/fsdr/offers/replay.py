"""Literal replays of the selection part of code 3 (reco_btl3) and code 4 (reco_btl2), for the parity test A-25 /
update test 8 only. They keep the original structure line by line (data_monthly / hybrid_monthly, the *_only
frames, recos_part1, cust_no_monthly_offers), the original lower-case columns and the original sorts, on a frame
that already holds the scores (the LLM score replaces the LightFM score). Never used to produce an output.
"""
from __future__ import annotations

import pandas as pd

MAP_ACTION_OFFERS = {   # code 3, active version: L2 for all
    "MAINTAIN": {"service": ["COMBO", "DATA"], "generosity": ["L2"]},
    "UPSELL": {"service": ["COMBO", "DATA"], "generosity": ["L2"]},
    "UPSELL VOICE": {"service": ["DATA", "COMBO"], "generosity": ["L2", "L2"]},
    "RETAIN": {"service": ["COMBO", "DATA"], "generosity": ["L2"]},
    "CROSS SELL DATA": {"service": ["VOICE", "COMBO"], "generosity": ["L2", "L2"]},
    "CROSS SELL VOICE": {"service": ["DATA", "COMBO"], "generosity": ["L2", "L2"]},
    "UPSELL DATA": {"service": ["VOICE", "COMBO"], "generosity": ["L2", "L2"]},
}
# code 3 calls: reco_btl3(action, validity, rev_1, rev_2) for the monthly offers (weekly: same with _weekly)
REV_ARGS = {"MAINTAIN": ("avg_rev_data", "avg_rev"), "UPSELL": ("avg_rev_data", "avg_rev"),
            "UPSELL VOICE": ("avg_rev", "avg_rev_data"), "RETAIN": ("avg_rev_data", "avg_rev"),
            "CROSS SELL DATA": ("avg_rev", "avg_rev_voice"), "CROSS SELL VOICE": ("avg_rev", "avg_rev_data"),
            "UPSELL DATA": ("avg_rev", "avg_rev_voice")}


def replay_reco_btl3(m: pd.DataFrame, action: str, validity: int, rev_1: str, rev_2: str) -> pd.DataFrame:
    """m = merge_behav_cust_recos of one action: access_method_val, btl_name, service, generosity, validity, price,
    score + the anchor columns rev_1 / rev_2."""
    mp = MAP_ACTION_OFFERS[action]
    g_data = mp["generosity"][0] if action in ("UPSELL", "MAINTAIN", "RETAIN") else mp["generosity"][1]
    k = 3 if action not in ("MAINTAIN", "RETAIN") else 2
    data_monthly = m.loc[(m["service"] == mp["service"][1]) & (m["generosity"] == g_data) & (m["validity"] == validity)
                         & (m[rev_1] < m["price"]) & (m["price"] < m[rev_1] + k)]
    hybrid_monthly = m.loc[(m["service"] == mp["service"][0]) & (m["generosity"] == mp["generosity"][0])
                           & (m["validity"] == validity) & (m[rev_2] < m["price"]) & (m["price"] < m[rev_2] + k)]
    hybrid_monthly_offers = hybrid_monthly.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, False]).drop_duplicates("access_method_val")
    data_monthly_offers = data_monthly.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, False]).drop_duplicates("access_method_val")
    hyb_data = hybrid_monthly_offers[hybrid_monthly_offers.access_method_val.isin(data_monthly_offers.access_method_val.unique())]
    data_hyb = data_monthly_offers[data_monthly_offers.access_method_val.isin(hybrid_monthly_offers.access_method_val.unique())]
    offers_1 = pd.concat([hyb_data, data_hyb])
    hybrid_only = hybrid_monthly_offers[~hybrid_monthly_offers.access_method_val.isin(data_monthly_offers.access_method_val.unique())]
    data_only = data_monthly_offers[~data_monthly_offers.access_method_val.isin(hybrid_monthly_offers.access_method_val.unique())]
    m_data_only = m[m.access_method_val.isin(data_only.access_method_val.unique())]
    m_hyb_only = m[m.access_method_val.isin(hybrid_only.access_method_val.unique())]
    data_2 = m_hyb_only.loc[(m_hyb_only["service"] == mp["service"][1]) & (m_hyb_only["generosity"] == g_data)
                            & (m_hyb_only["validity"] == validity) & (m_hyb_only[rev_1] < m_hyb_only["price"])]
    hybrid_2 = m_data_only.loc[(m_data_only["service"] == mp["service"][0]) & (m_data_only["generosity"] == mp["generosity"][0])
                               & (m_data_only["validity"] == validity) & (m_data_only[rev_2] < m_data_only["price"])]
    hybrid_2_offers = hybrid_2.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, True]).drop_duplicates("access_method_val")
    data_2_offers = data_2.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, True]).drop_duplicates("access_method_val")
    offers_2 = pd.concat([pd.concat([data_2_offers, hybrid_only]), pd.concat([hybrid_2_offers, data_only])])
    recos_part1 = pd.concat([offers_1, offers_2])
    none = m[~m.access_method_val.isin(recos_part1.access_method_val.unique())]
    data_3 = none.loc[(none["service"] == mp["service"][1]) & (none["generosity"] == g_data) & (none["validity"] == validity)
                      & (none[rev_1] < none["price"])]
    hybrid_3 = none.loc[(none["service"] == mp["service"][0]) & (none["generosity"] == mp["generosity"][0])
                        & (none["validity"] == validity) & (none[rev_2] < none["price"])]
    hybrid_3_offers = hybrid_3.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, True]).drop_duplicates("access_method_val")
    data_3_offers = data_3.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, True]).drop_duplicates("access_method_val")
    return pd.concat([offers_1, offers_2, pd.concat([data_3_offers, hybrid_3_offers])])


def replay_reco_btl2(m: pd.DataFrame, validity: int, rev_global: str, rev_data: str) -> pd.DataFrame:
    """Code 4: DATA vs rev_data, COMBO vs rev_global, k = 3 (m already holds the L2, no-VOICE universe)."""
    data_monthly = m.loc[(m["service"] == "DATA") & (m["validity"] == validity) & (m[rev_data] < m["price"]) & (m["price"] < m[rev_data] + 3)]
    hybrid_monthly = m.loc[(m["service"] == "COMBO") & (m["validity"] == validity) & (m[rev_global] < m["price"]) & (m["price"] < m[rev_global] + 3)]
    hybrid_monthly_offers = hybrid_monthly.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, False]).drop_duplicates("access_method_val")
    data_monthly_offers = data_monthly.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, False]).drop_duplicates("access_method_val")
    hyb_data = hybrid_monthly_offers[hybrid_monthly_offers.access_method_val.isin(data_monthly_offers.access_method_val.unique())]
    data_hyb = data_monthly_offers[data_monthly_offers.access_method_val.isin(hybrid_monthly_offers.access_method_val.unique())]
    offers_1 = pd.concat([hyb_data, data_hyb])
    hybrid_only = hybrid_monthly_offers[~hybrid_monthly_offers.access_method_val.isin(data_monthly_offers.access_method_val.unique())]
    data_only = data_monthly_offers[~data_monthly_offers.access_method_val.isin(hybrid_monthly_offers.access_method_val.unique())]
    m_data_only = m[m.access_method_val.isin(data_only.access_method_val.unique())]
    m_hyb_only = m[m.access_method_val.isin(hybrid_only.access_method_val.unique())]
    data_2 = m_hyb_only.loc[(m_hyb_only["service"] == "DATA") & (m_hyb_only["validity"] == validity) & (m_hyb_only[rev_data] < m_hyb_only["price"])]
    hybrid_2 = m_data_only.loc[(m_data_only["service"] == "COMBO") & (m_data_only["validity"] == validity) & (m_data_only[rev_global] < m_data_only["price"])]
    hybrid_2_offers = hybrid_2.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, True]).drop_duplicates("access_method_val")
    data_2_offers = data_2.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, True]).drop_duplicates("access_method_val")
    offers_2 = pd.concat([pd.concat([data_2_offers, hybrid_only]), pd.concat([hybrid_2_offers, data_only])])
    recos_part1 = pd.concat([offers_1, offers_2])
    none = m[~m.access_method_val.isin(recos_part1.access_method_val.unique())]
    data_3 = none.loc[(none["service"] == "DATA") & (none["validity"] == validity) & (none[rev_data] < none["price"])]
    hybrid_3 = none.loc[(none["service"] == "COMBO") & (none["validity"] == validity) & (none[rev_global] < none["price"])]
    hybrid_3_offers = hybrid_3.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, True]).drop_duplicates("access_method_val")
    data_3_offers = data_3.sort_values(by=["access_method_val", "score", "price"], ascending=[True, False, True]).drop_duplicates("access_method_val")
    return pd.concat([offers_1, offers_2, pd.concat([data_3_offers, hybrid_3_offers])])
