"""V3.0 offers of the four groups (update prompt §4): universes, anchors, the price selections and the reco tables.

    universe.py   B-20b recommendable offers; constraint_candidates (T1C, T2C) and free_candidates (T1F, T2F)
    anchors.py    B-26 anchors per offer type, CC-03 / CF-03 data share from the customer's own spend
    selection.py  select_code3 (T1C, fallback of T2C), select_code4 (T1F, fallback of T2F), select_challenger (T2C, T2F)
    replay.py     literal replays of reco_btl3 / reco_btl2 selection, for the parity test only
    recos.py      the code-2 tables and the 8 reco tables of a panel (fallback scores until the LLM agents)

Every function works on whole frames (one row per customer x candidate): no per-customer Python loop.
"""
