"""Pure helpers behind the Gradio recommenders: MSISDN parsing, the customer
description line, the profile card and the batch table / CSV."""
from __future__ import annotations

import csv

from nbx.ui import gradio_app
from nbx.ui import theme as ui
from nbx.core import insights


def test_parse_msisdns_accepts_mixed_separators_and_reports_garbage():
    ids, bad = gradio_app.parse_msisdns("15, 7  22\n31;15 abc +44")
    assert ids == [15, 7, 22, 31, 44]  # order kept, duplicate 15 dropped, '+' allowed
    assert bad == ["abc"]
    assert gradio_app.parse_msisdns("") == ([], [])


def test_describe_and_profile_card():
    c = insights.customer_360(15)
    line = ui.describe(c)
    assert line.startswith("#15 · ") and "% churn)" in line and "risk" in line
    card = ui.render_profile(c)
    for label in ("ARPU", "Tenure", "Preferred channel", "Propensity scores", "Recent purchases"):
        assert label in card


def test_batch_table_and_csv():
    rows = [
        {"msisdn": 15, "description": "#15 · x", "action": "RETAIN", "channel": "SMS · 19:00", "offers": ["A", "B"],
         "ev": "1.0 OMR/mo", "position": "mixed", "critic": "approve", "why": "short", "narrative": "full why", "time": "20s"},
        {"msisdn": 99999, "error": "unknown MSISDN"},
        {"msisdn": 7, "description": "#7 · y", "pending": "agents are working…"},
    ]
    html = ui.render_batch(rows, "nbo")
    assert "RETAIN" in html and "unknown MSISDN" in html and "agents are working" in html and "Vs competitors" in html
    assert "Vs competitors" not in ui.render_batch(rows[:1], "nba")
    path = gradio_app._write_batch_csv(rows[:2], "nbo")
    with open(path, encoding="utf-8-sig") as f:
        out = list(csv.DictReader(f))
    assert out[0]["offers"] == "A | B" and out[0]["narrative"] == "full why" and out[1]["error"] == "unknown MSISDN"
