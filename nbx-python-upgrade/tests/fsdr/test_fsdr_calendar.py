"""Anchor date T and every BR §4 window."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from nbx.fsdr import calendar as cal


@pytest.mark.parametrize("d, expected", [
    (date(2026, 3, 12), date(2026, 3, 12)),     # the reference Thursday itself
    (date(2026, 10, 22), date(2026, 10, 22)),   # a Thursday
    (date(2026, 10, 21), date(2026, 10, 15)),   # a Wednesday -> previous Thursday
    (date(2026, 10, 16), date(2026, 10, 15)),   # the Friday after
    (date(2027, 1, 1), date(2026, 12, 31)),     # year rollover
    (date(2026, 1, 1), date(2026, 1, 1)),       # a Thursday before the reference date
    (date(2026, 1, 3), date(2026, 1, 1)),       # a Saturday before the reference date
])
def test_anchor_is_last_thursday_on_or_before(d, expected):
    t = cal.anchor_date(d)
    assert t == expected and t.weekday() == 3 and 0 <= (d - t).days <= 6


def test_subscription_windows_include_the_anchor_day():
    t = date(2026, 10, 15)
    w = cal.subs_windows(t)
    assert (w["W1"].start, w["W1"].end) == (t - timedelta(6), t)
    assert (w["W4"].start, w["W4"].end) == (t - timedelta(27), t - timedelta(21))
    assert (w["M1"].days, w["M2"].days, w["M3"].days) == (30, 30, 30)
    assert w["M3"].start == t - timedelta(89)


def test_activity_windows_exclude_the_anchor_day_and_m1_equals_four_weeks():
    t = date(2026, 10, 15)
    w = cal.activity_windows(t)
    assert w["W1"].end == t - timedelta(1) and w["W1"].start == t - timedelta(7)
    assert (w["M1"].days, w["M2"].days, w["M3"].days) == (28, 28, 29)
    assert w["M1"].start == w["W4"].start and w["M1"].end == w["W1"].end
    for a, b in (("W1", "W2"), ("W2", "W3"), ("W3", "W4"), ("M1", "M2"), ("M2", "M3")):
        assert w[b].end + timedelta(1) == w[a].start      # contiguous, no gap, no overlap


def test_pricing_windows_lag_two_days():
    d = date(2026, 10, 21)
    s = cal.subscriptions_tg_windows(d)
    assert (s["W1"].start, s["W1"].end) == (d - timedelta(8), d - timedelta(2))
    assert (s["M1"].start, s["M1"].end) == (d - timedelta(31), d - timedelta(2))
    p = cal.payg_windows(d)
    assert (p["M1"].start, p["M1"].end) == (d - timedelta(30), d - timedelta(1))


def test_campaign_windows():
    d = date(2026, 10, 21)
    c = cal.campaign_windows(d)
    assert (c["targeting"].start, c["targeting"].end) == (d - timedelta(8), d - timedelta(2))   # BR-21
    assert (c["reward"].start, c["reward"].end) == (d - timedelta(7), d - timedelta(1))         # BR-23
    assert c["interactions"].start == d - timedelta(270)                                          # BR-24


@pytest.mark.parametrize("d, expected", [(date(2026, 10, 21), (9, 2026)), (date(2027, 1, 10), (12, 2026)),
                                         (date(2027, 1, 2), (11, 2026)), (date(2027, 3, 2), (1, 2027))])
def test_location_month_is_previous_month_of_d_minus_2(d, expected):
    assert cal.location_month(d) == expected


@pytest.mark.parametrize("offset", range(7))
def test_history_start_covers_every_window_for_any_weekday(offset):
    d = date(2026, 10, 15) + timedelta(offset)
    start = cal.usage_history_start(d)
    t = cal.anchor_date(d)
    groups = [cal.subs_windows(t), cal.subs_windows(d), cal.activity_windows(t), cal.activity_windows(d),
              cal.subscriptions_tg_windows(d), cal.usage_tg_volume_windows(d), cal.payg_windows(d), cal.payg_windows(t)]
    assert min(w.start for g in groups for w in g.values()) >= start
    assert max(w.end for g in groups for w in g.values()) <= d
