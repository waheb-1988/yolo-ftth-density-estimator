"""Anchor date and every aggregation window of the business rules (BR §4).

D = run date, T = campaign anchor = most recent Thursday on or before D.
Windows are inclusive [start, end] date ranges. The generator aggregates its
daily history with exactly these windows, and the validation / features use
them to know what each column covers, so the two can never disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from nbx.fsdr import config

WEEKS = ["W1", "W2", "W3", "W4"]      # W1 = most recent week
MONTHS = ["M1", "M2", "M3"]           # M1 = most recent month
PERIODS = WEEKS + MONTHS


@dataclass(frozen=True)
class Window:
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end

    def as_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "days": self.days}


def anchor_date(run_date: date) -> date:
    """T = D - (((D - 2026-03-12) MOD 7) + 7) MOD 7 -- the last Thursday on or before D."""
    return run_date - timedelta(days=(((run_date - config.REFERENCE_THURSDAY).days % 7) + 7) % 7)


def _w(anchor: date, start_off: int, end_off: int) -> Window:
    return Window(anchor + timedelta(days=start_off), anchor + timedelta(days=end_off))


# Offsets (start, end) relative to the anchor, per BR §4.
_SUBS_OFFSETS = {                     # BR 4.1 -- window INCLUDES the anchor day
    "W1": (-6, 0), "W2": (-13, -7), "W3": (-20, -14), "W4": (-27, -21),
    "M1": (-29, 0), "M2": (-59, -30), "M3": (-89, -60),
}
_ACTIVITY_OFFSETS = {                 # BR 4.2 -- window EXCLUDES the anchor day
    "W1": (-7, -1), "W2": (-14, -8), "W3": (-21, -15), "W4": (-28, -22),
    "M1": (-28, -1), "M2": (-56, -29), "M3": (-85, -57),
}
_SUBSCRIPTIONS_TG_OFFSETS = {         # BR 4.3 subscriptions_tg, anchored on D (2-day lag)
    "W1": (-8, -2), "W2": (-15, -9), "W3": (-22, -16), "W4": (-29, -23),
    "M1": (-31, -2), "M2": (-61, -32), "M3": (-91, -62),
}
_USAGE_TG_VOLUME_OFFSETS = {          # BR 4.3 usage_data_voice_payg_tg voice & data p1..p3
    "p1": (-31, -2), "p2": (-61, -32), "p3": (-91, -62),
}
_PAYG_OFFSETS = {                     # BR 4.3 PAYG windows (usage_data_voice_payg_tg on D, paygo_activity_auto on T)
    "M1": (-30, -1), "M2": (-60, -31), "M3": (-90, -61),
    "W1": (-7, -1), "W2": (-14, -8), "W3": (-21, -15), "W4": (-28, -22),
}


def subs_windows(anchor: date) -> dict[str, Window]:
    return {p: _w(anchor, *o) for p, o in _SUBS_OFFSETS.items()}


def activity_windows(anchor: date) -> dict[str, Window]:
    return {p: _w(anchor, *o) for p, o in _ACTIVITY_OFFSETS.items()}


def subscriptions_tg_windows(run_date: date) -> dict[str, Window]:
    return {p: _w(run_date, *o) for p, o in _SUBSCRIPTIONS_TG_OFFSETS.items()}


def usage_tg_volume_windows(run_date: date) -> dict[str, Window]:
    return {p: _w(run_date, *o) for p, o in _USAGE_TG_VOLUME_OFFSETS.items()}


def payg_windows(anchor: date) -> dict[str, Window]:
    return {p: _w(anchor, *o) for p, o in _PAYG_OFFSETS.items()}


def campaign_windows(run_date: date) -> dict[str, Window]:
    """BR-21 .. BR-24, relative to D."""
    return {
        "targeting": _w(run_date, *config.TARGETING_WINDOW),      # BR-21
        "arbitration": _w(run_date, *config.ARBITRATION_WINDOW),  # BR-22
        "reward": _w(run_date, *config.REWARD_WINDOW),            # BR-23
        "interactions": _w(run_date, *config.INTERACTION_WINDOW),  # BR-24
    }


def snapshot_date(run_date: date) -> date:
    """BR-01: base snapshot is ACTIVITY_DATE = D-2."""
    return run_date - timedelta(days=config.SNAPSHOT_LAG_DAYS)


def location_month(run_date: date) -> tuple[int, int]:
    """BR-13: previous calendar month of D-2 -> (month, year); January rolls back to December."""
    snap = snapshot_date(run_date)
    return (12, snap.year - 1) if snap.month == 1 else (snap.month - 1, snap.year)


def usage_history_start(run_date: date) -> date:
    """First day any usage / subscription window can need (D-96)."""
    return run_date - timedelta(days=config.HISTORY_DAYS_USAGE - 1)


def all_windows(run_date: date) -> dict:
    """Every window for D, for the manifest, the UI and the docs."""
    t = anchor_date(run_date)
    d = run_date
    return {
        "run_date": d.isoformat(),
        "anchor_date": t.isoformat(),
        "snapshot_date": snapshot_date(d).isoformat(),
        "location_month": dict(zip(("month", "year"), location_month(d))),
        "subs_amt_auto": {p: w.as_dict() for p, w in subs_windows(t).items()},
        "subs_amt_score": {p: w.as_dict() for p, w in subs_windows(d).items()},
        "activities_auto": {p: w.as_dict() for p, w in activity_windows(t).items()},
        "activities_score": {p: w.as_dict() for p, w in activity_windows(d).items()},
        "subscriptions_tg": {p: w.as_dict() for p, w in subscriptions_tg_windows(d).items()},
        "usage_tg_volume": {p: w.as_dict() for p, w in usage_tg_volume_windows(d).items()},
        "usage_tg_payg": {p: w.as_dict() for p, w in payg_windows(d).items()},
        "paygo_activity_auto": {p: w.as_dict() for p, w in payg_windows(t).items()},
        "campaign": {k: w.as_dict() for k, w in campaign_windows(d).items()},
    }
