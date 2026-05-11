"""CSV + executive text report export for FALCON."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from app.config import ARPU_MONTHLY_OMR, FTTH_UPTAKE_RATE, HOUSEHOLD_SIZE


def build_dataframe(scored_zones: list, zones_data: dict) -> pd.DataFrame:
    zone_meta = {z["id"]: z for z in zones_data["zones"]}
    rows = []
    for s in scored_zones:
        z = zone_meta.get(s["zone_id"], {})
        rows.append({
            "Zone ID": s["zone_id"],
            "Zone Name": s["zone_name"],
            "Priority Score": s["priority_score"],
            "Tier": s["tier"],
            "Vehicles Detected": s["vehicle_count"],
            "Density (veh/km²)": s["density_per_km2"],
            "Population": s["population"],
            "Area (km²)": s["area_km2"],
            "FTTH Coverage (%)": s["ftth_coverage_pct"],
            "4G Coverage (%)": z.get("current_4g_coverage_pct", 0),
            "Coverage Gap (%)": s["coverage_gap_pct"],
            "Commercial Density": s.get("commercial_density", ""),
            "Est. Investment (OMR)": s["recommended_investment_omr"],
            "ROI Payback (yrs)": s["roi_years"],
        })
    return pd.DataFrame(rows)


def export_csv(scored_zones: list, zones_data: dict) -> str:
    df = build_dataframe(scored_zones, zones_data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def generate_executive_summary(scored_zones: list, zones_data: dict) -> str:
    high = [z for z in scored_zones if z["tier"] == "High Priority"]
    med = [z for z in scored_zones if z["tier"] == "Medium Priority"]
    low = [z for z in scored_zones if z["tier"] == "Low Priority"]

    total_v = sum(z["vehicle_count"] for z in scored_zones)
    zones_meta = zones_data["zones"]
    avg_ftth = sum(z["ftth_coverage_pct"] for z in zones_meta) / len(zones_meta)
    avg_4g = sum(z["current_4g_coverage_pct"] for z in zones_meta) / len(zones_meta)
    total_pop = sum(z["population"] for z in zones_meta)
    total_inv_high = sum(z["recommended_investment_omr"] for z in high)
    total_inv_all = sum(z["recommended_investment_omr"] for z in scored_zones)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "=" * 62

    lines = [
        "",
        "  FALCON — FTTH EXPANSION PRIORITY REPORT",
        "  Muscat City — AI-Powered Zone Analysis",
        f"  Generated: {now}",
        sep,
        "",
        "EXECUTIVE SUMMARY",
        "─" * 40,
        f"Using YOLOv8-based vehicle density analysis across {len(scored_zones)}",
        "urban zones in Muscat, Oman, FALCON has identified priority",
        "areas for 4G/FTTH infrastructure expansion.",
        "",
        "KEY METRICS",
        "─" * 40,
        f"  Zones Analysed           : {len(scored_zones)}",
        f"  High Priority Zones      : {len(high)}",
        f"  Medium Priority Zones    : {len(med)}",
        f"  Low Priority Zones       : {len(low)}",
        f"  Total Vehicles Detected  : {total_v:,}",
        f"  Total Population Covered : {total_pop:,}",
        f"  Avg. FTTH Coverage       : {avg_ftth:.1f}%",
        f"  Avg. 4G Coverage         : {avg_4g:.1f}%",
        f"  City-wide FTTH Gap       : {100 - avg_ftth:.1f}%",
        "",
        "TOP 5 PRIORITY ZONES",
        "─" * 40,
    ]

    top5 = scored_zones[:5]
    for i, z in enumerate(top5, 1):
        lines += [
            f"  {i}. {z['zone_name']}",
            f"     Priority Score    : {z['priority_score']:.1f}/100",
            f"     Tier              : {z['tier']}",
            f"     Vehicles Detected : {z['vehicle_count']:,}",
            f"     FTTH Gap          : {z['coverage_gap_pct']:.0f}%",
            f"     Population        : {z['population']:,}",
            f"     Est. Investment   : {z['recommended_investment_omr']:,} OMR",
            f"     ROI Payback       : {z['roi_years']:.1f} years",
            "",
        ]

    lines += [
        "INVESTMENT SUMMARY",
        "─" * 40,
        f"  High Priority Zone Investment : {total_inv_high:,} OMR",
        f"  Full City FTTH Roll-out       : {total_inv_all:,} OMR",
        "",
        "FINANCIAL PROJECTIONS (5-year, High Priority Zones)",
        "─" * 40,
    ]

    for z in high:
        annual_rev = (z["population"] / HOUSEHOLD_SIZE) * FTTH_UPTAKE_RATE * ARPU_MONTHLY_OMR * 12
        rev_5yr = annual_rev * 5
        lines.append(
            f"  {z['zone_name']:<28} "
            f"5yr Revenue: {rev_5yr:,.0f} OMR")

    lines += [
        "",
        "METHODOLOGY",
        "─" * 40,
        "  Detection Model  : YOLOv8 (COCO, vehicle classes 2/3/5/7)",
        "  Scoring Weights  : Density 40% | Coverage Gap 35% | Population 25%",
        "  Image Resolution : 640 × 640 px synthetic aerial imagery",
        f"  Analysis Date    : {now}",
        "  Tool             : FALCON v1.0 — Flavia AI Coverage &",
        "                     Location Opportunity Network",
        "",
        sep,
    ]

    return "\n".join(lines)
