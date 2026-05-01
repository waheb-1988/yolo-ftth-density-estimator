"""
FALCON build_demo.py — full pipeline: generate networks, detect cars,
score zones, build map, export reports, launch Streamlit.

Usage:  python scripts/build_demo.py
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

# ── Make stdout UTF-8 safe on Windows ────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
from app.config import (DATA_DIR, OUTPUTS_DIR, WAHEB_MARKET_SHARE,
                         RAMI_MARKET_SHARE, DON_MARKET_SHARE,
                         UNSERVED_MARKET_SHARE, TOTAL_MUSCAT_HOMES)
from app.data_loader import initialize_demo_data, load_zones, load_network_data
from app.network_generator import NetworkGenerator
from app.map_builder import save_competitive_map
from app.report_exporter import export_csv, generate_executive_summary


def _step(n: int, total: int, msg: str):
    print(f"  [{n}/{total}] {msg} ...", flush=True)


def generate_executive_summary_competitive(scored_zones: list,
                                            zones_data: dict,
                                            network_data: dict) -> str:
    ms = network_data.get("market_summary", {})
    opp = ms.get("opportunity_zone_details", [])
    top3 = opp[:3]

    additional_homes = int(TOTAL_MUSCAT_HOMES * 0.08)   # 10% -> 18%
    new_subs = int(additional_homes * 0.42)
    annual_rev = int(new_subs * 5.5 * 12)
    investment = int((additional_homes // 150) * 25_000)

    threat_zone = "Ruwi CBD"
    rami_cab = ms.get("cabinet_counts", {}).get("RAMI", {})
    if rami_cab:
        threat_zone = max(rami_cab, key=rami_cab.get)

    lines = [
        "",
        "FALCON COMPETITIVE ANALYSIS -- WAHEB TELECOM",
        "=" * 52,
        "",
        f"Current Position: {WAHEB_MARKET_SHARE:.0f}% market share (3rd place)",
        f"Total Muscat Market: {TOTAL_MUSCAT_HOMES:,} homes",
        "",
        "MARKET POSITIONS:",
        f"  Waheb    {WAHEB_MARKET_SHARE:>5.0f}%  (our company)",
        f"  RAMI     {RAMI_MARKET_SHARE:>5.0f}%  (dominant — Ruwi/Wattayah)",
        f"  Don      {DON_MARKET_SHARE:>5.0f}%  (growing — Seeb/Azaiba)",
        f"  Unserved {UNSERVED_MARKET_SHARE:>5.0f}%  (= {int(TOTAL_MUSCAT_HOMES*UNSERVED_MARKET_SHARE/100):,} homes = our opportunity)",
        "",
        "TOP 3 IMMEDIATE OPPORTUNITIES:",
    ]
    for i, o in enumerate(top3, 1):
        lines += [
            f"  {i}. {o['zone']}",
            f"     Competitor cabinets: {o['competitor_cabinets']}  "
            f"| Waheb cabinets: {o['waheb_cabinets']}",
            f"     Est. annual revenue: {o['est_annual_revenue_omr']:,} OMR",
            "",
        ]

    lines += [
        "GROWTH SCENARIO: Deploy 30 new cabinets in opportunity zones",
        "-" * 52,
        f"  Projected share: {WAHEB_MARKET_SHARE:.0f}% -> 18% within 12 months",
        f"  Additional homes connected: {additional_homes:,}",
        f"  New subscribers:            {new_subs:,}",
        f"  Additional annual revenue:  {annual_rev:,} OMR",
        f"  Investment required:        {investment:,} OMR",
        f"  Payback period:             {int(investment/(annual_rev/12))} months",
        "",
        f"THREAT ALERT: RAMI expanding in {threat_zone}.",
        "  Recommended action: Deploy Waheb cabinet within 60 days.",
        "",
        "=" * 52,
    ]
    return "\n".join(lines)


def main():
    TOTAL = 10
    print()
    print("=" * 58)
    print("  FALCON Full Demo Pipeline")
    print("  Muscat Telecom Competitive Intelligence")
    print("=" * 58)
    print()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1-4: Generate network data
    _step(1, TOTAL, "Generating Waheb FTTH network (45 cab | 12 exch)")
    _step(2, TOTAL, "Generating RAMI network (95 cab | 28 exch)")
    _step(3, TOTAL, "Generating Don network (70 cab | 18 exch)")
    _step(4, TOTAL, "Computing market_summary.json")
    zones_data = load_zones()
    gen = NetworkGenerator(zones_data, seed=42)
    network = gen.save_all(DATA_DIR)
    waheb = network["waheb"]
    rami  = network["rami"]
    don   = network["don"]
    ms    = network["market_summary"]

    def _count(assets, typ):
        return sum(1 for a in assets if a["type"] == typ)

    print(f"      Waheb:  {_count(waheb,'cabinet')} cab, "
          f"{_count(waheb,'exchange')} exch, "
          f"{_count(waheb,'home')} homes")
    print(f"      RAMI:   {_count(rami,'cabinet')} cab, "
          f"{_count(rami,'exchange')} exch, "
          f"{_count(rami,'home')} homes")
    print(f"      Don:    {_count(don,'cabinet')} cab, "
          f"{_count(don,'exchange')} exch, "
          f"{_count(don,'home')} homes")

    # Step 5-6: YOLO detection + zone scoring
    _step(5, TOTAL, "Running FALCON zone scoring (aerial imagery + YOLO pipeline)")
    _step(6, TOTAL, "Generating zone priority scores")
    results = initialize_demo_data(force=False)
    scored = results["scored_zones"]
    top3 = scored[:3]
    for i, z in enumerate(top3, 1):
        print(f"      #{i}: {z['zone_name']}  score={z['priority_score']:.1f}  "
              f"tier={z['tier']}")

    # Step 7: Build competitive map HTML
    _step(7, TOTAL, "Building competitive Folium map -> outputs/falcon_competitive_map.html")
    network_data_loaded = load_network_data()
    map_path = save_competitive_map(
        zones_data, scored, network_data_loaded,
        OUTPUTS_DIR / "falcon_competitive_map.html")
    print(f"      Saved: {map_path}")

    # Step 8: Export CSV
    _step(8, TOTAL, "Exporting FALCON_Report.csv")
    csv_data = export_csv(scored, zones_data)
    csv_path = OUTPUTS_DIR / "FALCON_Report.csv"
    csv_path.write_text(csv_data, encoding="utf-8")
    print(f"      Saved: {csv_path}")

    # Step 9: Executive summary
    _step(9, TOTAL, "Writing FALCON_Executive_Summary.txt")
    summary = generate_executive_summary_competitive(scored, zones_data, network)
    txt_path = OUTPUTS_DIR / "FALCON_Executive_Summary.txt"
    txt_path.write_text(summary, encoding="utf-8")
    print(f"      Saved: {txt_path}")
    print()
    print("  SUMMARY PREVIEW:")
    for line in summary.split("\n")[1:12]:
        print(f"    {line}")
    print("    ...")

    # Step 10: Launch Streamlit
    _step(10, TOTAL, "Launching FALCON dashboard")
    unserved = int(TOTAL_MUSCAT_HOMES * UNSERVED_MARKET_SHARE / 100)
    print()
    print("=" * 58)
    print("  FALCON DEMO READY")
    print("  " + "-" * 54)
    print(f"  Waheb Network:  {_count(waheb,'cabinet')} cabinets | "
          f"{WAHEB_MARKET_SHARE:.0f}% share")
    print(f"  RAMI  Network:  {_count(rami,'cabinet')} cabinets | "
          f"{RAMI_MARKET_SHARE:.0f}% share")
    print(f"  Don   Network:  {_count(don,'cabinet')} cabinets | "
          f"{DON_MARKET_SHARE:.0f}% share")
    print(f"  Opportunity:    {unserved:,} unserved homes")
    print("  " + "-" * 54)
    print("  Dashboard: http://localhost:8501")
    print(f"  Map HTML:  {map_path}")
    print("=" * 58)
    print()

    import os
    os.environ["FALCON_DEMO_MODE"] = "1"
    dashboard_path = ROOT / "app" / "dashboard.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#02C39A",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f0f2f6",
        "--theme.textColor", "#1a1a2e",
    ]
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n[FALCON] Stopped.")


if __name__ == "__main__":
    main()
