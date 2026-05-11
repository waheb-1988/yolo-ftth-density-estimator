"""FALCON — Streamlit dashboard (6 tabs: Map | Rankings | Detection | Business Case | Competitive Intel | AI Insights)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on path when run via `streamlit run app/dashboard.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.logger import setup_logging
from app.config import LOG_LEVEL, LOG_TO_FILE, LOGS_DIR
setup_logging(level=LOG_LEVEL, log_to_file=LOG_TO_FILE, log_dir=LOGS_DIR)
import logging
_log = logging.getLogger(__name__)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from app.config import (APP_ICON, APP_SUBTITLE, APP_TITLE, TIER_COLORS,
                         WAHEB_COLOR, RAMI_COLOR, DON_COLOR,
                         WAHEB_MARKET_SHARE, RAMI_MARKET_SHARE,
                         DON_MARKET_SHARE, UNSERVED_MARKET_SHARE,
                         TOTAL_MUSCAT_HOMES, WAHEB_TARGET_SHARE_2026,
                         COST_PER_CABINET_OMR, HOMES_PER_CABINET,
                         ARPU_MONTHLY_OMR, HOUSEHOLD_SIZE, FTTH_UPTAKE_RATE)
from app.data_loader import load_network_data, load_or_generate_results, load_zones
from app.map_builder import build_map
from app.report_exporter import export_csv, generate_executive_summary

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title=f"{APP_TITLE} | FTTH Planner",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_MODE = os.environ.get("FALCON_DEMO_MODE", "1") == "1"

CUSTOM_CSS = """
<style>
  /* Header */
  .falcon-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: white; padding: 20px 28px; border-radius: 12px; margin-bottom: 18px;
  }
  .falcon-header h1 { margin: 0; font-size: 2.2rem; letter-spacing: 2px; }
  .falcon-header p  { margin: 4px 0 0; opacity: 0.75; font-size: 0.9rem; }
  .mode-badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.75rem; font-weight:600; margin-top:8px;
  }
  .demo-badge  { background:#e94560; color:white; }
  .prod-badge  { background:#28a745; color:white; }

  /* Metric cards */
  .metric-row { display:flex; gap:14px; margin-bottom:16px; flex-wrap:wrap; }
  .mcard {
    flex:1; min-width:140px; background:#f8f9fa; border-radius:10px;
    padding:14px 18px; border-left:4px solid #1a1a2e;
  }
  .mcard .val { font-size:1.7rem; font-weight:700; color:#1a1a2e; }
  .mcard .lbl { font-size:0.78rem; color:#666; margin-top:2px; }

  /* Priority tier badges */
  .high-p  { color:#dc3545; font-weight:700; }
  .med-p   { color:#fd7e14; font-weight:700; }
  .low-p   { color:#28a745; font-weight:700; }

  /* Image grid caption */
  .img-caption {
    text-align:center; font-size:0.80rem; font-weight:600;
    margin-top:4px; color:#333;
  }
</style>
"""

# ── Cached data loaders ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading zone data …")
def _load_zones():
    return load_zones()


@st.cache_data(show_spinner="Initialising FALCON analysis …")
def _load_results(demo: bool):
    zones_data = _load_zones()
    return load_or_generate_results(zones_data, demo_mode=demo)


@st.cache_data(show_spinner="Loading competitive network data …", ttl=300)
def _load_network_data():
    return load_network_data()


@st.cache_data(show_spinner="Running telecom simulation …", ttl=600)
def _load_telecom_sim(zones_json_str: str):
    """Cached telecom simulation — re-runs only when zones data changes."""
    import json
    from app.telecom_simulator import TelecomSimulator
    zones_data = json.loads(zones_json_str)
    sim = TelecomSimulator()
    return sim.simulate_all(zones_data)


@st.cache_data(show_spinner="Running zone clustering …", ttl=600)
def _load_clusters(scored_tuple, zones_json_str: str, n_clusters: int):
    import json
    from app.forecaster import cluster_zones
    scored = [dict(s) if not isinstance(s, dict) else s for s in scored_tuple]
    zones_data = json.loads(zones_json_str)
    return cluster_zones(scored, zones_data, n_clusters=n_clusters)


@st.cache_data(show_spinner=False)
def _build_df(scored_zones_tuple):
    # Each element is a tuple of (key, value) pairs — convert back to dict
    rows = []
    for s_raw in scored_zones_tuple:
        s = dict(s_raw) if isinstance(s_raw, tuple) else s_raw
        rows.append({
            "Zone": s["zone_name"],
            "Score": s["priority_score"],
            "Tier": s["tier"],
            "Vehicles": s["vehicle_count"],
            "Density": s["density_per_km2"],
            "FTTH %": s["ftth_coverage_pct"],
            "Gap %": s["coverage_gap_pct"],
            "Population": s["population"],
            "Investment (OMR)": s["recommended_investment_omr"],
            "ROI (yrs)": s["roi_years"],
        })
    return pd.DataFrame(rows)


# ── Sidebar ─────────────────────────────────────────────────────────────────────

def render_sidebar(results: dict, zones_data: dict):
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 6px;">
      <span style="font-size:2.5rem;">🦅</span><br>
      <span style="font-size:1.1rem;font-weight:700;color:#1a1a2e;">FALCON</span><br>
      <span style="font-size:0.72rem;color:#888;">FTTH Coverage Optimizer</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    mode_label = "DEMO MODE" if DEMO_MODE else "PRODUCTION"
    badge_cls = "demo-badge" if DEMO_MODE else "prod-badge"
    st.markdown(
        f'<span class="mode-badge {badge_cls}">{mode_label}</span>',
        unsafe_allow_html=True)
    st.caption("Run `python run.py --demo` to regenerate data.")
    st.divider()

    scored = results["scored_zones"]
    all_tiers = ["All"] + sorted({s["tier"] for s in scored})
    sel_tier = st.selectbox("Filter by Tier", all_tiers, key="tier_filter")
    filtered = scored if sel_tier == "All" else [
        s for s in scored if s["tier"] == sel_tier]
    st.caption(f"{len(filtered)} zone(s) selected")
    st.session_state["filtered_zones"] = filtered

    st.divider()
    st.subheader("📥 Export Reports")

    csv_data = export_csv(scored, zones_data)
    st.download_button(
        label="⬇ Download CSV Report",
        data=csv_data,
        file_name="falcon_muscat_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

    summary = generate_executive_summary(scored, zones_data)
    st.download_button(
        label="⬇ Executive Summary (.txt)",
        data=summary,
        file_name="falcon_executive_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()
    st.caption(
        "**FALCON v1.0** · Flavia AI Coverage & Location\n"
        "Opportunity Network · Muscat, Oman · 2026")


# ── Tab 1 — Map ─────────────────────────────────────────────────────────────────

def render_map_tab(zones_data: dict, results: dict):
    scored = results["scored_zones"]
    filtered = st.session_state.get("filtered_zones", scored)

    high_n = sum(1 for s in scored if s["tier"] == "High Priority")
    med_n = sum(1 for s in scored if s["tier"] == "Medium Priority")
    low_n = sum(1 for s in scored if s["tier"] == "Low Priority")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Zones", len(scored))
    c2.metric("🔴 High Priority", high_n)
    c3.metric("🟠 Medium Priority", med_n)
    c4.metric("🟢 Low Priority", low_n)

    st.markdown("---")
    network_data = _load_network_data()
    if network_data:
        st.subheader("Competitive Network Map — Muscat City")
        st.caption(
            "Toggle layers via control (top-right). "
            "Green=Waheb | Blue=RAMI | Red=Don | Gold=Opportunity zones.")
    else:
        st.subheader("Interactive Priority Map — Muscat City")
        st.caption(
            "Run `python scripts/build_demo.py` to add competitive network layers.")

    try:
        from streamlit_folium import st_folium
        folium_map = build_map(zones_data, scored, network_data)
        st_folium(folium_map, height=580, use_container_width=True,
                  returned_objects=[])
    except ImportError:
        st.warning(
            "streamlit-folium not installed. "
            "Run: pip install streamlit-folium")
    except Exception as e:
        st.error(f"Map rendering error: {e}")

    with st.expander("Zone coordinate table"):
        rows = [{"Zone": z["name"], "Lat": z["lat"], "Lon": z["lon"],
                 "Score": next((s["priority_score"] for s in scored
                                if s["zone_id"] == z["id"]), 0)}
                for z in zones_data["zones"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)


# ── Tab 2 — Rankings ────────────────────────────────────────────────────────────

def render_rankings_tab(results: dict, zones_data: dict):
    scored = results["scored_zones"]
    filtered = st.session_state.get("filtered_zones", scored)

    # Top-3 hero cards
    top3 = sorted(scored, key=lambda x: x["priority_score"], reverse=True)[:3]
    medal = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for col, m, z in zip(cols, medal, top3):
        tc = TIER_COLORS.get(z["tier"], "#555")
        col.markdown(f"""
        <div style="background:#f8f9fa;border-radius:12px;padding:16px 18px;
                    border-left:5px solid {tc};">
          <div style="font-size:1.6rem;">{m}</div>
          <div style="font-weight:700;font-size:1.05rem;margin-top:4px;">
            {z['zone_name']}</div>
          <div style="font-size:2rem;font-weight:800;color:{tc};">
            {z['priority_score']:.1f}</div>
          <div style="font-size:0.78rem;color:#888;">/ 100 priority score</div>
          <div style="margin-top:8px;font-size:0.82rem;">
            🚗 {z['vehicle_count']} vehicles &nbsp;|&nbsp;
            📡 {z['coverage_gap_pct']:.0f}% gap
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Zone Priority Rankings")
        df = _build_df(tuple(
            tuple(s.items()) for s in filtered))
        df.insert(0, "Rank", range(1, len(df) + 1))

        def tier_color(tier):
            if tier == "High Priority":
                return "color: #dc3545; font-weight:bold"
            if tier == "Medium Priority":
                return "color: #fd7e14; font-weight:bold"
            return "color: #28a745; font-weight:bold"

        st.dataframe(
            df[["Rank", "Zone", "Score", "Tier", "Vehicles",
                "FTTH %", "Gap %", "Population", "Investment (OMR)"]],
            use_container_width=True,
            height=420,
            hide_index=True,
        )

    with col_right:
        st.subheader("Score Distribution")
        pie_df = pd.DataFrame({
            "Tier": ["High Priority", "Medium Priority", "Low Priority"],
            "Count": [
                sum(1 for s in filtered if s["tier"] == "High Priority"),
                sum(1 for s in filtered if s["tier"] == "Medium Priority"),
                sum(1 for s in filtered if s["tier"] == "Low Priority"),
            ]
        })
        fig_pie = px.pie(
            pie_df, values="Count", names="Tier",
            color="Tier",
            color_discrete_map={
                "High Priority": "#dc3545",
                "Medium Priority": "#fd7e14",
                "Low Priority": "#28a745",
            },
            hole=0.45,
        )
        fig_pie.update_layout(margin=dict(t=10, b=10), height=280,
                               showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Priority Scores — All Zones")
    bar_df = pd.DataFrame([
        {"Zone": s["zone_name"], "Score": s["priority_score"], "Tier": s["tier"]}
        for s in sorted(filtered, key=lambda x: x["priority_score"], reverse=True)
    ])
    fig_bar = px.bar(
        bar_df, x="Zone", y="Score", color="Tier",
        color_discrete_map={
            "High Priority": "#dc3545",
            "Medium Priority": "#fd7e14",
            "Low Priority": "#28a745",
        },
        text="Score",
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_bar.update_layout(
        xaxis_tickangle=-35, yaxis_range=[0, 105],
        margin=dict(t=20, b=80), height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 3 — Detection ───────────────────────────────────────────────────────────

def render_detection_tab(results: dict, zones_data: dict):
    scored = results["scored_zones"]
    img_results = results.get("image_results", {})
    zone_meta = {z["id"]: z for z in zones_data["zones"]}

    st.subheader("Vehicle Detection — Synthetic Aerial Imagery")
    st.caption(
        "640 × 640 synthetic aerial images generated per zone. "
        "Green boxes = detected vehicles. "
        "In production mode, replace with real satellite imagery.")

    col_a, col_b, col_c = st.columns(3)
    total_v = sum(d["vehicle_count"] for d in img_results.values())
    avg_v = total_v / max(len(img_results), 1)
    top_zone = max(img_results.items(), key=lambda x: x[1]["vehicle_count"],
                   default=(None, {"vehicle_count": 0}))
    col_a.metric("Total Vehicles", f"{total_v:,}")
    col_b.metric("Avg Vehicles / Zone", f"{avg_v:.1f}")
    if top_zone[0]:
        col_c.metric("Busiest Zone",
                     zone_meta.get(top_zone[0], {}).get("name", top_zone[0]),
                     f"{top_zone[1]['vehicle_count']} vehicles")

    st.markdown("---")

    # Zone image grid — 3 columns
    sorted_by_count = sorted(
        img_results.items(), key=lambda x: x[1]["vehicle_count"], reverse=True)
    cols_per_row = 3
    rows = [sorted_by_count[i:i + cols_per_row]
            for i in range(0, len(sorted_by_count), cols_per_row)]

    for row in rows:
        grid_cols = st.columns(cols_per_row)
        for col, (zid, idata) in zip(grid_cols, row):
            ann_path = idata.get("annotated_path", "")
            img_path = idata.get("image_path", "")
            zname = zone_meta.get(zid, {}).get("name", zid)
            vcount = idata["vehicle_count"]
            tier = next((s["tier"] for s in scored if s["zone_id"] == zid), "")
            tc = TIER_COLORS.get(tier, "#555")

            if Path(ann_path).exists():
                disp_path = ann_path
            elif Path(img_path).exists():
                disp_path = img_path
            else:
                col.warning(f"No image for {zname}")
                continue

            img = Image.open(disp_path)
            col.image(img, use_container_width=True)
            col.markdown(
                f"<div class='img-caption'>"
                f"<span style='color:{tc};'>●</span> "
                f"<b>{zname}</b> — {vcount} vehicles"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("🔍 Test YOLOv8 on Your Own Image")

    aerial_mode = st.checkbox(
        "Aerial / satellite mode (Google Earth, drone imagery)",
        value=True,
        help="Upscales 4× and uses sliding-window tiling — required for "
             "satellite images where vehicles are only 5–10 px wide.",
        key="aerial_mode_checkbox",
    )

    uploaded = st.file_uploader(
        "Upload a satellite or street-level image",
        type=["jpg", "jpeg", "png"],
        key="yolo_upload",
    )
    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        col1.image(img_pil, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Running YOLOv8 detection …"):
            try:
                from app.detector import get_detector
                from app.image_generator import draw_detections_on_image

                det = get_detector("n")
                if det.available:
                    if aerial_mode:
                        res = det.detect_aerial(img_pil)
                        mode_label = "YOLO aerial (4× upscale + tiling)"
                    else:
                        res = det.detect_image_object(img_pil, conf=0.25)
                        mode_label = "YOLO standard"

                    annotated = draw_detections_on_image(img_pil, res.boxes,
                                                          box_color=(0, 230, 0))
                    col2.image(annotated,
                               caption=f"Detected {res.count} vehicles ({mode_label})",
                               use_container_width=True)
                    st.success(
                        f"YOLOv8 detected **{res.count}** vehicles "
                        f"using **{mode_label}**.")
                else:
                    st.warning(
                        "YOLOv8 model not available. "
                        "Ensure `ultralytics` is installed and internet is available.")
                    col2.image(img_pil, caption="No detection (model unavailable)",
                               use_container_width=True)
            except Exception as e:
                st.error(f"Detection error: {e}")


# ── Tab 4 — Business Case ───────────────────────────────────────────────────────

def render_business_case_tab(results: dict, zones_data: dict):
    scored = results["scored_zones"]
    filtered = st.session_state.get("filtered_zones", scored)
    zones_meta = {z["id"]: z for z in zones_data["zones"]}

    high = [s for s in scored if s["tier"] == "High Priority"]
    total_inv = sum(s["recommended_investment_omr"] for s in filtered)
    total_inv_high = sum(s["recommended_investment_omr"] for s in high)
    avg_roi = (sum(s["roi_years"] for s in high) / max(len(high), 1))

    # Hero KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Est. Investment", f"{total_inv:,.0f} OMR",
              delta=f"{len(filtered)} zones",
              delta_color="off")
    k2.metric("High-Priority Investment", f"{total_inv_high:,.0f} OMR",
              delta=f"{len(high)} zones",
              delta_color="off")
    k3.metric("Avg. ROI Payback", f"{avg_roi:.1f} yrs",
              delta="High-priority zones",
              delta_color="off")
    pop_unserved = sum(
        int(s["population"] * s["coverage_gap_pct"] / 100) for s in filtered)
    k4.metric("Unserved Population", f"{pop_unserved:,}",
              delta="Potential new subscribers",
              delta_color="off")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Priority Score vs FTTH Coverage Gap")
        df_scatter = pd.DataFrame([
            {
                "Zone": s["zone_name"],
                "Priority Score": s["priority_score"],
                "Coverage Gap (%)": s["coverage_gap_pct"],
                "Population": s["population"],
                "Tier": s["tier"],
            }
            for s in filtered
        ])
        fig_scatter = px.scatter(
            df_scatter,
            x="Coverage Gap (%)",
            y="Priority Score",
            size="Population",
            color="Tier",
            hover_name="Zone",
            color_discrete_map={
                "High Priority": "#dc3545",
                "Medium Priority": "#fd7e14",
                "Low Priority": "#28a745",
            },
            size_max=35,
        )
        fig_scatter.update_layout(height=350, margin=dict(t=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with right:
        st.subheader("Estimated Investment by Zone")
        inv_df = pd.DataFrame([
            {
                "Zone": s["zone_name"],
                "Investment (OMR)": s["recommended_investment_omr"],
                "Tier": s["tier"],
            }
            for s in sorted(filtered,
                             key=lambda x: x["recommended_investment_omr"],
                             reverse=True)
        ])
        fig_inv = px.bar(
            inv_df, x="Investment (OMR)", y="Zone",
            color="Tier", orientation="h",
            color_discrete_map={
                "High Priority": "#dc3545",
                "Medium Priority": "#fd7e14",
                "Low Priority": "#28a745",
            },
        )
        fig_inv.update_layout(height=350, margin=dict(t=20, l=10),
                               yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_inv, use_container_width=True)

    st.markdown("---")
    st.subheader("5-Year Projected Revenue — High Priority Zones")

    if high:
        years = list(range(1, 6))
        fig_roi = go.Figure()
        for z in high:
            annual_rev = (z["population"] / HOUSEHOLD_SIZE) * FTTH_UPTAKE_RATE * ARPU_MONTHLY_OMR * 12
            cum_rev = [annual_rev * y for y in years]
            inv_line = [z["recommended_investment_omr"]] * 5
            fig_roi.add_trace(go.Scatter(
                x=years, y=cum_rev, mode="lines+markers",
                name=z["zone_name"], line=dict(width=2)))
        fig_roi.update_layout(
            xaxis_title="Year", yaxis_title="Cumulative Revenue (OMR)",
            height=340, margin=dict(t=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
        )
        st.plotly_chart(fig_roi, use_container_width=True)
    else:
        st.info("No high-priority zones in current filter selection.")

    st.markdown("---")
    st.subheader("Full Business Case Table")

    bc_rows = []
    for s in sorted(filtered, key=lambda x: x["priority_score"], reverse=True):
        annual_rev = (s["population"] / HOUSEHOLD_SIZE) * FTTH_UPTAKE_RATE * ARPU_MONTHLY_OMR * 12
        bc_rows.append({
            "Zone": s["zone_name"],
            "Tier": s["tier"],
            "Score": s["priority_score"],
            "Investment (OMR)": f"{s['recommended_investment_omr']:,}",
            "Annual Revenue (OMR)": f"{annual_rev:,.0f}",
            "5yr Revenue (OMR)": f"{annual_rev * 5:,.0f}",
            "ROI Payback (yrs)": s["roi_years"],
            "Coverage Gap": f"{s['coverage_gap_pct']:.0f}%",
        })
    st.dataframe(pd.DataFrame(bc_rows), use_container_width=True,
                 height=380, hide_index=True)


# ── Tab 5 — Competitive Intel ────────────────────────────────────────────────────

def _zone_shares(zname: str, intensity: str) -> dict:
    import random as _r
    rng = _r.Random(abs(hash(zname)) % 99991)
    if intensity == "high":
        rami  = rng.uniform(28, 44)
        don   = rng.uniform(16, 28)
        waheb = rng.uniform(5, 12)
    elif intensity == "medium":
        rami  = rng.uniform(18, 33)
        don   = rng.uniform(10, 22)
        waheb = rng.uniform(6, 14)
    else:
        rami  = rng.uniform(8, 20)
        don   = rng.uniform(6, 16)
        waheb = rng.uniform(4, 10)
    total = rami + don + waheb
    if total > 72:
        f = 72 / total
        rami, don, waheb = rami * f, don * f, waheb * f
    unserved = 100 - rami - don - waheb
    return {"Waheb": round(waheb, 1), "RAMI": round(rami, 1),
            "Don": round(don, 1), "Unserved": round(max(0, unserved), 1)}


def render_competitive_tab(network_data: dict | None, zones_data: dict):
    COMPETITIVE_INTENSITY_FALLBACK = {
        "Ruwi CBD": "high", "Mutrah Port": "high",
        "Qurum Heights": "high", "Al Khuwair": "high",
        "Shatti Al Qurum": "medium", "Al Azaiba": "medium",
        "Madinat Sultan Qaboos": "medium", "Al Wuttayyah": "medium",
        "Al Seeb City": "medium", "Ghubra": "low", "Bausher": "low",
        "Wadi Kabir": "low", "Sidab": "low",
        "Al Amerat": "low", "Bowsher Hills": "low",
    }

    if network_data is None:
        st.info(
            "No competitive data found. "
            "Run `python scripts/build_demo.py` to generate network data.")
        return

    ms = network_data.get("market_summary", {})
    intensity_map = ms.get("competitive_intensity", COMPETITIVE_INTENSITY_FALLBACK)
    cab_counts = ms.get("cabinet_counts", {})
    opp_details = {o["zone"]: o for o in
                   ms.get("opportunity_zone_details", [])}
    zones = zones_data["zones"]

    # ── Row 1: KPI cards ──────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    unserved_homes = int(TOTAL_MUSCAT_HOMES * UNSERVED_MARKET_SHARE / 100)

    def _kpi(col, label, value_str, subtitle, color):
        col.markdown(f"""
        <div style="background:#f8f9fa;border-radius:10px;padding:16px;
                    border-left:5px solid {color};text-align:center;">
          <div style="font-size:2rem;font-weight:800;color:{color};">
            {value_str}</div>
          <div style="font-weight:600;font-size:0.9rem;margin-top:2px;">
            {label}</div>
          <div style="font-size:0.75rem;color:#888;margin-top:4px;">
            {subtitle}</div>
        </div>""", unsafe_allow_html=True)

    _kpi(k1, "Waheb Market Share",
         f"{WAHEB_MARKET_SHARE:.0f}%", "Target: 25% by 2026", WAHEB_COLOR)
    _kpi(k2, "RAMI Market Share",
         f"{RAMI_MARKET_SHARE:.0f}%", "Dominant competitor", RAMI_COLOR)
    _kpi(k3, "Don Market Share",
         f"{DON_MARKET_SHARE:.0f}%", "Growing in Seeb/Azaiba", DON_COLOR)
    _kpi(k4, "Unserved Market",
         f"{UNSERVED_MARKET_SHARE:.0f}%",
         f"~{unserved_homes:,} homes available", "#FFA500")

    st.markdown("---")

    # ── Row 2: charts ─────────────────────────────────────────────────────────
    left, right = st.columns(2)

    # Stacked bar chart — market share by zone (top 10 by population)
    top10 = sorted(zones, key=lambda z: z["population"], reverse=True)[:10]
    bar_rows = []
    for z in top10:
        zn = z["name"]
        sh = _zone_shares(zn, intensity_map.get(zn, "medium"))
        for company, share in sh.items():
            bar_rows.append({"Zone": zn, "Company": company, "Share (%)": share})
    bar_df = pd.DataFrame(bar_rows)

    with left:
        st.subheader("Market Share by Zone (Top 10)")
        fig_bar = px.bar(
            bar_df, x="Zone", y="Share (%)", color="Company",
            color_discrete_map={
                "Waheb": WAHEB_COLOR, "RAMI": RAMI_COLOR,
                "Don": DON_COLOR, "Unserved": "#cccccc"},
            barmode="stack",
        )
        fig_bar.update_layout(
            height=340, margin=dict(t=10, b=80),
            xaxis_tickangle=-30,
            legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Opportunity ranking — horizontal bars
    opp_rows = []
    for z in zones:
        zn = z["name"]
        wc = cab_counts.get("Waheb", {}).get(zn, 0)
        rc = cab_counts.get("RAMI", {}).get(zn, 0)
        dc = cab_counts.get("Don", {}).get(zn, 0)
        opp_score = max(0, 12 - rc - dc - wc)
        opp_rows.append({"Zone": zn, "Opportunity Score": opp_score,
                          "Waheb": wc, "RAMI": rc, "Don": dc})
    opp_df = pd.DataFrame(opp_rows).sort_values(
        "Opportunity Score", ascending=True).tail(10)

    with right:
        st.subheader("Waheb Opportunity Score by Zone")
        fig_opp = px.bar(
            opp_df, x="Opportunity Score", y="Zone",
            orientation="h",
            color="Opportunity Score",
            color_continuous_scale=[[0, "#d4f5e9"], [1, WAHEB_COLOR]],
            hover_data=["Waheb", "RAMI", "Don"],
        )
        fig_opp.update_layout(height=340, margin=dict(t=10, l=10),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_opp, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Competitive gap table ──────────────────────────────────────────
    st.subheader("Competitive Gap Analysis")

    def _action(wc):
        if wc == 0:
            return "URGENT — No Waheb presence"
        if wc < 3:
            return "EXPAND — Low coverage"
        return "DEFEND — Hold position"

    def _action_color(action):
        if "URGENT" in action:
            return "#dc3545"
        if "EXPAND" in action:
            return "#fd7e14"
        return "#28a745"

    gap_rows = []
    for z in zones:
        zn = z["name"]
        wc = cab_counts.get("Waheb", {}).get(zn, 0)
        rc = cab_counts.get("RAMI", {}).get(zn, 0)
        dc = cab_counts.get("Don", {}).get(zn, 0)
        leader_n = max(rc, dc, 1)
        sh = _zone_shares(zn, intensity_map.get(zn, "medium"))
        action = _action(wc)
        gap_rows.append({
            "Zone":            zn,
            "Waheb Cab.":      wc,
            "RAMI Cab.":       rc,
            "Don Cab.":        dc,
            "Waheb Share %":   sh["Waheb"],
            "Gap to Leader":   f"{max(rc,dc) - wc:+d}",
            "Action":          action,
        })

    # Sort: URGENT first, EXPAND second, DEFEND last
    order = {"URGENT — No Waheb presence": 0,
             "EXPAND — Low coverage": 1,
             "DEFEND — Hold position": 2}
    gap_rows.sort(key=lambda r: order.get(r["Action"], 9))
    gap_df = pd.DataFrame(gap_rows)
    st.dataframe(gap_df, use_container_width=True, height=380, hide_index=True)

    st.markdown("---")

    # ── Row 4: Revenue opportunity calculator ─────────────────────────────────
    st.subheader("Revenue Opportunity Calculator")
    target_share = st.slider(
        "Target Market Share (%)", min_value=10, max_value=40,
        value=20, step=1, key="target_share_slider")

    additional_share_pct = target_share - WAHEB_MARKET_SHARE
    additional_homes     = int(TOTAL_MUSCAT_HOMES * additional_share_pct / 100)
    new_cabinets         = max(0, int(additional_homes / HOMES_PER_CABINET))
    investment_omr       = new_cabinets * COST_PER_CABINET_OMR
    new_subs             = additional_homes * FTTH_UPTAKE_RATE
    annual_revenue_omr   = int(new_subs * ARPU_MONTHLY_OMR * 12)
    payback_months       = (int(investment_omr / max(annual_revenue_omr / 12, 1))
                            if investment_omr > 0 else 0)

    ca, cb, cc, cd, ce = st.columns(5)
    ca.metric("Additional Homes", f"{additional_homes:,}")
    cb.metric("New Cabinets Needed", f"{new_cabinets:,}")
    cc.metric("Investment (OMR)", f"{investment_omr:,}")
    cd.metric("Annual Revenue (OMR)", f"{annual_revenue_omr:,}")
    ce.metric("Payback (months)", f"{payback_months}")

    if additional_share_pct > 0:
        st.info(
            f"Growing from **{WAHEB_MARKET_SHARE:.0f}%** to **{target_share}%** "
            f"requires connecting **{additional_homes:,}** new homes "
            f"({new_cabinets} cabinets, {investment_omr:,} OMR investment) "
            f"generating **{annual_revenue_omr:,} OMR/year** — "
            f"payback in **{payback_months} months**.")
    else:
        st.success("You are at or above the target share. Defend current position.")


# ── Tab 6 — AI Insights ─────────────────────────────────────────────────────────

def render_ai_insights_tab(results: dict, zones_data: dict):
    """
    Tab 6: ML-powered insights.
    Sections:
      1. Telecom market simulation (5G / FTTH / mobile KPIs)
      2. Zone clustering
      3. Demand prediction scores vs YOLO scores
      4. Revenue forecast comparison
      5. Traffic trend projection
    """
    import json
    from app.forecaster import (forecast_revenue, market_share_projection,
                                 predict_demand_score, traffic_trend)

    scored = results["scored_zones"]
    zones_json_str = json.dumps(zones_data)

    st.subheader("AI-Powered Market Intelligence")
    st.caption(
        "Combines ML clustering, demand prediction, revenue forecasting, "
        "and telecom simulation. All data is simulated.")

    # ── Section 1 — Telecom simulation KPIs ──────────────────────────────────
    st.markdown("#### Telecom Market Simulation")
    sim_data = _load_telecom_sim(zones_json_str)
    totals   = sim_data.get("market_totals", {})

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mobile Subscribers", f"{totals.get('total_mobile_subscribers', 0):,}")
    k2.metric("FTTH Subscribers",   f"{totals.get('total_ftth_subscribers', 0):,}")
    k3.metric("5G Subscribers",     f"{totals.get('total_5g_subscribers', 0):,}")
    k4.metric("Monthly Revenue (OMR)",
              f"{totals.get('total_monthly_revenue_omr', 0):,.0f}")
    k5.metric("Annual Revenue (OMR)",
              f"{totals.get('total_annual_revenue_omr', 0):,.0f}")

    # Telecom breakdown table
    profiles = sim_data.get("zone_profiles", {})
    if profiles:
        tele_rows = []
        for zid, p in profiles.items():
            tele_rows.append({
                "Zone":            p["zone_name"],
                "Mobile Subs":     p["mobile_subscribers"],
                "FTTH Subs":       p["ftth_subscribers"],
                "5G Subs":         p["subscribers_5g"],
                "5G Penetration":  f"{p['penetration_5g_pct']}%",
                "ARPU (OMR)":      p["arpu_blended"],
                "Monthly Rev.":    f"{p['monthly_revenue_omr']:,.0f}",
                "Avg Data (GB)":   p["avg_data_gb_per_user_month"],
            })
        tele_df = pd.DataFrame(tele_rows).sort_values("Monthly Rev.", ascending=False)
        with st.expander("Zone-level telecom profiles", expanded=False):
            st.dataframe(tele_df, use_container_width=True,
                         height=360, hide_index=True)

    st.markdown("---")

    # ── Section 2 — Clustering ────────────────────────────────────────────────
    st.markdown("#### Zone Cluster Analysis (K-Means)")
    n_clusters = st.slider("Number of clusters", 2, 5, 3, key="cluster_n")

    scored_tuple = tuple(tuple(s.items()) for s in scored)
    cluster_df = _load_clusters(scored_tuple, zones_json_str, n_clusters)

    if not cluster_df.empty:
        _cluster_colors = {
            "High Opportunity":   WAHEB_COLOR,
            "Moderate Potential": "#fd7e14",
            "Low Priority":       "#aaaaaa",
            "Very Low Priority":  "#dddddd",
            "Emerging":           "#9b59b6",
        }

        c_left, c_right = st.columns(2)
        with c_left:
            fig_clust = px.scatter(
                cluster_df,
                x="vehicle_density",
                y="coverage_gap",
                color="cluster_label",
                size="priority_score",
                hover_name="zone_name",
                color_discrete_map=_cluster_colors,
                labels={"vehicle_density": "Vehicle Density (veh/km²)",
                        "coverage_gap":    "Coverage Gap (%)"},
                title="Zone Clusters — Density vs Gap",
            )
            fig_clust.update_layout(height=350, margin=dict(t=40))
            st.plotly_chart(fig_clust, use_container_width=True)

        with c_right:
            cluster_summary = (
                cluster_df.groupby("cluster_label")
                .agg(zones=("zone_id", "count"),
                     avg_score=("priority_score", "mean"),
                     avg_gap=("coverage_gap", "mean"))
                .reset_index()
                .rename(columns={"cluster_label": "Cluster",
                                  "zones":         "# Zones",
                                  "avg_score":     "Avg Score",
                                  "avg_gap":       "Avg Gap %"})
            )
            cluster_summary["Avg Score"] = cluster_summary["Avg Score"].round(1)
            cluster_summary["Avg Gap %"] = cluster_summary["Avg Gap %"].round(1)
            st.markdown("**Cluster Summary**")
            st.dataframe(cluster_summary, use_container_width=True,
                         hide_index=True, height=200)

            st.markdown("**Zone-Cluster Assignments**")
            assign_df = cluster_df[["zone_name", "cluster_label", "priority_score"]]
            assign_df.columns = ["Zone", "Cluster", "Score"]
            st.dataframe(assign_df.sort_values("Score", ascending=False),
                         use_container_width=True, hide_index=True, height=250)
    else:
        st.warning(
            "K-Means clustering requires `scikit-learn`. "
            "Install with: `pip install scikit-learn`")

    st.markdown("---")

    # ── Section 3 — Demand score comparison ──────────────────────────────────
    st.markdown("#### Demand Prediction vs YOLO Score")
    st.caption(
        "YOLO score = vehicle-density-weighted algorithm. "
        "Demand score = population density + coverage gap + commercial activity. "
        "Agreement validates zone priority.")

    demand_rows = []
    zone_meta = {z["id"]: z for z in zones_data["zones"]}
    for s in scored:
        z = zone_meta.get(s["zone_id"], {})
        demand_rows.append({
            "Zone":          s["zone_name"],
            "YOLO Score":    s["priority_score"],
            "Demand Score":  predict_demand_score(z),
            "Delta":         round(s["priority_score"] - predict_demand_score(z), 1),
            "Tier":          s["tier"],
        })
    demand_df = pd.DataFrame(demand_rows).sort_values("YOLO Score", ascending=False)

    fig_comp = px.scatter(
        demand_df,
        x="Demand Score",
        y="YOLO Score",
        color="Tier",
        hover_name="Zone",
        color_discrete_map={
            "High Priority":   "#dc3545",
            "Medium Priority": "#fd7e14",
            "Low Priority":    "#28a745",
        },
        text="Zone",
        size_max=12,
        title="Demand Score vs YOLO Priority Score",
    )
    fig_comp.update_traces(textposition="top center", textfont_size=9)
    fig_comp.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                       line=dict(dash="dot", color="gray", width=1))
    fig_comp.update_layout(height=380, margin=dict(t=40))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # ── Section 4 — Revenue forecast ─────────────────────────────────────────
    st.markdown("#### 5-Year Revenue Forecast (Compound Growth Model)")
    growth_rate = st.slider("Annual subscriber growth rate (%)",
                             min_value=1, max_value=30, value=8,
                             key="growth_slider") / 100

    high_zones = [s for s in scored if s["tier"] == "High Priority"]
    if not high_zones:
        high_zones = sorted(scored, key=lambda x: x["priority_score"], reverse=True)[:3]

    fig_fc = go.Figure()
    for z in high_zones:
        fc = forecast_revenue(z, years=5, growth_rate=growth_rate)
        fig_fc.add_trace(go.Scatter(
            x=fc["years"],
            y=fc["cumulative_revenue"],
            mode="lines+markers",
            name=z["zone_name"],
            line=dict(width=2),
        ))
        if fc["breakeven_year"]:
            fig_fc.add_vline(
                x=fc["breakeven_year"],
                line_dash="dot",
                line_color="gray",
                annotation_text=f"{z['zone_name'][:8]} break-even",
                annotation_font_size=9,
            )
    fig_fc.update_layout(
        xaxis_title="Year",
        yaxis_title="Cumulative Revenue (OMR)",
        height=360,
        margin=dict(t=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")

    # ── Section 5 — Market share S-curve ─────────────────────────────────────
    st.markdown("#### Market Share Growth Projection (S-Curve)")
    col_a, col_b = st.columns(2)
    with col_a:
        ms_target = st.slider("Target market share (%)",
                               min_value=int(WAHEB_MARKET_SHARE),
                               max_value=45, value=25,
                               key="ms_target_ai")
    with col_b:
        ms_years = st.slider("Projection years", 3, 8, 5, key="ms_years_ai")

    shares = market_share_projection(
        WAHEB_MARKET_SHARE, ms_target, years=ms_years)
    fig_ms = go.Figure()
    fig_ms.add_trace(go.Scatter(
        x=list(range(1, ms_years + 1)),
        y=shares,
        mode="lines+markers",
        line=dict(color=WAHEB_COLOR, width=3),
        fill="tozeroy",
        fillcolor=f"rgba(2,195,154,0.15)",
        name="Waheb Market Share",
    ))
    fig_ms.add_hline(y=ms_target, line_dash="dash",
                     line_color="gray",
                     annotation_text=f"Target {ms_target}%")
    fig_ms.update_layout(
        xaxis_title="Year",
        yaxis_title="Market Share (%)",
        height=300,
        margin=dict(t=10),
    )
    st.plotly_chart(fig_ms, use_container_width=True)

    # ── Section 6 — Traffic trend ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Network Traffic Growth Trend (35 % YoY)")
    st.caption("ITU benchmark: data consumption grows ~35 % per year in GCC markets.")

    if profiles:
        from app.forecaster import traffic_trend
        top_profiles = [
            p for p in list(profiles.values())[:6]
        ]
        trend_rows = traffic_trend(top_profiles, years=5)
        trend_df = pd.DataFrame(trend_rows)
        fig_traf = px.line(
            trend_df,
            x="year",
            y="peak_gbps",
            color="zone_name",
            markers=True,
            labels={"year": "Year", "peak_gbps": "Peak Traffic (Gbps)",
                    "zone_name": "Zone"},
        )
        fig_traf.update_layout(height=320, margin=dict(t=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_traf, use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    mode_badge = (
        '<span class="mode-badge demo-badge">DEMO</span>'
        if DEMO_MODE
        else '<span class="mode-badge prod-badge">PRODUCTION</span>'
    )
    st.markdown(f"""
    <div class="falcon-header">
      <h1>🦅 {APP_TITLE}</h1>
      <p>{APP_SUBTITLE}</p>
      {mode_badge}
    </div>
    """, unsafe_allow_html=True)

    zones_data = _load_zones()
    results = _load_results(DEMO_MODE)

    if "filtered_zones" not in st.session_state:
        st.session_state["filtered_zones"] = results["scored_zones"]

    with st.sidebar:
        render_sidebar(results, zones_data)

    network_data = _load_network_data()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️  Map",
        "📊  Rankings",
        "🔍  Detection",
        "💼  Business Case",
        "⚔️  Competitive Intel",
        "🤖  AI Insights",
    ])

    with tab1:
        render_map_tab(zones_data, results)
    with tab2:
        render_rankings_tab(results, zones_data)
    with tab3:
        render_detection_tab(results, zones_data)
    with tab4:
        render_business_case_tab(results, zones_data)
    with tab5:
        render_competitive_tab(network_data, zones_data)
    with tab6:
        render_ai_insights_tab(results, zones_data)


if __name__ == "__main__":
    main()
