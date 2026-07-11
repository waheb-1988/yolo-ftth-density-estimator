"""Ooredoo Falcon — Streamlit dashboard (Landing | Login | Dashboard with 6 tabs)."""
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
                         ARPU_MONTHLY_OMR, HOUSEHOLD_SIZE, FTTH_UPTAKE_RATE,
                         HOUSEHOLDS_PER_BUILDING,
                         OOREDOO_DEFAULT_SHARE, COMPETITORS_DEFAULT_SHARE,
                         WAHEB_BRAND_NAME, RAMI_BRAND_NAME, DON_BRAND_NAME,
                         OTHERS_BRAND_NAME)
from app.data_loader import load_network_data, load_or_generate_results, load_zones
from app.map_builder import build_map
from app.report_exporter import export_csv, generate_executive_summary
from app.zone_intelligence import (ZoneIntelligence, latest_zone_intelligence,
                                    save_zone_intelligence)
from app.marketing_intelligence import load_marketing_intelligence, save_marketing_intelligence

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Ooredoo Falcon | FTTH Intelligence Platform",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_MODE = os.environ.get("FALCON_DEMO_MODE", "1") == "1"

# ── Demo credentials ────────────────────────────────────────────────────────────
DEMO_EMAIL    = "test@app.ai"
DEMO_PASSWORD = "1234"

# ── Load SVG/image assets as base64 data URIs (avoids inline SVG ID conflicts) ──
import base64 as _b64, mimetypes as _mime

def _asset_as_data_uri(filename: str) -> str:
    """Return a base64 data-URI for any file in data/, or '' on missing."""
    path = Path(__file__).parent.parent / "data" / filename
    if not path.exists():
        return ""
    m = _mime.guess_type(str(path))[0] or ("image/svg+xml" if filename.endswith(".svg") else "image/jpeg")
    return f"data:{m};base64,{_b64.b64encode(path.read_bytes()).decode()}"

_LOGO_URI  = _asset_as_data_uri("ooredoo_falcon_logo.svg")   # 640×190 banner
_ICON_URI  = _asset_as_data_uri("ooredoo_falcon_icon.svg")   # 200×200 icon

# Real falcon photo overrides the geometric icon if present
_FALCON_PHOTO = (
    _asset_as_data_uri("falcon_photo.jpg") or
    _asset_as_data_uri("falcon_photo.png")
)
_ICON_URI = _FALCON_PHOTO or _ICON_URI   # prefer real photo

# ── Comprehensive Ooredoo CSS ───────────────────────────────────────────────────
OOREDOO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── CSS Variables ── */
:root {
  --oo-red:        #E10A0A;
  --oo-deep-red:   #B00000;
  --oo-dark-navy:  #1A1A2E;
  --oo-mid-navy:   #16213E;
  --oo-white:      #FFFFFF;
  --oo-page-bg:    #F5F7FA;
  --oo-shadow:     0 4px 20px rgba(0,0,0,0.08);
  --oo-radius:     12px;
}

/* ── Global ── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDeployButton"]  { display: none; }
[data-testid="stToolbar"]       { display: none; }
[data-testid="stDecoration"]    { display: none; }

/* ── App background ── */
.stApp {
  background-color: var(--oo-page-bg) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--oo-dark-navy) !important;
}
[data-testid="stSidebar"] * {
  color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stDivider,
[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] [data-testid="stCaption"] p {
  color: rgba(255,255,255,0.6) !important;
}
[data-testid="stSidebar"] .stSubheader {
  color: #FFFFFF !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--oo-white);
  border-radius: 10px 10px 0 0;
  border-bottom: 2px solid #e8eaf0;
  gap: 2px;
  padding: 0 8px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500;
  font-size: 0.88rem;
  color: #555 !important;
  padding: 10px 16px;
  border-radius: 8px 8px 0 0;
  transition: all 0.2s ease;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  color: var(--oo-red) !important;
  background: rgba(225,10,10,0.06);
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--oo-red) !important;
  border-bottom: 3px solid var(--oo-red) !important;
  font-weight: 700 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
  background: var(--oo-white);
  border-radius: 0 0 12px 12px;
  padding: 20px;
}

/* ── All buttons ── */
.stButton > button {
  background: var(--oo-red) !important;
  color: var(--oo-white) !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  padding: 10px 20px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 8px rgba(225,10,10,0.25) !important;
}
.stButton > button:hover {
  background: var(--oo-deep-red) !important;
  box-shadow: 0 4px 16px rgba(225,10,10,0.4) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
  background: var(--oo-dark-navy) !important;
  color: var(--oo-white) !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
  background: var(--oo-mid-navy) !important;
  box-shadow: 0 4px 14px rgba(26,26,46,0.4) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: var(--oo-white) !important;
  border-radius: var(--oo-radius) !important;
  padding: 16px 20px !important;
  border-left: 4px solid var(--oo-red) !important;
  box-shadow: var(--oo-shadow) !important;
}
[data-testid="stMetricValue"] {
  color: var(--oo-dark-navy) !important;
  font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
  color: #555 !important;
  font-weight: 500 !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--oo-red) !important;
  border-color: var(--oo-red) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
  background: var(--oo-red) !important;
}

/* ── Mode badges ── */
.mode-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  margin-top: 8px;
}
.demo-badge { background: #E10A0A; color: white; }
.prod-badge { background: #28a745; color: white; }

/* ── KPI component ── */
.oo-kpi {
  background: var(--oo-white);
  border-radius: var(--oo-radius);
  border-top: 4px solid var(--oo-red);
  padding: 20px 16px;
  text-align: center;
  box-shadow: var(--oo-shadow);
  height: 100%;
}
.oo-kpi .oo-kpi-value {
  font-size: 2.2rem;
  font-weight: 800;
  color: var(--oo-dark-navy);
  line-height: 1;
  margin-bottom: 6px;
}
.oo-kpi .oo-kpi-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: #444;
  margin-bottom: 4px;
}
.oo-kpi .oo-kpi-sub {
  font-size: 0.75rem;
  color: #888;
}

/* ── Card component ── */
.oo-card {
  background: var(--oo-white);
  border-radius: var(--oo-radius);
  padding: 20px 24px;
  box-shadow: var(--oo-shadow);
  margin-bottom: 16px;
}
.oo-card-accent {
  border-left: 4px solid var(--oo-red);
}

/* ── Badge pills ── */
.oo-badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.4px;
}
.oo-badge-red   { background: #E10A0A; color: white; }
.oo-badge-navy  { background: #1A1A2E; color: white; }
.oo-badge-green { background: #28a745; color: white; }
.oo-badge-amber { background: #fd7e14; color: white; }

/* ── Section titles ── */
.oo-section-title {
  color: var(--oo-dark-navy);
  font-size: 1.25rem;
  font-weight: 700;
  margin: 10px 0 18px;
  padding-bottom: 10px;
  border-bottom: 2px solid #edf0f5;
  display: flex;
  align-items: center;
  gap: 8px;
}
.oo-section-title::after {
  content: '';
  display: block;
  height: 3px;
  width: 48px;
  background: linear-gradient(90deg, var(--oo-red), transparent);
  border-radius: 4px;
  margin-left: 4px;
}

/* ── Hero section ── */
.oo-hero {
  background: linear-gradient(135deg, #0D0D1A 0%, #1A1A2E 55%, #2C0A0A 100%);
  border-radius: 16px;
  padding: 60px 40px;
  text-align: center;
  position: relative;
  overflow: hidden;
  margin-bottom: 24px;
}
.oo-hero::before {
  content: '';
  position: absolute;
  top: -80px; right: -80px;
  width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(225,10,10,0.18) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}
.oo-hero::after {
  content: '';
  position: absolute;
  bottom: -60px; left: -60px;
  width: 240px; height: 240px;
  background: radial-gradient(circle, rgba(225,10,10,0.12) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}

/* ── Feature cards (glass) ── */
.oo-feat {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 20px 16px;
  text-align: center;
  color: white;
  backdrop-filter: blur(6px);
  transition: all 0.2s ease;
}
.oo-feat:hover {
  background: rgba(255,255,255,0.10);
  border-color: rgba(225,10,10,0.5);
  transform: translateY(-2px);
}

/* ── Stat cards (glass) ── */
.oo-stat {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
  color: white;
}

/* ── Dashboard header ── */
.oo-header {
  background: linear-gradient(135deg, #0D0D1A 0%, #1A1A2E 60%, #2C0A0A 100%);
  color: white;
  padding: 22px 30px;
  border-radius: 14px;
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 6px 24px rgba(0,0,0,0.18);
}
.oo-header-left h1 {
  margin: 0;
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: 1.5px;
  color: white;
}
.oo-header-left p {
  margin: 4px 0 0;
  opacity: 0.72;
  font-size: 0.88rem;
  color: white;
}
.oo-header-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 6px;
}
.oo-city-label {
  color: rgba(255,255,255,0.65);
  font-size: 0.80rem;
  margin-top: 4px;
}

/* ── Priority tier colors ── */
.high-p  { color: #dc3545; font-weight: 700; }
.med-p   { color: #fd7e14; font-weight: 700; }
.low-p   { color: #28a745; font-weight: 700; }

/* ── Animations ── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(30px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 8px rgba(225,10,10,0.3); }
  50%       { box-shadow: 0 0 24px rgba(225,10,10,0.7); }
}
.anim-fade-up  { animation: fadeInUp 0.7s ease forwards; }
.anim-fade-in  { animation: fadeIn 0.5s ease forwards; }
.anim-glow     { animation: pulse-glow 2.5s infinite; }
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


# ── Helper: hide sidebar via CSS ────────────────────────────────────────────────

def _hide_sidebar():
    st.markdown("""
    <style>
      [data-testid="stSidebar"]            { display: none !important; }
      [data-testid="collapsedControl"]     { display: none !important; }
      .css-1rs6os                          { display: none !important; }
      section[data-testid="stSidebar"]     { display: none !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Landing Page ────────────────────────────────────────────────────────────────

def render_landing_page():
    _hide_sidebar()

    # ── Hero Section ──────────────────────────────────────────────────────────
    logo_html = (
        f'<div style="max-width:560px;margin:0 auto 24px;">'
        f'<img src="{_LOGO_URI}" style="width:100%;height:auto;" alt="Ooredoo Falcon"/></div>'
        if _LOGO_URI else
        '<div style="font-size:4rem;margin-bottom:16px;">🦅</div>'
        '<h1 style="font-size:3.2rem;font-weight:900;color:white;margin:0 0 12px;">'
        'Ooredoo Falcon</h1>'
    )
    st.markdown(f"""
    <div class="oo-hero anim-fade-in">
      {logo_html}
      <p style="font-size:1.15rem;color:rgba(255,255,255,0.80);max-width:600px;
                margin:0 auto 28px;line-height:1.6;">
        AI-Powered FTTH Market Intelligence &amp; Expansion Platform
      </p>
      <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;">
        <span class="oo-badge oo-badge-red">📡 Ooredoo FTTH</span>
        <span class="oo-badge" style="background:rgba(255,255,255,0.15);color:white;">
          Muscat, Oman
        </span>
        <span class="oo-badge" style="background:rgba(255,255,255,0.15);color:white;">
          2026 Platform
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats Row ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A1A2E,#16213E);
                border-radius:14px;padding:28px 24px;margin-bottom:24px;">
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">
        <div class="oo-stat">
          <div style="font-size:2.4rem;font-weight:900;color:#E10A0A;">15</div>
          <div style="font-size:0.88rem;color:rgba(255,255,255,0.8);margin-top:4px;
                      font-weight:600;">Zones Analyzed</div>
          <div style="font-size:0.74rem;color:rgba(255,255,255,0.5);margin-top:2px;">
            Across Muscat City</div>
        </div>
        <div class="oo-stat">
          <div style="font-size:2.4rem;font-weight:900;color:#E10A0A;">350K+</div>
          <div style="font-size:0.88rem;color:rgba(255,255,255,0.8);margin-top:4px;
                      font-weight:600;">Homes in Market</div>
          <div style="font-size:0.74rem;color:rgba(255,255,255,0.5);margin-top:2px;">
            Total addressable</div>
        </div>
        <div class="oo-stat">
          <div style="font-size:2.4rem;font-weight:900;color:#E10A0A;">30%</div>
          <div style="font-size:0.88rem;color:rgba(255,255,255,0.8);margin-top:4px;
                      font-weight:600;">Unserved Market</div>
          <div style="font-size:0.74rem;color:rgba(255,255,255,0.5);margin-top:2px;">
            Growth opportunity</div>
        </div>
        <div class="oo-stat">
          <div style="font-size:2.4rem;font-weight:900;color:#E10A0A;">95%</div>
          <div style="font-size:0.88rem;color:rgba(255,255,255,0.8);margin-top:4px;
                      font-weight:600;">Detection Accuracy</div>
          <div style="font-size:0.74rem;color:rgba(255,255,255,0.5);margin-top:2px;">
            YOLOv8 model</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Grid ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;margin-bottom:20px;">
      <span style="color:#E10A0A;font-size:0.85rem;font-weight:700;
                   letter-spacing:2px;text-transform:uppercase;">Platform Capabilities</span>
      <h2 style="color:#1A1A2E;font-size:1.8rem;font-weight:800;margin:8px 0 4px;">
        Everything You Need to Win FTTH Market Share
      </h2>
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("🔬", "Computer Vision",       "YOLOv8 aerial building & vehicle detection for fiber density scoring"),
        ("📊", "Market Intelligence",   "Zone-level priority rankings with investment & ROI calculations"),
        ("⚔️", "Competition Analysis",  "Real-time competitive gap analysis across Muscat zones"),
        ("🎯", "Customer Acquisition",  "Market share modeling & revenue opportunity by operator"),
        ("🗺️", "Expansion Planning",    "Interactive map with fiber route simulation & OLT placement"),
        ("🤖", "AI Insights",           "ML clustering, demand prediction & 5-year revenue forecasting"),
    ]

    cols1 = st.columns(3)
    for i, (icon, title, desc) in enumerate(features[:3]):
        cols1[i].markdown(f"""
        <div style="background:white;border-radius:12px;padding:24px 20px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.08);text-align:center;
                    border-top:4px solid #E10A0A;height:100%;margin-bottom:16px;">
          <div style="font-size:2.4rem;margin-bottom:12px;">{icon}</div>
          <div style="font-size:1.0rem;font-weight:700;color:#1A1A2E;margin-bottom:8px;">
            {title}</div>
          <div style="font-size:0.82rem;color:#666;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    cols2 = st.columns(3)
    for i, (icon, title, desc) in enumerate(features[3:]):
        cols2[i].markdown(f"""
        <div style="background:white;border-radius:12px;padding:24px 20px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.08);text-align:center;
                    border-top:4px solid #E10A0A;height:100%;margin-bottom:16px;">
          <div style="font-size:2.4rem;margin-bottom:12px;">{icon}</div>
          <div style="font-size:1.0rem;font-weight:700;color:#1A1A2E;margin-bottom:8px;">
            {title}</div>
          <div style="font-size:0.82rem;color:#666;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Business Value Section ─────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A1A2E,#16213E);
                border-radius:14px;padding:32px 28px;margin:24px 0;">
      <div style="text-align:center;margin-bottom:24px;">
        <span style="color:#E10A0A;font-size:0.82rem;font-weight:700;
                     letter-spacing:2px;text-transform:uppercase;">Business Value</span>
        <h2 style="color:white;font-size:1.6rem;font-weight:800;margin:8px 0 0;">
          Built for Every Stakeholder
        </h2>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;">
        <div class="oo-feat">
          <div style="font-size:2rem;margin-bottom:10px;">📣</div>
          <div style="font-weight:700;font-size:1.0rem;margin-bottom:8px;">Marketing</div>
          <div style="font-size:0.82rem;color:rgba(255,255,255,0.72);line-height:1.5;">
            Identify high-value zones for targeted FTTH campaigns.
            Segment customers by demand signal &amp; competitive pressure.
          </div>
        </div>
        <div class="oo-feat">
          <div style="font-size:2rem;margin-bottom:10px;">⚙️</div>
          <div style="font-weight:700;font-size:1.0rem;margin-bottom:8px;">Operations</div>
          <div style="font-size:0.82rem;color:rgba(255,255,255,0.72);line-height:1.5;">
            Prioritize fiber rollout with AI-scored deployment maps.
            Optimize cabinet placement &amp; reduce infrastructure waste.
          </div>
        </div>
        <div class="oo-feat">
          <div style="font-size:2rem;margin-bottom:10px;">📈</div>
          <div style="font-weight:700;font-size:1.0rem;margin-bottom:8px;">Management</div>
          <div style="font-size:0.82rem;color:rgba(255,255,255,0.72);line-height:1.5;">
            5-year revenue forecasts &amp; ROI breakeven analysis.
            Executive summaries exportable in one click.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA Button ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    cta_left, cta_mid, cta_right = st.columns([2, 1.5, 2])
    with cta_mid:
        if st.button("🚀 Enter Platform", use_container_width=True, key="landing_cta"):
            st.session_state.page = "login"
            st.rerun()

    st.markdown("""
    <div style="text-align:center;margin-top:12px;">
      <span style="font-size:0.78rem;color:#999;">
        Secure platform &middot; Demo credentials provided on login page
      </span>
    </div>
    <br>
    """, unsafe_allow_html=True)


# ── Login Page ──────────────────────────────────────────────────────────────────

def render_login_page():
    _hide_sidebar()

    # Dark gradient page background
    st.markdown("""
    <style>
    .stApp {
      background: linear-gradient(135deg, #0D0D1A 0%, #1A1A2E 60%, #16213E 100%) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        # White card header (pure HTML — no Streamlit form elements inside)
        if _ICON_URI:
            _is_photo = bool(_FALCON_PHOTO)
            _style = (
                "width:100%;height:100%;object-fit:cover;"
                if _is_photo else "width:100%;height:auto;"
            )
            _wrap = (
                "border-radius:50%;overflow:hidden;border:3px solid #E10A0A;"
                if _is_photo else ""
            )
            login_icon = (
                f'<div style="width:90px;height:90px;margin:0 auto 10px;{_wrap}">'
                f'<img src="{_ICON_URI}" style="{_style}" alt="Falcon"/></div>'
            )
        else:
            login_icon = '<div style="font-size:3.5rem;margin-bottom:10px;">🦅</div>'
        st.markdown(f"""
        <div style="background:white;border-radius:16px 16px 0 0;padding:36px 36px 20px;
                    box-shadow:0 20px 60px rgba(0,0,0,0.35);text-align:center;">
          {login_icon}
          <h2 style="color:#1A1A2E;font-size:1.6rem;font-weight:800;margin:0 0 6px;">
            Ooredoo Falcon</h2>
          <p style="color:#888;font-size:0.88rem;margin:0 0 16px;">
            FTTH Intelligence Platform</p>
          <hr style="border:none;border-top:1px solid #eee;">
        </div>
        """, unsafe_allow_html=True)

        with st.form(key="login_form"):
            email = st.text_input(
                "Email Address",
                placeholder="your@email.com",
                key="login_email",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="••••••••",
                key="login_password",
            )
            submitted = st.form_submit_button(
                "Sign In →",
                use_container_width=True,
            )

        if submitted:
            if email == DEMO_EMAIL and password == DEMO_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("Invalid credentials. Please check your email and password.")

        st.markdown("""
        <div style="background:#fff8e8;border:1px solid #ffd78c;border-radius:8px;
                    padding:12px 16px;margin-top:16px;text-align:center;font-size:0.82rem;
                    color:#856404;">
          <b>Demo Access</b><br>
          Email: <code>test@app.ai</code> &nbsp;/&nbsp; Password: <code>1234</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        back_col1, back_col2, back_col3 = st.columns([1, 2, 1])
        with back_col2:
            if st.button("← Back to Home", use_container_width=True, key="login_back"):
                st.session_state.page = "landing"
                st.rerun()


# ── Dashboard Header ────────────────────────────────────────────────────────────

def render_app_header():
    mode_badge_html = (
        '<span class="mode-badge demo-badge">DEMO</span>'
        if DEMO_MODE
        else '<span class="mode-badge prod-badge">PRODUCTION</span>'
    )
    if _ICON_URI:
        _ph = bool(_FALCON_PHOTO)
        icon_html = (
            f'<div style="width:54px;height:54px;flex-shrink:0;'
            f'{"border-radius:50%;overflow:hidden;border:2px solid rgba(225,10,10,0.6);" if _ph else ""}">'
            f'<img src="{_ICON_URI}" style="width:100%;height:{"100%;object-fit:cover" if _ph else "auto"};" alt="Falcon"/></div>'
        )
    else:
        icon_html = '<span style="font-size:2.4rem;">🦅</span>'
    st.markdown(f"""
    <div class="oo-header">
      <div class="oo-header-left" style="display:flex;align-items:center;gap:14px;">
        {icon_html}
        <div>
          <h1 style="margin:0;font-size:1.7rem;font-weight:800;letter-spacing:1px;">
            Ooredoo Falcon</h1>
          <p style="margin:2px 0 0;font-size:0.76rem;color:rgba(255,255,255,0.60);">
            AI-Powered FTTH Market Intelligence Platform</p>
        </div>
      </div>
      <div class="oo-header-right">
        <div style="display:flex;gap:8px;align-items:center;">
          <span class="oo-badge oo-badge-red anim-glow">📡 Ooredoo FTTH</span>
          {mode_badge_html}
        </div>
        <div class="oo-city-label">Muscat, Oman &bull; 2026</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────────

def render_sidebar(results: dict, zones_data: dict):
    if _ICON_URI:
        _ph = bool(_FALCON_PHOTO)
        icon_sidebar = (
            f'<div style="width:72px;height:72px;margin:0 auto 6px;'
            f'{"border-radius:50%;overflow:hidden;border:3px solid #E10A0A;" if _ph else ""}">'
            f'<img src="{_ICON_URI}" style="width:100%;height:{"100%;object-fit:cover" if _ph else "auto"};" alt="Falcon"/></div>'
        )
    else:
        icon_sidebar = '<span style="font-size:2.8rem;">🦅</span><br>'
    st.markdown(f"""
    <div style="text-align:center;padding:14px 0 10px;">
      {icon_sidebar}
      <span style="font-size:1.1rem;font-weight:800;color:#FFFFFF;letter-spacing:1px;">
        Ooredoo Falcon</span><br>
      <span style="font-size:0.72rem;color:rgba(255,255,255,0.55);">
        FTTH Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

    # Ooredoo FTTH badge
    st.markdown("""
    <div style="text-align:center;margin:8px 0 14px;">
      <span style="background:#E10A0A;color:white;padding:5px 18px;
                   border-radius:20px;font-size:0.82rem;font-weight:700;
                   letter-spacing:0.8px;">
        📡 Ooredoo FTTH
      </span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    mode_label = "DEMO MODE" if DEMO_MODE else "PRODUCTION"
    badge_cls  = "demo-badge" if DEMO_MODE else "prod-badge"
    st.markdown(
        f'<span class="mode-badge {badge_cls}">{mode_label}</span>',
        unsafe_allow_html=True)
    st.caption("Run `python run.py --demo` to regenerate data.")
    st.divider()

    # All zones shown by default
    scored = results["scored_zones"]
    st.session_state["filtered_zones"] = scored

    # Competition section
    st.subheader("⚔️ Competition")
    unserved_homes = int(TOTAL_MUSCAT_HOMES * UNSERVED_MARKET_SHARE / 100)

    competitors = [
        ("📡 Ooredoo",  f"{DON_MARKET_SHARE:.0f}%",       DON_COLOR,   "Main operator"),
        ("🔷 Omantel",  f"{RAMI_MARKET_SHARE:.0f}%",      RAMI_COLOR,  "Dominant competitor"),
        ("🟢 Awasr",    f"{WAHEB_MARKET_SHARE:.0f}%",     WAHEB_COLOR, "Growing challenger"),
        ("⬜ Others",   f"{UNSERVED_MARKET_SHARE:.0f}%",  "#FFA500",   "Addressable market"),
    ]
    for name, share, color, subtitle in competitors:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:7px 10px;border-left:3px solid {color};
                    margin-bottom:5px;background:rgba(255,255,255,0.07);
                    border-radius:0 6px 6px 0;">
          <div>
            <div style="font-size:0.82rem;font-weight:600;color:white;">{name}</div>
            <div style="font-size:0.70rem;color:rgba(255,255,255,0.5);">{subtitle}</div>
          </div>
          <span style="font-size:1.0rem;font-weight:700;color:{color};">{share}</span>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        f"Total market: {TOTAL_MUSCAT_HOMES:,} homes · "
        f"Unserved: {unserved_homes:,}")
    st.divider()

    # Generate Presentation button (PPTX if script exists)
    pptx_script = Path(__file__).parent.parent / "scripts" / "generate_pptx.py"
    if pptx_script.exists():
        if st.button("📊 Generate Presentation", use_container_width=True,
                     key="gen_pptx_btn"):
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(pptx_script)],
                    capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    st.success("Presentation generated!")
                else:
                    st.error(f"Error: {result.stderr[:200]}")
            except Exception as e:
                st.error(f"Could not generate presentation: {e}")
        st.divider()

    # Export section
    st.subheader("📥 Export Reports")

    csv_data = export_csv(scored, zones_data)
    st.download_button(
        label="⬇ Download CSV Report",
        data=csv_data,
        file_name="ooredoo_falcon_muscat_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

    summary = generate_executive_summary(scored, zones_data)
    st.download_button(
        label="⬇ Executive Summary (.txt)",
        data=summary,
        file_name="ooredoo_falcon_executive_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()
    st.caption(
        "**Ooredoo Falcon v2.0** · AI-Powered FTTH\n"
        "Market Intelligence · Muscat, Oman · 2026")


# ── Coverage Map (workflow stage 1) ─────────────────────────────────────────────

def render_map_tab(zones_data: dict, results: dict):
    st.markdown('<div class="oo-section-title">🗺️ Coverage Map — Muscat City</div>',
                unsafe_allow_html=True)

    scored   = results["scored_zones"]
    filtered = st.session_state.get("filtered_zones", scored)

    high_n = sum(1 for s in scored if s["tier"] == "High Priority")
    med_n  = sum(1 for s in scored if s["tier"] == "Medium Priority")
    low_n  = sum(1 for s in scored if s["tier"] == "Low Priority")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Zones",        len(scored))
    c2.metric("🔴 High Priority",   high_n)
    c3.metric("🟠 Medium Priority", med_n)
    c4.metric("🟢 Low Priority",    low_n)

    st.markdown("---")
    network_data = _load_network_data()
    if network_data:
        st.subheader("Competitive Network Map — Muscat City")
        st.caption(
            "Toggle layers via control (top-right). "
            "Green=Awasr | Blue=Omantel | Red=Ooredoo | Gold=Opportunity zones.")
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


# ── Computer Vision Engine (workflow stage 2) ───────────────────────────────────

# Midan Residence, Ghubrah — anchor coordinates for fiber simulation
_MIDAN_LAT = 23.6074
_MIDAN_LON = 58.5122

# Simulated fiber infrastructure around Midan Residence
_FIBER_POINTS = {
    "olt": {
        "lat": _MIDAN_LAT, "lon": _MIDAN_LON,
        "label": "OLT Cabinet — Midan Residence",
        "icon": "star", "color": "red",
    },
    "splitters": [
        {"lat": 23.6083, "lon": 58.5107, "label": "Splitter A", "homes": 32},
        {"lat": 23.6063, "lon": 58.5138, "label": "Splitter B", "homes": 28},
        {"lat": 23.6091, "lon": 58.5133, "label": "Splitter C", "homes": 35},
        {"lat": 23.6058, "lon": 58.5112, "label": "Splitter D", "homes": 24},
        {"lat": 23.6079, "lon": 58.5148, "label": "Splitter E", "homes": 30},
    ],
}


def _build_fiber_map() -> object:
    """Return a Folium map with simulated FTTH fiber points near Midan Residence."""
    import folium

    m = folium.Map(
        location=[_MIDAN_LAT, _MIDAN_LON],
        zoom_start=16,
        tiles="OpenStreetMap",
    )

    olt = _FIBER_POINTS["olt"]
    folium.Marker(
        [olt["lat"], olt["lon"]],
        popup=folium.Popup(olt["label"], max_width=200),
        tooltip=olt["label"],
        icon=folium.Icon(color=olt["color"], icon=olt["icon"], prefix="fa"),
    ).add_to(m)

    for sp in _FIBER_POINTS["splitters"]:
        folium.CircleMarker(
            [sp["lat"], sp["lon"]],
            radius=9,
            popup=folium.Popup(f"{sp['label']} — {sp['homes']} homes", max_width=180),
            tooltip=f"{sp['label']} ({sp['homes']} homes)",
            color="#0055AA",
            fill=True,
            fill_color="#0099FF",
            fill_opacity=0.85,
        ).add_to(m)
        folium.PolyLine(
            [[olt["lat"], olt["lon"]], [sp["lat"], sp["lon"]]],
            color="#E10A0A",
            weight=2.5,
            opacity=0.85,
            dash_array="6 4",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _ftth_priority(building_count: int) -> tuple[str, str, str]:
    """Return (label, color, advice) based on detected building count."""
    if building_count >= 25:
        return ("High Priority", "#dc3545",
                "Immediate deployment recommended — high building density detected.")
    if building_count >= 12:
        return ("Medium Priority", "#fd7e14",
                "Plan fiber rollout within 6 months.")
    return ("Low Priority", "#28a745",
            "Include in Phase 2 expansion plan.")


def render_cv_engine_tab(results: dict, zones_data: dict):
    """Computer Vision Engine — position 2 in the FALCON workflow.

    Analyzes aerial imagery for buildings/cars and extracts Zone Intelligence
    (market potential, FTTH opportunity, coverage signal) that is persisted
    for the Marketing Intelligence Engine and Executive Report stages.
    """
    st.markdown('<div class="oo-section-title">🔬 Computer Vision Engine — FTTH Site Analysis</div>',
                unsafe_allow_html=True)
    st.caption(
        "Upload an aerial screenshot of any Ghubrah location. "
        "The custom YOLOv8 model counts **buildings** and **cars** "
        "to estimate fiber deployment priority.")

    # ── Row 1: fiber map + upload controls ────────────────────────────────────
    col_map, col_ctrl = st.columns([1, 1])

    with col_map:
        st.markdown("##### Simulated Fiber Network — Midan Residence, Ghubrah")
        st.caption(
            "Red star = OLT cabinet &nbsp;|&nbsp; "
            "Blue circles = distribution splitters &nbsp;|&nbsp; "
            "Dashed red = fiber routes (Ooredoo)")
        try:
            from streamlit_folium import st_folium
            fiber_map = _build_fiber_map()
            st_folium(fiber_map, height=340, use_container_width=True,
                      returned_objects=[])
        except Exception as e:
            st.info(f"Map unavailable: {e}")

        total_homes = sum(sp["homes"] for sp in _FIBER_POINTS["splitters"])
        m1, m2 = st.columns(2)
        m1.metric("Simulated Splitters", len(_FIBER_POINTS["splitters"]))
        m2.metric("Homes Passed (sim)",  total_homes)

    with col_ctrl:
        st.markdown("##### Upload Aerial Screenshot")
        st.caption(
            "Take a screenshot near the fiber location "
            "(Google Maps satellite view, drone photo, etc.)")

        aerial_mode = st.checkbox(
            "Aerial / satellite mode (4x upscale + tiling)",
            value=True,
            help="Required for Google Earth / satellite images "
                 "where buildings and cars appear small.",
            key="aerial_mode_checkbox",
        )
        conf_thresh = st.slider(
            "Detection confidence threshold", 0.10, 0.60, 0.20, 0.05,
            key="conf_thresh_slider",
        )

        st.markdown("""
        <div style="background:#f0f7ff;border-radius:8px;padding:10px 14px;
                    border-left:4px solid #E10A0A;font-size:0.82rem;">
          <b>Color legend:</b><br>
          <span style="color:#FF8C00;">&#9632;</span> <b>Building</b> — counted for FTTH density score<br>
          <span style="color:#00DCFF;">&#9632;</span> <b>Car</b> — counted for traffic activity score
        </div>
        """, unsafe_allow_html=True)

    # ── Row 2: uploader ────────────────────────────────────────────────────────
    st.markdown("---")
    uploaded = st.file_uploader(
        "Drop your screenshot here",
        type=["jpg", "jpeg", "png"],
        key="yolo_upload",
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info(
            "No image uploaded yet. "
            "Take a satellite screenshot of the Ghubrah area "
            "(e.g. Google Maps in satellite mode) and upload it above.")
        return

    img_pil = Image.open(uploaded).convert("RGB")
    W, H = img_pil.size

    # ── Row 3: run detection ───────────────────────────────────────────────────
    with st.spinner("Running YOLOv8 detection on custom FTTH model ..."):
        try:
            from app.detector import get_detector
            from app.image_generator import draw_detections_on_image

            det = get_detector("n")
            if not det.available:
                st.error("YOLOv8 model not available. "
                         "Ensure `ultralytics` is installed.")
                return

            if aerial_mode:
                res = det.detect_aerial(img_pil, conf=conf_thresh)
                mode_label = "aerial (4x upscale + tiling)"
            else:
                res = det.detect_image_object(img_pil, conf=conf_thresh)
                mode_label = "standard"

            annotated = draw_detections_on_image(img_pil, res.boxes, width=2)

        except Exception as e:
            st.error(f"Detection error: {e}")
            return

    # ── Row 4: side-by-side images ─────────────────────────────────────────────
    img_col, ann_col = st.columns(2)
    img_col.image(img_pil,    caption=f"Original ({W}x{H})",
                  use_container_width=True)
    ann_col.image(annotated,
                  caption=f"Detected — {res.count} objects ({mode_label})",
                  use_container_width=True)

    # ── Row 5: per-class counts ────────────────────────────────────────────────
    buildings = [b for b in res.boxes if b[4] == "building"]
    cars      = [b for b in res.boxes if b[4] == "car"]
    n_buildings, n_cars = len(buildings), len(cars)
    priority_label, priority_color, priority_advice = _ftth_priority(n_buildings)

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"""
    <div style="background:#fff8f0;border-radius:10px;padding:16px;
                border-left:5px solid #FF8C00;text-align:center;">
      <div style="font-size:2.4rem;font-weight:800;color:#FF8C00;">{n_buildings}</div>
      <div style="font-weight:600;font-size:0.9rem;margin-top:2px;">Buildings Detected</div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px;">FTTH candidate units</div>
    </div>""", unsafe_allow_html=True)

    k2.markdown(f"""
    <div style="background:#f0fbff;border-radius:10px;padding:16px;
                border-left:5px solid #00AACC;text-align:center;">
      <div style="font-size:2.4rem;font-weight:800;color:#00AACC;">{n_cars}</div>
      <div style="font-weight:600;font-size:0.9rem;margin-top:2px;">Cars Detected</div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px;">Traffic density signal</div>
    </div>""", unsafe_allow_html=True)

    k3.markdown(f"""
    <div style="background:#f8f9fa;border-radius:10px;padding:16px;
                border-left:5px solid #555;text-align:center;">
      <div style="font-size:2.4rem;font-weight:800;color:#333;">{res.count}</div>
      <div style="font-weight:600;font-size:0.9rem;margin-top:2px;">Total Objects</div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px;">Model: custom FTTH</div>
    </div>""", unsafe_allow_html=True)

    k4.markdown(f"""
    <div style="background:#f9f9f9;border-radius:10px;padding:16px;
                border-left:5px solid {priority_color};text-align:center;">
      <div style="font-size:1.3rem;font-weight:800;color:{priority_color};">
        {priority_label}</div>
      <div style="font-weight:600;font-size:0.9rem;margin-top:4px;">FTTH Priority</div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px;">
        Based on building count</div>
    </div>""", unsafe_allow_html=True)

    # ── Row 6: recommendation + breakdown ─────────────────────────────────────
    st.markdown("---")
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"""
        <div style="background:#f0fff4;border-radius:10px;padding:16px 20px;
                    border-left:5px solid {priority_color};">
          <b style="font-size:1.05rem;">FTTH Deployment Recommendation</b><br><br>
          {priority_advice}<br><br>
          <b>Estimated homes to connect:</b> {n_buildings}<br>
          <b>Nearest simulated OLT:</b> Midan Residence Cabinet
            ({_MIDAN_LAT:.4f} N, {_MIDAN_LON:.4f} E)<br>
          <b>Estimated cabinets needed:</b>
            {max(1, round(n_buildings / 32))} distribution splitter(s)<br>
          <b>Detection mode:</b> {mode_label}
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("**Detection Breakdown**")
        total = max(res.count, 1)
        breakdown = []
        if n_buildings:
            breakdown.append({
                "Class": "building",
                "Count": n_buildings,
                "% of total": f"{n_buildings / total * 100:.0f}%",
                "Box color": "Orange",
            })
        if n_cars:
            breakdown.append({
                "Class": "car",
                "Count": n_cars,
                "% of total": f"{n_cars / total * 100:.0f}%",
                "Box color": "Cyan",
            })
        if breakdown:
            st.dataframe(
                pd.DataFrame(breakdown),
                use_container_width=True,
                hide_index=True,
                height=120,
            )
        else:
            st.info("No objects detected. Try lowering the confidence threshold.")

    # ── Row 7: Market Potential Analysis ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Market Potential & Customer Acquisition")
    st.caption(
        "Configure market share assumptions below. Calculations update automatically "
        "based on the number of buildings detected in the image.")

    mp_left, mp_right = st.columns([1, 1])

    with mp_left:
        st.markdown("##### Market Share Assumptions")
        ooredoo_share = st.slider(
            "Ooredoo Market Share (%)",
            min_value=5, max_value=70,
            value=int(OOREDOO_DEFAULT_SHARE),
            step=1,
            key="ooredoo_share_slider",
            help="Estimated percentage of the addressable market captured by Ooredoo FTTH",
        )
        competitor_share = st.slider(
            "Competitors Market Share (%)",
            min_value=5, max_value=80,
            value=int(COMPETITORS_DEFAULT_SHARE),
            step=1,
            key="competitor_share_slider",
            help="Combined market share of all other operators (Omantel + Awasr)",
        )
        unserved_share = max(0, 100 - ooredoo_share - competitor_share)

        if ooredoo_share + competitor_share > 100:
            st.warning(
                "Total allocated share exceeds 100%. "
                "Reduce Ooredoo or Competitors share.")
        else:
            st.info(
                f"Remaining addressable (unserved): **{unserved_share:.0f}%** "
                f"of total market")

    with mp_right:
        st.markdown("##### Market Share Distribution")
        share_data = pd.DataFrame({
            "Operator": ["Ooredoo", "Competitors", "Unserved"],
            "Share (%)": [
                ooredoo_share,
                competitor_share,
                unserved_share,
            ],
        })
        fig_share = px.pie(
            share_data, values="Share (%)", names="Operator",
            color="Operator",
            color_discrete_map={
                "Ooredoo":     DON_COLOR,
                "Competitors": RAMI_COLOR,
                "Unserved":    "#FFA500",
            },
            hole=0.42,
        )
        fig_share.update_layout(
            margin=dict(t=10, b=10), height=230, showlegend=True)
        st.plotly_chart(fig_share, use_container_width=True)

    # ── Customer acquisition breakdown ─────────────────────────────────────────
    st.markdown("##### Estimated Customer Acquisition")

    total_market_potential  = int(n_buildings * HOUSEHOLDS_PER_BUILDING)
    ooredoo_customers       = int(total_market_potential * ooredoo_share / 100)
    competitor_customers    = int(total_market_potential * competitor_share / 100)
    remaining_addressable   = int(total_market_potential * unserved_share / 100)
    ooredoo_ftth_subs       = int(ooredoo_customers * FTTH_UPTAKE_RATE)
    annual_revenue_omr      = ooredoo_ftth_subs * ARPU_MONTHLY_OMR * 12

    acq_c1, acq_c2, acq_c3, acq_c4 = st.columns(4)
    acq_c1.markdown(f"""
    <div style="background:#f8f9fa;border-radius:10px;padding:16px;
                border-left:5px solid #333;text-align:center;">
      <div style="font-size:2rem;font-weight:800;color:#333;">
        {total_market_potential:,}</div>
      <div style="font-weight:600;font-size:0.85rem;margin-top:4px;">
        Total Market Potential</div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px;">
        {n_buildings} bldgs × {HOUSEHOLDS_PER_BUILDING:.1f} households</div>
    </div>""", unsafe_allow_html=True)

    acq_c2.markdown(f"""
    <div style="background:#fff0f0;border-radius:10px;padding:16px;
                border-left:5px solid {DON_COLOR};text-align:center;">
      <div style="font-size:2rem;font-weight:800;color:{DON_COLOR};">
        {ooredoo_customers:,}</div>
      <div style="font-weight:600;font-size:0.85rem;margin-top:4px;">
        Ooredoo Customers</div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px;">
        {ooredoo_share}% market share</div>
    </div>""", unsafe_allow_html=True)

    acq_c3.markdown(f"""
    <div style="background:#f0f4ff;border-radius:10px;padding:16px;
                border-left:5px solid {RAMI_COLOR};text-align:center;">
      <div style="font-size:2rem;font-weight:800;color:{RAMI_COLOR};">
        {competitor_customers:,}</div>
      <div style="font-weight:600;font-size:0.85rem;margin-top:4px;">
        Competitor Customers</div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px;">
        {competitor_share}% combined share</div>
    </div>""", unsafe_allow_html=True)

    acq_c4.markdown(f"""
    <div style="background:#fff8e8;border-radius:10px;padding:16px;
                border-left:5px solid #FFA500;text-align:center;">
      <div style="font-size:2rem;font-weight:800;color:#FFA500;">
        {remaining_addressable:,}</div>
      <div style="font-weight:600;font-size:0.85rem;margin-top:4px;">
        Remaining Addressable</div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px;">
        {unserved_share:.0f}% unserved market</div>
    </div>""", unsafe_allow_html=True)

    # ── Ooredoo revenue summary card ───────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#fff0f0;border-radius:10px;padding:16px 20px;
                border-left:5px solid {DON_COLOR};margin-top:14px;">
      <b style="font-size:1.0rem;color:{DON_COLOR};">
        📡 Ooredoo FTTH — Revenue Opportunity Summary</b><br><br>
      <b>{n_buildings}</b> buildings detected →
      <b>{total_market_potential:,}</b> estimated households
      ({HOUSEHOLDS_PER_BUILDING:.1f} avg. per building)<br>
      Ooredoo target customers: <b>{ooredoo_customers:,}</b>
        ({ooredoo_share}% share)<br>
      Estimated FTTH subscribers: <b>{ooredoo_ftth_subs:,}</b>
        ({FTTH_UPTAKE_RATE*100:.0f}% uptake rate)<br>
      Estimated annual revenue: <b>{annual_revenue_omr:,.0f} OMR/year</b>
        (at {ARPU_MONTHLY_OMR} OMR ARPU/month)
    </div>
    """, unsafe_allow_html=True)

    # ── Persist Zone Intelligence for downstream stages ────────────────────────
    zone_intel = ZoneIntelligence(
        source=getattr(uploaded, "name", "uploaded_image"),
        zone_id=None,
        zone_name="Midan Residence, Ghubrah",
        buildings_detected=n_buildings,
        cars_detected=n_cars,
        total_objects=res.count,
        market_potential_households=total_market_potential,
        population_density_per_km2=None,
        ftth_opportunity=priority_label,
        coverage_gap_pct=None,
        detection_mode=mode_label,
    ).to_dict()
    save_zone_intelligence(zone_intel)
    st.session_state["zone_intelligence"] = zone_intel
    st.success(
        "✅ Zone Intelligence saved — this output now feeds the "
        "Marketing Intelligence Engine and Executive Report stages.")

    # ── Row 8: top detections table ────────────────────────────────────────────
    st.markdown("---")
    if res.boxes:
        with st.expander("All detected objects (coordinates)"):
            rows = [
                {
                    "Class": b[4],
                    "Confidence": f"{b[5]:.2f}",
                    "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                    "Width px": b[2] - b[0],
                    "Height px": b[3] - b[1],
                }
                for b in sorted(res.boxes, key=lambda x: x[5], reverse=True)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         height=300, hide_index=True)


# ── Forecasting (workflow stage 5 — final) ──────────────────────────────────────

def render_forecasting_tab(results: dict, zones_data: dict):
    """
    Forecasting — position 5 (final stage) in the FALCON workflow.

    Uses historical telecom/traffic trends together with the Marketing
    Intelligence Engine's output (falling back to raw zone scoring if the
    Marketing Intelligence Engine hasn't run yet) to predict future market
    potential, customer growth, FTTH adoption opportunities, and expansion
    priorities. Kept separate from the Executive Report.

    Sections:
      1. Telecom market simulation (5G / FTTH / mobile KPIs) — historical baseline
      2. Zone clustering
      3. Demand prediction scores vs YOLO scores
      4. Revenue forecast comparison
      5. Traffic trend projection
    """
    import json
    from app.forecaster import (forecast_revenue, market_share_projection,
                                 predict_demand_score, traffic_trend)

    st.markdown('<div class="oo-section-title">📈 Forecasting — Market & Growth Projections</div>',
                unsafe_allow_html=True)

    mi_snapshot    = st.session_state.get("marketing_intelligence") or \
                      (load_marketing_intelligence() or {}).get("zones")
    scored         = mi_snapshot if mi_snapshot else results["scored_zones"]
    zones_json_str = json.dumps(zones_data)

    st.caption(
        "Combines historical telecom/traffic trends with Marketing "
        "Intelligence Engine outputs to project future market potential, "
        "customer growth, and FTTH expansion priorities. All data is simulated.")

    # ── Section 1 — Telecom simulation KPIs ──────────────────────────────────
    st.markdown("#### Telecom Market Simulation")
    sim_data = _load_telecom_sim(zones_json_str)
    totals   = sim_data.get("market_totals", {})

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mobile Subscribers",    f"{totals.get('total_mobile_subscribers', 0):,}")
    k2.metric("FTTH Subscribers",      f"{totals.get('total_ftth_subscribers', 0):,}")
    k3.metric("5G Subscribers",        f"{totals.get('total_5g_subscribers', 0):,}")
    k4.metric("Monthly Revenue (OMR)", f"{totals.get('total_monthly_revenue_omr', 0):,.0f}")
    k5.metric("Annual Revenue (OMR)",  f"{totals.get('total_annual_revenue_omr', 0):,.0f}")

    profiles = sim_data.get("zone_profiles", {})
    if profiles:
        tele_rows = []
        for zid, p in profiles.items():
            tele_rows.append({
                "Zone":           p["zone_name"],
                "Mobile Subs":    p["mobile_subscribers"],
                "FTTH Subs":      p["ftth_subscribers"],
                "5G Subs":        p["subscribers_5g"],
                "5G Penetration": f"{p['penetration_5g_pct']}%",
                "ARPU (OMR)":     p["arpu_blended"],
                "Monthly Rev.":   f"{p['monthly_revenue_omr']:,.0f}",
                "Avg Data (GB)":  p["avg_data_gb_per_user_month"],
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
    cluster_df   = _load_clusters(scored_tuple, zones_json_str, n_clusters)

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
            assign_df = cluster_df[["zone_name", "cluster_label", "priority_score"]].copy()
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
    zone_meta   = {z["id"]: z for z in zones_data["zones"]}
    for s in scored:
        z = zone_meta.get(s["zone_id"], {})
        demand_rows.append({
            "Zone":         s["zone_name"],
            "YOLO Score":   s["priority_score"],
            "Demand Score": predict_demand_score(z),
            "Delta":        round(s["priority_score"] - predict_demand_score(z), 1),
            "Tier":         s["tier"],
        })
    demand_df = pd.DataFrame(demand_rows).sort_values("YOLO Score", ascending=False)

    fig_comp = px.scatter(
        demand_df,
        x="Demand Score", y="YOLO Score",
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
        height=360, margin=dict(t=10),
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
        fillcolor="rgba(2,195,154,0.15)",
        name="Ooredoo Market Share",
    ))
    fig_ms.add_hline(y=ms_target, line_dash="dash",
                     line_color="gray",
                     annotation_text=f"Target {ms_target}%")
    fig_ms.update_layout(
        xaxis_title="Year", yaxis_title="Market Share (%)",
        height=300, margin=dict(t=10),
    )
    st.plotly_chart(fig_ms, use_container_width=True)

    # ── Section 6 — Traffic trend ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Network Traffic Growth Trend (35 % YoY)")
    st.caption("ITU benchmark: data consumption grows ~35 % per year in GCC markets.")

    if profiles:
        from app.forecaster import traffic_trend
        top_profiles = list(profiles.values())[:6]
        trend_rows   = traffic_trend(top_profiles, years=5)
        trend_df     = pd.DataFrame(trend_rows)
        fig_traf     = px.line(
            trend_df, x="year", y="peak_gbps", color="zone_name",
            markers=True,
            labels={"year": "Year", "peak_gbps": "Peak Traffic (Gbps)",
                    "zone_name": "Zone"},
        )
        fig_traf.update_layout(
            height=320, margin=dict(t=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_traf, use_container_width=True)


# ── Marketing Intelligence Engine (workflow stage 4) ────────────────────────────

def render_marketing_intelligence_tab(results: dict, zones_data: dict):
    """Marketing Intelligence Engine — position 4 in the FALCON workflow.

    Consumes Zone Intelligence (Computer Vision Engine output, when available)
    plus zone scoring to produce market potential assessment, customer
    opportunity identification, competitive positioning analysis, FTTH
    expansion recommendations, and target-area prioritization. Results are
    persisted for the Executive Report and Forecasting stages.
    """
    st.markdown(
        '<div class="oo-section-title">📊 Marketing Intelligence Engine</div>',
        unsafe_allow_html=True)

    scored     = results["scored_zones"]
    zone_intel = st.session_state.get("zone_intelligence") or latest_zone_intelligence()

    if zone_intel:
        st.caption(
            f"🔬 Refined with Zone Intelligence from the Computer Vision Engine — "
            f"**{zone_intel.get('buildings_detected', 0)} buildings** detected → "
            f"**{zone_intel.get('market_potential_households', 0):,}** household potential.")
    else:
        st.caption(
            "No Computer Vision Engine output yet — estimates below use zone "
            "population data. Upload an aerial image in the Computer Vision "
            "Engine tab to refine them with detected buildings.")

    cv_households = zone_intel["market_potential_households"] if zone_intel else 0

    # ── Build the Marketing Intelligence record for every zone ──────────────────
    mi_zones = []
    for s in scored:
        households = int(s["population"] / HOUSEHOLD_SIZE)
        if cv_households and s["tier"] == "High Priority":
            households = max(households, cv_households)
        target_customers = int(households * DON_MARKET_SHARE / 100)
        ftth_subs         = int(target_customers * FTTH_UPTAKE_RATE)
        annual_revenue    = ftth_subs * ARPU_MONTHLY_OMR * 12
        mi_zones.append({
            **s,
            "market_potential_households": households,
            "target_customers": target_customers,
            "ftth_subscribers": ftth_subs,
            "annual_revenue_omr": annual_revenue,
            "expansion_recommendation": (
                "Immediate deployment" if s["tier"] == "High Priority" else
                "Plan within 6 months"  if s["tier"] == "Medium Priority" else
                "Phase 2 expansion"),
            "competitive_notes": (
                f"{s['coverage_gap_pct']:.0f}% coverage gap vs. "
                f"{RAMI_MARKET_SHARE:.0f}% {RAMI_BRAND_NAME} / "
                f"{WAHEB_MARKET_SHARE:.0f}% {WAHEB_BRAND_NAME} share"),
        })
    mi_zones.sort(key=lambda z: z["priority_score"], reverse=True)
    for rank, z in enumerate(mi_zones, 1):
        z["target_priority_rank"] = rank

    save_marketing_intelligence(mi_zones)
    st.session_state["marketing_intelligence"] = mi_zones

    total_hh   = sum(z["market_potential_households"] for z in mi_zones)
    total_tc   = sum(z["target_customers"] for z in mi_zones)
    total_subs = sum(z["ftth_subscribers"] for z in mi_zones)
    total_rev  = sum(z["annual_revenue_omr"] for z in mi_zones)

    # ── 1. Market Potential Assessment ───────────────────────────────────────────
    st.markdown("#### 1. Market Potential Assessment")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Addressable Households",    f"{total_hh:,}")
    k2.metric("Ooredoo Target Customers",   f"{total_tc:,}")
    k3.metric("Est. FTTH Subscribers",      f"{total_subs:,}")
    k4.metric("Est. Annual Revenue",        f"{total_rev:,.0f} OMR")

    st.markdown("---")

    # ── 2. Customer Opportunity Identification ───────────────────────────────────
    st.markdown("#### 2. Customer Opportunity Identification")
    opp_df = pd.DataFrame([{
        "Zone": z["zone_name"], "Tier": z["tier"],
        "Households": z["market_potential_households"],
        "Target Customers": z["target_customers"],
        "FTTH Subscribers": z["ftth_subscribers"],
        "Annual Revenue (OMR)": f"{z['annual_revenue_omr']:,.0f}",
    } for z in mi_zones[:10]])
    st.dataframe(opp_df, use_container_width=True, hide_index=True, height=280)

    st.markdown("---")

    # ── 3. Competitive Positioning Analysis ──────────────────────────────────────
    st.markdown("#### 3. Competitive Positioning Analysis")
    comp_left, comp_right = st.columns([1, 1])
    with comp_left:
        share_df = pd.DataFrame({
            "Operator": [DON_BRAND_NAME, RAMI_BRAND_NAME, WAHEB_BRAND_NAME, OTHERS_BRAND_NAME],
            "Share (%)": [DON_MARKET_SHARE, RAMI_MARKET_SHARE,
                          WAHEB_MARKET_SHARE, UNSERVED_MARKET_SHARE],
        })
        fig_comp = px.pie(
            share_df, values="Share (%)", names="Operator", color="Operator",
            color_discrete_map={DON_BRAND_NAME: DON_COLOR, RAMI_BRAND_NAME: RAMI_COLOR,
                                 WAHEB_BRAND_NAME: WAHEB_COLOR, OTHERS_BRAND_NAME: "#aaaaaa"},
            hole=0.42,
        )
        fig_comp.update_layout(height=260, margin=dict(t=10))
        st.plotly_chart(fig_comp, use_container_width=True)
    with comp_right:
        st.markdown("**Zone-level competitive notes**")
        for z in mi_zones[:5]:
            st.markdown(f"- **{z['zone_name']}** ({z['tier']}): {z['competitive_notes']}")

    st.markdown("---")

    # ── 4. FTTH Expansion Recommendations ────────────────────────────────────────
    st.markdown("#### 4. FTTH Expansion Recommendations")
    high = [z for z in mi_zones if z["tier"] == "High Priority"]
    med  = [z for z in mi_zones if z["tier"] == "Medium Priority"]
    low  = [z for z in mi_zones if z["tier"] == "Low Priority"]
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:16px 20px;
                box-shadow:0 4px 12px rgba(0,0,0,0.08);">
      <b>Phase 1 — Immediate ({len(high)} zones):</b>
        {', '.join(z['zone_name'] for z in high[:5]) or 'None'}<br>
      <b>Phase 2 — 6-Month Plan ({len(med)} zones):</b>
        {', '.join(z['zone_name'] for z in med[:5]) or 'None'}<br>
      <b>Phase 3 — Long-Term ({len(low)} zones):</b>
        {', '.join(z['zone_name'] for z in low[:5]) or 'None'}
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 5. Target Area Prioritization ────────────────────────────────────────────
    st.markdown("#### 5. Target Area Prioritization")
    rank_df = pd.DataFrame([{
        "Rank": z["target_priority_rank"], "Zone": z["zone_name"],
        "Tier": z["tier"], "Priority Score": z["priority_score"],
        "Recommendation": z["expansion_recommendation"],
    } for z in mi_zones])
    st.dataframe(rank_df, use_container_width=True, hide_index=True, height=320)

    st.success(
        "✅ Marketing Intelligence saved — feeds the Executive Report and "
        "Forecasting stages.")

    # ── Zone deep-dive — AI narrative + go-to-market plan ────────────────────────
    st.markdown("---")
    with st.expander("🔎 Zone Deep-Dive — AI Narrative & Go-to-Market Plan"):
        zone_names    = [z["zone_name"] for z in mi_zones]
        selected_name = st.selectbox("Select Zone", zone_names, key="mi_deep_dive_zone")
        selected      = next(z for z in mi_zones if z["zone_name"] == selected_name)

        if st.button("Generate Deep-Dive", key="mi_deep_dive_btn"):
            tier_color = TIER_COLORS.get(selected["tier"], "#fd7e14")

            def _mi_card(title: str, body: str, accent: str = "#1A1A2E"):
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:0;
                            box-shadow:0 4px 16px rgba(0,0,0,0.08);margin-bottom:16px;overflow:hidden;">
                  <div style="background:{accent};padding:12px 20px;">
                    <span style="color:white;font-weight:700;font-size:1.0rem;">{title}</span>
                  </div>
                  <div style="padding:16px 20px;font-size:0.88rem;color:#333;line-height:1.65;">
                    {body}
                  </div>
                </div>""", unsafe_allow_html=True)

            _mi_card("📍 Area Overview",
                f"<b>Zone:</b> {selected['zone_name']}, Muscat, Oman<br>"
                f"<b>Population:</b> {selected['population']:,} residents<br>"
                f"<b>Priority Score:</b> {selected['priority_score']:.1f} / 100<br>"
                f"<b>Tier:</b> <span style='color:{tier_color};font-weight:700;'>"
                f"{selected['tier']}</span><br>"
                f"<b>Coverage Gap:</b> {selected['coverage_gap_pct']:.0f}% of zone "
                f"currently unserved by FTTH")

            _mi_card("👥 Customer Opportunity",
                f"<b>Market Potential:</b> {selected['market_potential_households']:,} households<br>"
                f"<b>Target Customers:</b> {selected['target_customers']:,}<br>"
                f"<b>Est. FTTH Subscribers:</b> {selected['ftth_subscribers']:,}<br>"
                f"<b>Est. Annual Revenue:</b> {selected['annual_revenue_omr']:,.0f} OMR/year")

            _mi_card("⚔️ Competitive Positioning",
                f"{selected['competitive_notes']}<br>"
                f"{RAMI_BRAND_NAME} {RAMI_MARKET_SHARE:.0f}% | "
                f"{WAHEB_BRAND_NAME} {WAHEB_MARKET_SHARE:.0f}% | "
                f"{DON_BRAND_NAME} {DON_MARKET_SHARE:.0f}% | "
                f"Unserved {UNSERVED_MARKET_SHARE:.0f}%")

            _mi_card("📋 FTTH Expansion Recommendation",
                f"<b>{selected['expansion_recommendation']}</b><br>"
                f"Estimated Investment: {selected['recommended_investment_omr']:,} OMR<br>"
                f"Expected ROI Payback: {selected['roi_years']:.1f} years",
                accent="#065A82")

            hh            = selected["market_potential_households"]
            sales_team_sz = max(3, int(hh / 500))
            budget_omr    = int(hh * 8)
            _mi_card("🎯 Go-To-Market Plan",
                "<b>Target Segments:</b><br>"
                f"&bull; Families: ~{int(hh*0.45):,} households<br>"
                f"&bull; High-income HH: ~{int(hh*0.20):,} households<br>"
                f"&bull; SMEs: ~{int(hh*0.10):,} businesses<br>"
                f"&bull; New Developments: ~{int(hh*0.15):,} units<br><br>"
                f"<b>Sales Team Size:</b> {sales_team_sz} agents<br>"
                f"<b>Campaign Budget:</b> ~{budget_omr:,} OMR<br>"
                "<b>Channels:</b> SMS, digital, door-to-door, retail, social media",
                accent="#28a745")



# ── Executive Report (workflow stage 3) ─────────────────────────────────────────

def render_executive_report_tab(results: dict, zones_data: dict):
    """Executive Report — position 3 in the FALCON workflow.

    Synthesizes only Marketing Intelligence Engine outputs into a
    business-facing report (market potential, competition, financial
    opportunity, rollout, risk). No raw Computer Vision detections or
    technical model details are included.
    """
    st.markdown(
        '<div class="oo-section-title">📋 Executive Report — Ooredoo FTTH Expansion</div>',
        unsafe_allow_html=True)

    mi_data = st.session_state.get("marketing_intelligence") or \
              (load_marketing_intelligence() or {}).get("zones")
    scored  = mi_data if mi_data else results["scored_zones"]
    if not mi_data:
        st.info(
            "Showing zone-scoring estimates. Visit the Marketing Intelligence "
            "Engine tab to generate refined figures for this report.")

    zones    = zones_data["zones"]
    n_zones  = len(scored)
    high     = [s for s in scored if s["tier"] == "High Priority"]
    med      = [s for s in scored if s["tier"] == "Medium Priority"]
    low      = [s for s in scored if s["tier"] == "Low Priority"]
    top_zone = max(scored, key=lambda x: x["priority_score"], default={})

    total_inv     = sum(s["recommended_investment_omr"] for s in high)
    avg_gap       = sum(s["coverage_gap_pct"] for s in scored) / max(n_zones, 1)
    total_pop     = sum(s.get("population", 0) for s in scored)
    total_hh      = int(total_pop / 3.5)
    ooredoo_hh    = int(total_hh * DON_MARKET_SHARE / 100)
    omantel_hh    = int(total_hh * RAMI_MARKET_SHARE / 100)
    awasr_hh      = int(total_hh * WAHEB_MARKET_SHARE / 100)
    others_hh     = int(total_hh * UNSERVED_MARKET_SHARE / 100)
    annual_rev_hi = sum((s.get("population", 0) / HOUSEHOLD_SIZE) *
                        FTTH_UPTAKE_RATE * ARPU_MONTHLY_OMR * 12 for s in high)
    roi_avg       = sum(s["roi_years"] for s in high) / max(len(high), 1)

    st.markdown(f"""
    <div style="background:#f8f9fa;border-radius:10px;padding:14px 20px;
                margin-bottom:20px;font-size:0.82rem;color:#666;">
      <b>Ooredoo Falcon Intelligence Platform</b> &bull;
      Muscat, Oman &bull; 2026 &bull; Version 2.0 &bull; CONFIDENTIAL
    </div>""", unsafe_allow_html=True)

    def _report_card(title: str, body: str, accent: str = "#1A1A2E"):
        st.markdown(f"""
        <div style="background:white;border-radius:12px;margin-bottom:16px;
                    box-shadow:0 4px 16px rgba(0,0,0,0.07);overflow:hidden;">
          <div style="background:{accent};padding:10px 20px;">
            <span style="color:white;font-weight:700;">{title}</span>
          </div>
          <div style="padding:16px 20px;font-size:0.88rem;color:#333;line-height:1.7;">
            {body}
          </div>
        </div>""", unsafe_allow_html=True)

    _report_card("1. Executive Summary",
        f"Ooredoo Falcon analyzed <b>{n_zones} zones</b> across Muscat, Oman using "
        f"Marketing Intelligence scoring combining market potential, FTTH coverage gaps, "
        f"and population data. "
        f"<b>{len(high)} High Priority</b> zones identified for immediate fiber deployment, "
        f"<b>{len(med)} Medium Priority</b> for 6-month planning, and "
        f"<b>{len(low)} Low Priority</b> for Phase 2. "
        f"Top-ranked zone: <b>{top_zone.get('zone_name', 'N/A')}</b> "
        f"(score: {top_zone.get('priority_score', 0):.1f}/100). "
        f"Estimated total investment for High Priority zones: <b>{total_inv:,.0f} OMR</b>.")

    _report_card("2. Geographic Analysis",
        f"<b>Total Zones Analyzed:</b> {n_zones} across Muscat Governorate<br>"
        f"<b>High Priority Zones:</b> {len(high)} — immediate deployment<br>"
        f"<b>Medium Priority Zones:</b> {len(med)} — 6-month pipeline<br>"
        f"<b>Low Priority Zones:</b> {len(low)} — Phase 2 expansion<br>"
        f"<b>Average Coverage Gap:</b> {avg_gap:.1f}% across all zones<br>"
        f"<b>Total Population Served:</b> {total_pop:,} residents")

    _report_card("3. Market Potential",
        f"<b>Total Addressable Households:</b> {total_hh:,}<br>"
        f"<b>Ooredoo Target ({DON_MARKET_SHARE:.0f}%):</b> {ooredoo_hh:,} households<br>"
        f"<b>Omantel ({RAMI_MARKET_SHARE:.0f}%):</b> {omantel_hh:,} households<br>"
        f"<b>Awasr ({WAHEB_MARKET_SHARE:.0f}%):</b> {awasr_hh:,} households<br>"
        f"<b>Others ({UNSERVED_MARKET_SHARE:.0f}%):</b> {others_hh:,} households")

    _report_card("4. Competition Analysis",
        f"<b>Omantel:</b> {RAMI_MARKET_SHARE:.0f}% — dominant incumbent with legacy DSL converting to fiber<br>"
        f"<b>Awasr:</b> {WAHEB_MARKET_SHARE:.0f}% — aggressive pricing, growing residential footprint<br>"
        f"<b>Ooredoo:</b> {DON_MARKET_SHARE:.0f}% — strong brand, mobile bundle advantage, expansion headroom<br>"
        f"<b>Others / Unserved:</b> {UNSERVED_MARKET_SHARE:.0f}% — addressable greenfield opportunity<br>"
        "Strategic recommendation: Prioritize zones with <40% Omantel presence for first-mover advantage.")

    _report_card("5. Financial Opportunity",
        f"<b>High Priority Zone Investment:</b> {total_inv:,.0f} OMR<br>"
        f"<b>Expected Annual Revenue (High Priority):</b> {annual_rev_hi:,.0f} OMR/year<br>"
        f"<b>Average ROI Payback:</b> {roi_avg:.1f} years<br>"
        f"<b>5-Year Revenue Projection:</b> {annual_rev_hi * 5:,.0f} OMR (flat) / "
        f"{int(annual_rev_hi * ((1.08**5 - 1) / 0.08)):,.0f} OMR (8% CAGR)<br>"
        f"<b>ARPU Assumption:</b> {ARPU_MONTHLY_OMR} OMR/month | "
        f"<b>Uptake Rate:</b> {FTTH_UPTAKE_RATE*100:.0f}%",
        accent="#28a745")

    _report_card("6. Recommended Rollout",
        f"<b>Phase 1 (Q1 2026) — High Priority:</b> "
        f"{', '.join(s['zone_name'] for s in high[:3]) or 'N/A'} — "
        f"Investment: {total_inv:,.0f} OMR<br>"
        f"<b>Phase 2 (Q2–Q3 2026) — Medium Priority:</b> "
        f"{len(med)} zones, planned after Phase 1 ROI confirmation<br>"
        f"<b>Phase 3 (Q4 2026+) — Low Priority:</b> "
        f"{len(low)} zones, dependent on market conditions")

    _report_card("7. Risk Assessment",
        "<b>Risk 1 — Competitive Response:</b> Omantel/Awasr may accelerate rollout in targeted zones. "
        "<i>Mitigation: Move quickly in High Priority zones; lock in multi-year customer contracts.</i><br>"
        "<b>Risk 2 — Capital Constraints:</b> High upfront CAPEX. "
        "<i>Mitigation: Phased rollout aligned with ROI validation; consider developer partnerships for MDUs.</i><br>"
        "<b>Risk 3 — Adoption Rate:</b> FTTH uptake may lag projections. "
        "<i>Mitigation: Bundle promotions with Ooredoo mobile; price-match Awasr starter packages.</i>",
        accent="#dc3545")

    _report_card("8. Final Decision Score",
        f"Based on AI scoring across {n_zones} zones, Ooredoo Falcon recommends "
        f"<b>immediate action</b> in {len(high)} High Priority zones. "
        f"Combined opportunity: <b>{total_inv:,.0f} OMR investment</b> generating "
        f"<b>{annual_rev_hi:,.0f} OMR/year</b> with average payback of <b>{roi_avg:.1f} years</b>. "
        "Overall expansion confidence: <b>HIGH</b>.")

    st.markdown("---")
    st.markdown("#### Download Reports")
    dl_c1, dl_c2, dl_c3 = st.columns(3)

    with dl_c1:
        pptx_script = Path(__file__).parent.parent / "scripts" / "generate_pptx.py"
        if pptx_script.exists():
            if st.button("📥 Download PowerPoint", use_container_width=True, key="exec_pptx"):
                import subprocess
                r = subprocess.run([sys.executable, str(pptx_script)],
                                   capture_output=True, text=True, timeout=90)
                if r.returncode == 0:
                    pptx_path = (Path(__file__).parent.parent / "outputs"
                                 / "ooredoo_falcon_presentation.pptx")
                    if pptx_path.exists():
                        st.download_button(
                            "⬇ Save PPTX",
                            data=pptx_path.read_bytes(),
                            file_name="ooredoo_falcon_presentation.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        )
                    else:
                        st.success("Presentation generated in outputs/ folder.")
                else:
                    st.error(f"Error: {r.stderr[:200]}")

    with dl_c2:
        csv_data = export_csv(scored, zones_data)
        st.download_button(
            "📥 Download CSV Report",
            data=csv_data,
            file_name="ooredoo_falcon_muscat_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_c3:
        summary = generate_executive_summary(scored, zones_data)
        st.download_button(
            "📥 Executive Summary (.txt)",
            data=summary,
            file_name="ooredoo_falcon_executive_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "landing"
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Inject CSS on every page
    st.markdown(OOREDOO_CSS, unsafe_allow_html=True)

    if st.session_state.page == "landing":
        render_landing_page()

    elif st.session_state.page == "login":
        render_login_page()

    elif st.session_state.page == "dashboard" and st.session_state.authenticated:
        # Load data
        zones_data = _load_zones()
        results    = _load_results(DEMO_MODE)

        if "filtered_zones" not in st.session_state:
            st.session_state["filtered_zones"] = results["scored_zones"]

        with st.sidebar:
            render_sidebar(results, zones_data)

        render_app_header()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️  Coverage Map",
            "🔬  Computer Vision Engine",
            "📋  Executive Report",
            "📊  Marketing Intelligence Engine",
            "📈  Forecasting",
        ])

        with tab1:
            render_map_tab(zones_data, results)
        with tab2:
            render_cv_engine_tab(results, zones_data)
        with tab3:
            render_executive_report_tab(results, zones_data)
        with tab4:
            render_marketing_intelligence_tab(results, zones_data)
        with tab5:
            render_forecasting_tab(results, zones_data)

    else:
        # Not authenticated or unknown page — redirect to login
        st.session_state.page = "login"
        st.rerun()


if __name__ == "__main__":
    main()
