"""FALCON — Enhanced Folium map builder with competitive network layers."""
from __future__ import annotations

from pathlib import Path

import folium
from folium.plugins import Fullscreen, HeatMap

from app.config import (DON_COLOR, OUTPUTS_DIR, RAMI_COLOR, TIER_FOLIUM_COLORS,
                         WAHEB_COLOR, WAHEB_MARKET_SHARE, RAMI_MARKET_SHARE,
                         DON_MARKET_SHARE, UNSERVED_MARKET_SHARE)

# ── Popup templates ───────────────────────────────────────────────────────────

def _zone_popup(zone: dict, score_info: dict) -> str:
    tier   = score_info.get("tier", "Low Priority")
    score  = score_info.get("priority_score", 0)
    vehicles = score_info.get("vehicle_count", 0)
    gap    = score_info.get("coverage_gap_pct", 0)
    inv    = score_info.get("recommended_investment_omr", 0)
    roi    = score_info.get("roi_years", 0)
    tc = {"High Priority": "#dc3545",
          "Medium Priority": "#fd7e14",
          "Low Priority": "#28a745"}.get(tier, "#555")
    return f"""
    <div style="font-family:Arial,sans-serif;min-width:220px;font-size:13px;">
      <div style="background:#1a1a2e;color:white;padding:8px 12px;
                  border-radius:6px 6px 0 0;margin-bottom:8px;">
        <b style="font-size:14px;">{zone['name']}</b>
      </div>
      <table style="width:100%;border-collapse:collapse;">
        <tr><td><b>Priority Score</b></td>
            <td style="text-align:right;font-weight:bold;color:{tc};">
            {score:.1f}/100</td></tr>
        <tr><td><b>Tier</b></td>
            <td style="text-align:right;color:{tc};font-weight:bold;">{tier}</td></tr>
        <tr><td><b>Vehicles</b></td>
            <td style="text-align:right;">{vehicles:,}</td></tr>
        <tr><td><b>Population</b></td>
            <td style="text-align:right;">{zone['population']:,}</td></tr>
        <tr><td><b>FTTH Coverage</b></td>
            <td style="text-align:right;">{zone['ftth_coverage_pct']}%</td></tr>
        <tr><td><b>Coverage Gap</b></td>
            <td style="text-align:right;color:#dc3545;">{gap:.0f}%</td></tr>
        <tr><td><b>Investment</b></td>
            <td style="text-align:right;">{inv:,} OMR</td></tr>
        <tr><td><b>ROI Payback</b></td>
            <td style="text-align:right;">{roi:.1f} yrs</td></tr>
      </table>
    </div>"""


def _asset_popup(asset: dict) -> str:
    color_map = {"WAH": WAHEB_COLOR, "RAMI": RAMI_COLOR, "DON": DON_COLOR}
    c = color_map.get(asset["company"][:3], "#555")
    cap = (f"<br><b>Capacity used:</b> {asset['capacity_used_pct']}%"
           if asset["type"] in ("cabinet", "exchange") else "")
    return (f"<div style='font-family:Arial;font-size:12px;'>"
            f"<b style='color:{c};'>{asset['company']}</b> | "
            f"{asset['type'].title()}<br>"
            f"<b>ID:</b> {asset['asset_id']}<br>"
            f"<b>Zone:</b> {asset['zone']}<br>"
            f"<b>Status:</b> {asset['status']}"
            f"{cap}<br>"
            f"<b>Homes:</b> {asset['connected_homes']:,}<br>"
            f"<b>Installed:</b> {asset['install_date']}"
            f"</div>")


# ── Layer builders ────────────────────────────────────────────────────────────

def _add_company_layer(m: folium.Map, assets: list[dict], company: str,
                        color: str, emoji: str) -> folium.FeatureGroup:
    """Add all markers + coverage circles for one company."""
    layer = folium.FeatureGroup(name=f"{emoji} {company} Network", show=True)

    cov_layer = folium.FeatureGroup(
        name=f"{emoji} {company} Coverage", show=False)

    for asset in assets:
        lat, lng = asset["lat"], asset["lng"]
        atype = asset["type"]

        if atype == "exchange":
            radius = 14
        elif atype == "cabinet":
            radius = 7
        else:
            radius = 3

        if atype == "home":
            continue  # homes added to separate layer

        tip = f"{company} | {atype.title()} | {asset['status']}"
        folium.CircleMarker(
            location=[lat, lng],
            radius=radius,
            popup=folium.Popup(_asset_popup(asset), max_width=240),
            tooltip=tip,
            color="white",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.90,
        ).add_to(layer)

        # Coverage circle
        cov_radius_m = int(
            {"exchange": 3000, "cabinet": 800}.get(atype, 200))
        opacity = 0.10 if atype == "exchange" else 0.20
        folium.Circle(
            location=[lat, lng],
            radius=cov_radius_m,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
        ).add_to(cov_layer)

    layer.add_to(m)
    cov_layer.add_to(m)
    return layer


def _add_home_layer(m: folium.Map, waheb: list, rami: list,
                     don: list) -> None:
    layer = folium.FeatureGroup(name="Home Connections", show=False)
    for assets, color in [(waheb, WAHEB_COLOR), (rami, RAMI_COLOR),
                           (don, DON_COLOR)]:
        for a in assets:
            if a["type"] != "home":
                continue
            folium.CircleMarker(
                location=[a["lat"], a["lng"]],
                radius=2,
                tooltip=a["asset_id"],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=0,
            ).add_to(layer)
    layer.add_to(m)


def _add_opportunity_layer(m: folium.Map, zones_data: dict,
                             market_summary: dict) -> None:
    opp_names = set(market_summary.get("waheb_opportunity_zones", []))
    opp_details = {o["zone"]: o for o in
                   market_summary.get("opportunity_zone_details", [])}
    zone_lookup = {z["name"]: z for z in zones_data["zones"]}

    layer = folium.FeatureGroup(name="Opportunity Zones", show=True)

    for zname in opp_names:
        zone = zone_lookup.get(zname)
        if not zone:
            continue
        od = opp_details.get(zname, {})
        rev = od.get("est_annual_revenue_omr", 0)
        radius_m = int(zone["area_km2"] ** 0.5 * 900)

        popup_html = (
            f"<div style='font-family:Arial;font-size:12px;'>"
            f"<b style='color:#FFD700;'>OPPORTUNITY ZONE</b><br>"
            f"<b>{zname}</b><br>"
            f"Rev. potential: <b>{rev:,} OMR/yr</b><br>"
            f"Competitor cabinets: {od.get('competitor_cabinets', '?')}<br>"
            f"Waheb cabinets: {od.get('waheb_cabinets', '?')}"
            f"</div>")

        folium.Circle(
            location=[zone["lat"], zone["lon"]],
            radius=radius_m,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"Opportunity: {zname}",
            color="#FFD700",
            weight=2,
            fill=True,
            fill_color="#FFD700",
            fill_opacity=0.25,
        ).add_to(layer)

    layer.add_to(m)


def _add_zone_markers(m: folium.Map, zones_data: dict,
                       scored_zones: list) -> None:
    score_lookup = {s["zone_id"]: s for s in scored_zones}
    zone_lookup  = {z["id"]:      z for z in zones_data["zones"]}
    layer = folium.FeatureGroup(name="Zone Priority Scores", show=True)

    for zid, zone in zone_lookup.items():
        si    = score_lookup.get(zid, {})
        tier  = si.get("tier", "Low Priority")
        score = si.get("priority_score", 0)
        color = TIER_FOLIUM_COLORS.get(tier, "blue")

        folium.CircleMarker(
            location=[zone["lat"], zone["lon"]],
            radius=max(8, min(18, int(score / 6))),
            popup=folium.Popup(_zone_popup(zone, si), max_width=280),
            tooltip=f"{zone['name']} | Score: {score:.1f} | {tier}",
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
        ).add_to(layer)

    layer.add_to(m)


# ── Legend & info panels ──────────────────────────────────────────────────────

_MARKET_LEGEND = f"""
<div style="position:fixed;bottom:30px;right:30px;z-index:1000;
            background:white;padding:14px 18px;border-radius:10px;
            box-shadow:0 2px 10px rgba(0,0,0,.3);font-family:Arial;
            font-size:12px;min-width:180px;">
  <b style="font-size:13px;">Market Share</b><br>
  <hr style="margin:6px 0;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
    <div style="width:13px;height:13px;border-radius:50%;
                background:{WAHEB_COLOR};"></div>
    <span>Waheb: {WAHEB_MARKET_SHARE:.0f}%</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
    <div style="width:13px;height:13px;border-radius:50%;
                background:{RAMI_COLOR};"></div>
    <span>RAMI: {RAMI_MARKET_SHARE:.0f}%</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
    <div style="width:13px;height:13px;border-radius:50%;
                background:{DON_COLOR};"></div>
    <span>Don: {DON_MARKET_SHARE:.0f}%</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="width:13px;height:13px;border-radius:50%;
                background:#aaa;"></div>
    <span>Unserved: {UNSERVED_MARKET_SHARE:.0f}%</span>
  </div>
</div>
"""


def _info_panel_html(waheb: list, rami: list, don: list) -> str:
    def cab(assets):
        return sum(1 for a in assets if a["type"] == "cabinet")
    def exch(assets):
        return sum(1 for a in assets if a["type"] == "exchange")
    return f"""
<div style="position:fixed;top:80px;right:10px;z-index:1000;
            background:white;padding:10px 14px;border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,.25);font-family:Arial;
            font-size:12px;line-height:1.7;">
  <b>FTTH Asset Count</b><br>
  <span style="color:{WAHEB_COLOR};font-weight:700;">
    Waheb: {cab(waheb)} cab | {exch(waheb)} exch</span><br>
  <span style="color:{RAMI_COLOR};font-weight:700;">
    RAMI:  {cab(rami)} cab | {exch(rami)} exch</span><br>
  <span style="color:{DON_COLOR};font-weight:700;">
    Don:   {cab(don)} cab | {exch(don)} exch</span>
</div>
"""


# ── Main public function ──────────────────────────────────────────────────────

def build_map(zones_data: dict, scored_zones: list,
              network_data: dict | None = None) -> folium.Map:
    """
    Build Folium map.  When network_data is provided the map shows all
    competitive layers; otherwise falls back to zone-only view.
    """
    m = folium.Map(
        location=[23.5880, 58.4800],
        zoom_start=12,
        tiles="OpenStreetMap",
    )
    Fullscreen(position="topleft").add_to(m)

    if network_data:
        waheb = network_data.get("waheb", [])
        rami  = network_data.get("rami",  [])
        don   = network_data.get("don",   [])
        ms    = network_data.get("market_summary", {})

        _add_company_layer(m, waheb, "Waheb", WAHEB_COLOR, "🟢")
        _add_company_layer(m, rami,  "RAMI",  RAMI_COLOR,  "🔵")
        _add_company_layer(m, don,   "Don",   DON_COLOR,   "🔴")
        _add_home_layer(m, waheb, rami, don)
        if ms:
            _add_opportunity_layer(m, zones_data, ms)

        m.get_root().html.add_child(folium.Element(_MARKET_LEGEND))
        m.get_root().html.add_child(
            folium.Element(_info_panel_html(waheb, rami, don)))
    else:
        # Fallback: heatmap only
        score_lookup = {s["zone_id"]: s for s in scored_zones}
        heat_data = [
            [z["lat"], z["lon"],
             score_lookup.get(z["id"], {}).get("priority_score", 0) / 100]
            for z in zones_data["zones"]
        ]
        heat_layer = folium.FeatureGroup(name="Priority Heatmap", show=False)
        HeatMap(heat_data, radius=45, blur=35, min_opacity=0.3).add_to(
            heat_layer)
        heat_layer.add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:12px 16px;border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,.3);font-family:Arial;
                    font-size:12px;">
          <b>Priority Tier</b><br><br>
          <span style="color:#dc3545;">&#9679;</span> High<br>
          <span style="color:#fd7e14;">&#9679;</span> Medium<br>
          <span style="color:#28a745;">&#9679;</span> Low
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    _add_zone_markers(m, zones_data, scored_zones)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def save_competitive_map(zones_data: dict, scored_zones: list,
                          network_data: dict,
                          output_path: str | Path | None = None) -> Path:
    if output_path is None:
        output_path = OUTPUTS_DIR / "falcon_competitive_map.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m = build_map(zones_data, scored_zones, network_data)
    m.save(str(output_path))
    return output_path
