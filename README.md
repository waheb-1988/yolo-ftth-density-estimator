# FALCON — Flavia AI Coverage & Location Opportunity Network

> AI-powered competitive intelligence for Waheb Telecom

FALCON uses YOLOv8 to detect vehicles in aerial imagery, scores 15 Muscat city
zones by demand density, and maps the full telecom competitive landscape
(Waheb vs RAMI vs Don) to identify FTTH expansion opportunities.

---

## Quick Start

```bash
pip install -r requirements.txt
python scripts/build_demo.py
```

The script runs the full pipeline and opens the dashboard at **http://localhost:8501**.

---

## What It Shows

| Tab | Content |
|-----|---------|
| 🗺️ Map | Waheb (green) / RAMI (blue) / Don (red) networks on Folium map with coverage circles, opportunity zones, and zone priority markers |
| 📊 Rankings | Priority table, pie chart, bar chart — sorted by AI-computed score |
| 🔍 Detection | Synthetic aerial images with bounding boxes; upload-your-own for real YOLOv8 |
| 💼 Business Case | ROI curves, investment table, coverage gap scatter |
| ⚔️ Competitive Intel | Market share KPIs, zone-by-zone stacked bar chart, opportunity ranking, gap table (URGENT/EXPAND/DEFEND), revenue calculator slider |

---

## Market Position

| Company | Market Share | Cabinets | Strong Zones |
|---------|-------------|---------|--------------|
| **Waheb** (us) | **10%** | 45 | Ghubra, MQ, Bausher |
| RAMI | 35% | 95 | Ruwi, Mutrah, Wattayah |
| Don | 25% | 70 | Shatti, Azaiba, Al Khuwair |
| Unserved | 30% | — | ~105,000 homes = **our opportunity** |

---

## Revenue Rule of Thumb

Each 1% market share gain ≈ **700,000 OMR/year** in new revenue.  
Growing 10% → 18% requires ~30 new cabinets (750,000 OMR investment),
paying back in **~13 months**.

---

## Project Structure

```
yolo-ftth-density-estimator/
├── app/
│   ├── config.py              # Paths, market shares, cost constants
│   ├── image_generator.py     # Synthetic 640×640 aerial images (PIL)
│   ├── detector.py            # YOLOv8 detection engine
│   ├── scorer.py              # Zone priority scoring algorithm
│   ├── network_generator.py   # Waheb / RAMI / Don asset generator
│   ├── data_loader.py         # Load zones, cache results, load networks
│   ├── map_builder.py         # Folium 6-layer competitive map
│   ├── report_exporter.py     # CSV + executive summary export
│   └── dashboard.py           # Streamlit 5-tab dashboard
├── data/
│   ├── zones.json             # 15 Muscat zones with metadata
│   ├── waheb_network.json     # 237 Waheb assets (generated)
│   ├── rami_network.json      # 543 RAMI assets (generated)
│   ├── don_network.json       # 378 Don assets (generated)
│   ├── market_summary.json    # Shares, intensity, opportunity zones
│   ├── sample_images/         # Synthetic aerial images (640×640)
│   └── results/               # Cached scoring results
├── outputs/
│   ├── falcon_competitive_map.html   # Standalone Folium map
│   ├── FALCON_Report.csv             # Full zone analysis
│   └── FALCON_Executive_Summary.txt # Boardroom summary
├── scripts/
│   └── build_demo.py          # Full pipeline launcher
├── run.py                     # Simple launcher (python run.py --demo)
└── requirements.txt
```

---

## Alternative Launch

```bash
python run.py --demo            # Simple demo launch
python run.py --demo --port 8502
```

---

## Scoring Algorithm

Priority Score (0–100) = **40%** vehicle density + **35%** coverage gap + **25%** population

---

*FALCON v1.0 · Muscat FTTH Competitive Intelligence · Waheb Telecom · 2026*
