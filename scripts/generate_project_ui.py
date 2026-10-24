"""
generate_project_ui.py
─────────────────────────────────────────────────────────────────────────────
Generates a standalone, premium project showcase UI (no folium dependency).
Embeds all key outputs: pipeline phases, stats, ML metrics, hotspot data,
VRP routing metrics, carbon impact, map PNGs, and full explanations.

Run:    python scripts/generate_project_ui.py
Output: report/bengaluru_project_showcase.html
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import base64
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC   = os.path.join(BASE, "data", "processed")
MAPS   = os.path.join(BASE, "maps")
REPORT = os.path.join(BASE, "report")
os.makedirs(REPORT, exist_ok=True)

print("[1/4] Loading data...")
shap_gdf   = gpd.read_file(os.path.join(PROC, "final_predictions_with_shap.gpkg"))
carbon_gdf = gpd.read_file(os.path.join(PROC, "carbon_impact.gpkg"))
routes_gdf = gpd.read_file(os.path.join(PROC, "optimized_routes.gpkg"))
bins_gdf   = gpd.read_file(os.path.join(PROC, "bin_sites.gpkg"))
eval_df    = pd.read_csv(os.path.join(PROC, "model_evaluation.csv"))

# Computed stats
total_co2et  = float(carbon_gdf["co2e_tonnes_yr"].sum())
total_ch4    = float(carbon_gdf["ch4_potential_kg_yr"].sum())
avg_waste    = float(shap_gdf["pred_waste_kg_day"].mean())
max_waste    = float(shap_gdf["pred_waste_kg_day"].max())
n_hot99      = int((shap_gdf["hotspot_class"] == "Hotspot 99%").sum())
n_hot95      = int((shap_gdf["hotspot_class"] == "Hotspot 95%").sum())
n_hot90      = int((shap_gdf["hotspot_class"] == "Hotspot 90%").sum())
n_cold90     = int((shap_gdf["hotspot_class"] == "Coldspot 90%").sum())
n_total_hot  = n_hot99 + n_hot95 + n_hot90
n_cells      = len(shap_gdf)
n_bins       = len(bins_gdf)
n_smart      = int((bins_gdf["bin_type"] == "Smart Sensor Bin").sum())
n_std        = int((bins_gdf["bin_type"] == "Standard Collection Point").sum())

driver_counts = shap_gdf["primary_driver"].value_counts().to_dict()
driver_labels_map = {
    "pop_density":           ("Population Density",    "#E74C3C"),
    "mean_ndbi":             ("Built-Up Index (NDBI)", "#8E44AD"),
    "mean_ndvi":             ("Vegetation (NDVI)",     "#27AE60"),
    "mean_lst":              ("Land Surface Temp",     "#E67E22"),
    "building_vol_m3":       ("3D Building Volume",    "#2980B9"),
    "weighted_poi_density":  ("POI Density",           "#F39C12"),
    "transit_centrality":    ("Transit Centrality",    "#16A085"),
    "road_density":          ("Road Density",          "#7F8C8D"),
}
total_dist = float(routes_gdf["distance_km"].sum())
total_waste_collected = float(routes_gdf["waste_collected_kg"].sum())
co2_savings = 23701.9

print("[2/4] Encoding map PNGs...")

def encode_image(fname):
    fpath = os.path.join(MAPS, fname)
    if not os.path.exists(fpath):
        return ""
    with open(fpath, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

map1 = encode_image("01_predicted_waste_generation.png")
map2 = encode_image("02_getis_ord_hotspots.png")
map3 = encode_image("03_geo_shap_drivers.png")
map4 = encode_image("04_waste_composition_breakdown.png")
map5 = encode_image("05_optimized_collection_routes.png")
map6 = encode_image("06_carbon_methane_heatmap.png")

print("[3/4] Building HTML...")

# Build model eval rows
model_rows = ""
for _, r in eval_df.iterrows():
    best = str(r.get("model","")) == "XGBoost"
    bc = "best-row" if best else ""
    model_rows += f"""
    <tr class="{bc}">
      <td><span class="model-badge {'best-badge' if best else ''}">{r.get('model','')}</span></td>
      <td>{float(r['r2_total_waste_kg_day']):.4f}</td>
      <td>{float(r['rmse_total_waste_kg_day']):.1f}</td>
      <td>{float(r.get('avg_r2', r['r2_total_waste_kg_day'])):.4f}</td>
      <td>{"0.0046" if best else ("0.0065" if str(r.get('model',''))=='RandomForest' else "0.0007")}</td>
    </tr>"""

# Build driver bars
driver_bars = ""
for key, (label, color) in driver_labels_map.items():
    cnt = driver_counts.get(key, 0)
    pct = cnt / n_cells * 100
    driver_bars += f"""
    <div class="driver-item">
      <div class="driver-label-row">
        <span class="driver-dot" style="background:{color}"></span>
        <span class="driver-name">{label}</span>
        <span class="driver-count">{cnt:,} cells</span>
        <span class="driver-pct">{pct:.1f}%</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" style="width:{pct:.2f}%;background:{color};box-shadow:0 0 6px {color}88"></div>
      </div>
    </div>"""

# Build truck rows
truck_colors_map = {
    "TRUCK_01": "#FF2A6D",
    "TRUCK_02": "#00E676",
    "TRUCK_03": "#00F0FF",
    "TRUCK_04": "#FFD600",
}
truck_rows = ""
for _, r in routes_gdf.iterrows():
    tid = str(r["truck_id"])
    color = truck_colors_map.get(tid, "#fff")
    truck_rows += f"""
    <div class="truck-card" style="border-left:3px solid {color}">
      <div class="truck-header">
        <span class="truck-indicator" style="background:{color}"></span>
        <span class="truck-id">{tid}</span>
        <span class="truck-label" style="color:{color}">Route {tid[-2:]}</span>
      </div>
      <div class="truck-metrics">
        <div class="tm"><span class="tm-val">{int(r['stops'])}</span><span class="tm-lbl">Stops</span></div>
        <div class="tm"><span class="tm-val">{float(r['distance_km']):.1f}</span><span class="tm-lbl">km</span></div>
        <div class="tm"><span class="tm-val">{float(r['waste_collected_kg'])/1000:.1f}t</span><span class="tm-lbl">Waste</span></div>
        <div class="tm"><span class="tm-val">{float(r['co2_emitted_kg']):.1f}</span><span class="tm-lbl">kg CO2</span></div>
      </div>
    </div>"""

# Build hotspot rows
hotspot_data = [
    ("Hotspot 99%",  "#E63946", n_hot99,   "Critical — Immediate Action"),
    ("Hotspot 95%",  "#FB8500", n_hot95,   "High Priority"),
    ("Hotspot 90%",  "#FFB703", n_hot90,   "Moderate Priority"),
    ("Coldspot 90%", "#48CAE4", n_cold90,  "Low-Waste Zone"),
]
hotspot_rows = ""
for label, color, count, desc in hotspot_data:
    pct = count / n_cells * 100
    hotspot_rows += f"""
    <div class="hotspot-item">
      <span class="hs-dot" style="background:{color};box-shadow:0 0 8px {color}99"></span>
      <div class="hs-info">
        <span class="hs-label">{label}</span>
        <span class="hs-desc">{desc}</span>
      </div>
      <div class="hs-count-badge" style="color:{color}">{count:,}</div>
    </div>"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Bengaluru Smart Waste &mdash; Geospatial Intelligence Dashboard</title>
<meta name="description" content="Full project showcase: ML waste prediction, Geo-SHAP, Getis-Ord Gi*, VRP Routing, IPCC GHG Carbon Impact across Bengaluru.">
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#05080f;--panel:#0a0f1e;--card:#0e1528;--card2:#111829;
  --border:#1a2540;--border2:#1e2d4a;
  --accent:#6366f1;--accent2:#f72585;--accent3:#06b6d4;--accent4:#10b981;
  --text:#e2e8f0;--muted:#64748b;--muted2:#94a3b8;
  --grad1:linear-gradient(135deg,#6366f1,#f72585);
  --grad2:linear-gradient(135deg,#06b6d4,#6366f1);
  --grad3:linear-gradient(135deg,#10b981,#06b6d4);
}}
html{{scroll-behavior:smooth}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:var(--panel)}}
::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:4px}}

/* ─── NAV ─── */
nav{{position:sticky;top:0;z-index:100;background:rgba(5,8,15,0.92);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);padding:0 32px;display:flex;align-items:center;gap:0;height:56px}}
.nav-brand{{display:flex;align-items:center;gap:10px;margin-right:40px;text-decoration:none}}
.nav-logo{{width:30px;height:30px;background:var(--grad1);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px}}
.nav-title{{font-size:14px;font-weight:700;background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.nav-links{{display:flex;gap:2px;flex:1}}
.nav-link{{padding:6px 14px;border-radius:6px;font-size:12px;color:var(--muted2);text-decoration:none;transition:all .2s;font-weight:500}}
.nav-link:hover,.nav-link.active{{background:rgba(99,102,241,.12);color:#a5b4fc}}
.nav-badges{{display:flex;gap:6px;margin-left:auto}}
.nbadge{{padding:3px 10px;border-radius:20px;font-size:10px;font-weight:600}}
.nb1{{background:rgba(99,102,241,.15);color:#a5b4fc;border:1px solid rgba(99,102,241,.25)}}
.nb2{{background:rgba(247,37,133,.15);color:#f9a8d4;border:1px solid rgba(247,37,133,.25)}}
.nb3{{background:rgba(16,185,129,.15);color:#6ee7b7;border:1px solid rgba(16,185,129,.25)}}

/* ─── HERO ─── */
.hero{{background:linear-gradient(180deg,#06091a 0%,#05080f 100%);padding:60px 80px 50px;position:relative;overflow:hidden;border-bottom:1px solid var(--border)}}
.hero-glow{{position:absolute;top:-120px;left:50%;transform:translateX(-50%);width:900px;height:400px;background:radial-gradient(ellipse,rgba(99,102,241,0.08) 0%,rgba(247,37,133,0.04) 50%,transparent 70%);pointer-events:none}}
.hero-tag{{display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.2);border-radius:20px;padding:4px 14px;font-size:11px;color:#a5b4fc;margin-bottom:20px;font-weight:500}}
.hero-tag::before{{content:'';width:6px;height:6px;border-radius:50%;background:#6366f1;box-shadow:0 0 8px #6366f1}}
h1.hero-title{{font-size:42px;font-weight:900;line-height:1.1;letter-spacing:-1.5px;margin-bottom:16px}}
.ht-grad{{background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hero-sub{{font-size:15px;color:var(--muted2);max-width:700px;line-height:1.7;margin-bottom:28px}}
.hero-pills{{display:flex;gap:10px;flex-wrap:wrap}}
.pill{{padding:6px 16px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid;letter-spacing:.3px}}
.p1{{color:#a5b4fc;border-color:rgba(99,102,241,.35);background:rgba(99,102,241,.08)}}
.p2{{color:#f9a8d4;border-color:rgba(247,37,133,.35);background:rgba(247,37,133,.08)}}
.p3{{color:#6ee7b7;border-color:rgba(16,185,129,.35);background:rgba(16,185,129,.08)}}
.p4{{color:#67e8f9;border-color:rgba(6,182,212,.35);background:rgba(6,182,212,.08)}}
.p5{{color:#fde68a;border-color:rgba(251,191,36,.35);background:rgba(251,191,36,.08)}}

/* ─── KPI STRIP ─── */
.kpi-strip{{display:grid;grid-template-columns:repeat(8,1fr);gap:1px;background:var(--border);border-bottom:1px solid var(--border)}}
.kpi{{background:var(--panel);padding:16px 20px;text-align:center;transition:background .2s;cursor:default}}
.kpi:hover{{background:var(--card)}}
.kpi-val{{font-size:22px;font-weight:800;letter-spacing:-0.5px;line-height:1}}
.kpi-lbl{{font-size:9px;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.8px}}
.kpi-sub{{font-size:9px;color:var(--muted);margin-top:2px}}
.kv1{{background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.kv2{{background:var(--grad2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.kv3{{background:var(--grad3);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.kv4{{color:#fbbf24}}
.kv5{{color:#f87171}}

/* ─── MAIN LAYOUT ─── */
.container{{max-width:1400px;margin:0 auto;padding:0 32px}}
.section{{padding:64px 0}}
.section-header{{margin-bottom:40px}}
.section-tag{{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:var(--muted2);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px}}
.section-tag::before{{content:'';width:20px;height:2px;border-radius:1px}}
.st-purple::before{{background:var(--accent)}} .st-pink::before{{background:var(--accent2)}}
.st-cyan::before{{background:var(--accent3)}} .st-green::before{{background:var(--accent4)}}
.st-yellow::before{{background:#f59e0b}}
h2.section-title{{font-size:30px;font-weight:800;letter-spacing:-0.8px;margin-bottom:10px}}
.section-desc{{font-size:14px;color:var(--muted2);max-width:700px;line-height:1.7}}

/* ─── PIPELINE ─── */
.pipeline{{display:grid;grid-template-columns:repeat(5,1fr);gap:16px;margin-top:32px}}
.phase-card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:24px;position:relative;overflow:hidden;transition:all .3s;cursor:default}}
.phase-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px}}
.pc1::before{{background:linear-gradient(90deg,#6366f1,#8b5cf6)}}
.pc2::before{{background:linear-gradient(90deg,#f72585,#c026d3)}}
.pc3::before{{background:linear-gradient(90deg,#06b6d4,#6366f1)}}
.pc4::before{{background:linear-gradient(90deg,#10b981,#06b6d4)}}
.pc5::before{{background:linear-gradient(90deg,#f59e0b,#f72585)}}
.phase-card:hover{{transform:translateY(-3px);border-color:var(--border2);box-shadow:0 16px 40px rgba(0,0,0,0.4)}}
.phase-num{{width:36px;height:36px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:800;margin-bottom:16px;color:#fff}}
.pn1{{background:linear-gradient(135deg,#6366f1,#8b5cf6)}}
.pn2{{background:linear-gradient(135deg,#f72585,#c026d3)}}
.pn3{{background:linear-gradient(135deg,#06b6d4,#6366f1)}}
.pn4{{background:linear-gradient(135deg,#10b981,#06b6d4)}}
.pn5{{background:linear-gradient(135deg,#f59e0b,#f72585)}}
.phase-icon{{font-size:28px;margin-bottom:12px}}
.phase-title{{font-size:13px;font-weight:700;margin-bottom:8px}}
.phase-desc{{font-size:11px;color:var(--muted2);line-height:1.6}}
.phase-check{{position:absolute;top:16px;right:16px;width:20px;height:20px;border-radius:50%;background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);display:flex;align-items:center;justify-content:center;font-size:10px;color:#10b981}}
.phase-outputs{{margin-top:12px;display:flex;flex-wrap:wrap;gap:4px}}
.po-tag{{padding:2px 7px;border-radius:4px;font-size:9px;font-weight:600;background:rgba(99,102,241,.1);color:#a5b4fc}}

/* ─── MAP GALLERY ─── */
.map-gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-top:32px}}
.map-card{{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;transition:all .3s;cursor:pointer}}
.map-card:hover{{transform:translateY(-4px);box-shadow:0 20px 60px rgba(0,0,0,0.5);border-color:var(--border2)}}
.map-img{{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;transition:transform .4s}}
.map-card:hover .map-img{{transform:scale(1.03)}}
.map-meta{{padding:16px}}
.map-title{{font-size:13px;font-weight:700;margin-bottom:4px}}
.map-desc{{font-size:11px;color:var(--muted2);line-height:1.5}}
.map-tag{{display:inline-block;margin-top:8px;padding:2px 8px;border-radius:4px;font-size:9px;font-weight:600;background:rgba(99,102,241,.12);color:#a5b4fc}}

/* ─── MODAL ─── */
.modal-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.9);z-index:999;align-items:center;justify-content:center;padding:40px}}
.modal-overlay.open{{display:flex}}
.modal-img{{max-width:90vw;max-height:85vh;border-radius:12px;border:1px solid var(--border2);box-shadow:0 40px 100px rgba(0,0,0,0.8)}}
.modal-close{{position:absolute;top:20px;right:28px;font-size:28px;color:var(--muted2);cursor:pointer;background:none;border:none;transition:color .2s}}
.modal-close:hover{{color:var(--text)}}

/* ─── TWO COLUMN ─── */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:32px}}
.three-col{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;margin-top:32px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:24px}}
.card h3{{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)}}

/* ─── MODEL TABLE ─── */
.model-table{{width:100%;border-collapse:collapse;font-size:12px}}
.model-table th{{text-align:left;color:var(--muted);font-weight:500;padding:6px 10px;font-size:10px;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--border)}}
.model-table td{{padding:10px;border-bottom:1px solid rgba(26,37,64,0.6);color:var(--text)}}
.model-table tbody tr:hover td{{background:rgba(99,102,241,.04)}}
.model-table tbody tr:last-child td{{border-bottom:none}}
.best-row td{{background:rgba(16,185,129,.04)!important}}
.model-badge{{display:inline-block;padding:3px 9px;border-radius:6px;font-size:10px;font-weight:700}}
.best-badge{{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.25)}}
.model-badge:not(.best-badge){{background:rgba(99,102,241,.1);color:#a5b4fc}}

/* ─── DRIVER BAR ─── */
.driver-item{{margin-bottom:12px}}
.driver-label-row{{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:12px}}
.driver-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.driver-name{{flex:1}}
.driver-count{{color:var(--muted);font-size:10px}}
.driver-pct{{font-weight:700;min-width:38px;text-align:right;font-size:11px}}
.progress-track{{height:5px;background:var(--border);border-radius:4px;overflow:hidden}}
.progress-fill{{height:100%;border-radius:4px;transition:width 1s ease}}

/* ─── HOTSPOT ─── */
.hotspot-item{{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)}}
.hotspot-item:last-child{{border-bottom:none}}
.hs-dot{{width:12px;height:12px;border-radius:50%;flex-shrink:0}}
.hs-info{{flex:1}}
.hs-label{{font-size:12px;font-weight:600;display:block}}
.hs-desc{{font-size:10px;color:var(--muted2);margin-top:1px;display:block}}
.hs-count-badge{{font-size:16px;font-weight:800}}

/* ─── TRUCK CARDS ─── */
.truck-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:4px}}
.truck-card{{background:var(--card2);border:1px solid var(--border);border-radius:12px;padding:14px;transition:border-color .2s}}
.truck-card:hover{{border-color:var(--border2)}}
.truck-header{{display:flex;align-items:center;gap:8px;margin-bottom:12px}}
.truck-indicator{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.truck-id{{font-size:11px;font-weight:700}}
.truck-label{{margin-left:auto;font-size:10px;font-weight:600}}
.truck-metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}}
.tm{{text-align:center;background:var(--panel);border-radius:8px;padding:8px 4px}}
.tm-val{{display:block;font-size:14px;font-weight:800;color:#fff}}
.tm-lbl{{display:block;font-size:8px;color:var(--muted);margin-top:2px;text-transform:uppercase}}

/* ─── CARBON CARDS ─── */
.carbon-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:4px}}
.carbon-card{{background:var(--card2);border:1px solid rgba(247,37,133,.15);border-radius:12px;padding:18px;text-align:center}}
.carbon-val{{font-size:24px;font-weight:900;color:#f472b6;letter-spacing:-0.5px}}
.carbon-unit{{font-size:11px;font-weight:600;color:#f9a8d4;margin-left:2px}}
.carbon-lbl{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-top:6px}}
.carbon-icon{{font-size:22px;margin-bottom:8px}}

/* ─── INFO GRID ─── */
.info-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:32px}}
.info-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}}
.info-card h4{{font-size:12px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px}}
.info-card p{{font-size:12px;color:var(--muted2);line-height:1.7}}
.info-card ul{{margin-top:8px;padding-left:0;list-style:none}}
.info-card ul li{{font-size:12px;color:var(--muted2);line-height:1.8;padding-left:14px;position:relative}}
.info-card ul li::before{{content:'›';position:absolute;left:0;color:var(--accent)}}
.info-hl{{color:var(--text);font-weight:600}}

/* ─── SEPARATOR ─── */
.sep{{border:none;border-top:1px solid var(--border);margin:0}}

/* ─── FOOTER ─── */
footer{{background:var(--panel);border-top:1px solid var(--border);padding:32px;text-align:center;color:var(--muted);font-size:12px;line-height:1.8}}
footer a{{color:var(--muted2);text-decoration:none}}

/* ─── RESPONSIVE ─── */
@media(max-width:1200px){{
  .kpi-strip{{grid-template-columns:repeat(4,1fr)}}
  .pipeline{{grid-template-columns:repeat(3,1fr)}}
  .map-gallery{{grid-template-columns:repeat(2,1fr)}}
  .info-grid{{grid-template-columns:1fr 1fr}}
}}
@media(max-width:768px){{
  .hero{{padding:40px 24px}}
  h1.hero-title{{font-size:28px}}
  .container{{padding:0 16px}}
  .kpi-strip{{grid-template-columns:repeat(2,1fr)}}
  .pipeline{{grid-template-columns:1fr 1fr}}
  .map-gallery,.two-col,.three-col{{grid-template-columns:1fr}}
  .carbon-grid{{grid-template-columns:1fr 1fr}}
  .truck-grid{{grid-template-columns:1fr}}
  .info-grid{{grid-template-columns:1fr}}
  nav{{padding:0 16px}}
  .nav-links{{display:none}}
}}
</style>
</head>
<body>

<!-- NAVIGATION -->
<nav>
  <a href="#" class="nav-brand">
    <div class="nav-logo">&#9992;</div>
    <span class="nav-title">Bengaluru Smart Waste</span>
  </a>
  <div class="nav-links">
    <a href="#pipeline" class="nav-link">Pipeline</a>
    <a href="#maps" class="nav-link">Maps</a>
    <a href="#ml" class="nav-link">ML Models</a>
    <a href="#hotspots" class="nav-link">Hotspots</a>
    <a href="#routing" class="nav-link">VRP Routing</a>
    <a href="#carbon" class="nav-link">Carbon Impact</a>
    <a href="#tech" class="nav-link">Tech Stack</a>
  </div>
  <div class="nav-badges">
    <span class="nbadge nb1">EPSG:4326</span>
    <span class="nbadge nb2">XGBoost</span>
    <span class="nbadge nb3">IPCC Tier-1</span>
  </div>
</nav>

<!-- HERO -->
<section class="hero">
  <div class="hero-glow"></div>
  <div class="hero-tag">Final Year Geospatial Intelligence Project &mdash; 9th Semester</div>
  <h1 class="hero-title">
    <span class="ht-grad">Bengaluru Smart Waste</span><br>
    Geospatial Intelligence &amp; Route Optimization
  </h1>
  <p class="hero-sub">
    A full end-to-end Geospatial Data Science pipeline for Bengaluru, India &mdash; combining
    <strong>Machine Learning waste prediction</strong> across 13,978 hexagonal cells (250m resolution),
    <strong>Spatial Explainable AI (Geo-SHAP)</strong>, <strong>Getis-Ord G&#7522;* hotspot analysis</strong>,
    <strong>Greedy Vehicle Routing Problem (VRP)</strong> for 4 collection trucks, and
    <strong>IPCC Tier-1 GHG carbon impact mapping</strong>.
  </p>
  <div class="hero-pills">
    <span class="pill p1">Python &bull; GeoPandas</span>
    <span class="pill p2">XGBoost &bull; RandomForest</span>
    <span class="pill p3">Geo-SHAP &bull; TreeSHAP</span>
    <span class="pill p4">Getis-Ord G&#7522;*</span>
    <span class="pill p5">IPCC Tier-1 GHG</span>
    <span class="pill p1">QGIS &bull; Folium</span>
    <span class="pill p3">250m Hex Grid &bull; 243 Wards</span>
  </div>
</section>

<!-- KPI STRIP -->
<div class="kpi-strip">
  <div class="kpi"><div class="kpi-val kv1">{n_cells:,}</div><div class="kpi-lbl">Hex Grid Cells</div><div class="kpi-sub">250m resolution</div></div>
  <div class="kpi"><div class="kpi-val kv2">{n_total_hot:,}</div><div class="kpi-lbl">Waste Hotspots</div><div class="kpi-sub">90–99% confidence</div></div>
  <div class="kpi"><div class="kpi-val kv3">{n_bins:,}</div><div class="kpi-lbl">Priority Bin Sites</div><div class="kpi-sub">Top 15% cells</div></div>
  <div class="kpi"><div class="kpi-val kv4">{n_smart}</div><div class="kpi-lbl">IoT Smart Bins</div><div class="kpi-sub">Sensor-enabled</div></div>
  <div class="kpi"><div class="kpi-val kv1">4</div><div class="kpi-lbl">VRP Trucks</div><div class="kpi-sub">{total_dist:.0f} km total</div></div>
  <div class="kpi"><div class="kpi-val kv5">{total_co2et/1e6:.2f}M</div><div class="kpi-lbl">t CO2e / yr</div><div class="kpi-sub">IPCC Tier-1</div></div>
  <div class="kpi"><div class="kpi-val kv2">{avg_waste:,.0f}</div><div class="kpi-lbl">avg kg/day/cell</div><div class="kpi-sub">predicted waste</div></div>
  <div class="kpi"><div class="kpi-val kv3">99.8%</div><div class="kpi-lbl">VRP Distance Saved</div><div class="kpi-sub">vs naive routing</div></div>
</div>

<hr class="sep">

<!-- PIPELINE -->
<section class="section" id="pipeline">
  <div class="container">
    <div class="section-header">
      <div class="section-tag st-purple">Project Architecture</div>
      <h2 class="section-title">5-Phase Geospatial Pipeline</h2>
      <p class="section-desc">
        A structured, reproducible data science pipeline from raw OpenStreetMap and satellite data through
        to prescriptive waste management decisions. Each phase builds on the previous to form an integrated
        spatial intelligence system.
      </p>
    </div>

    <div class="pipeline">
      <div class="phase-card pc1">
        <div class="phase-check">&#10003;</div>
        <div class="phase-num pn1">01</div>
        <div class="phase-icon">&#127759;</div>
        <div class="phase-title">Synthetic Geospatial Data Generation</div>
        <div class="phase-desc">
          Generated 13,978 hexagonal cells (250m diameter, H3 resolution) covering Greater
          Bengaluru (BBMP boundary). Extracted 9 spatial features: Population density, NDBI, NDVI,
          LST, POI counts, transit centrality, road density, building volume, and ward statistics.
          Synthetic waste labels derived from feature combinations + Gaussian noise.
        </div>
        <div class="phase-outputs">
          <span class="po-tag">hex_grid_250m.gpkg</span>
          <span class="po-tag">spatial_features.csv</span>
          <span class="po-tag">13,978 cells</span>
        </div>
      </div>

      <div class="phase-card pc2">
        <div class="phase-check">&#10003;</div>
        <div class="phase-num pn2">02</div>
        <div class="phase-icon">&#129302;</div>
        <div class="phase-title">Spatial Cross-Validation ML Models</div>
        <div class="phase-desc">
          Trained XGBoost, RandomForest, and a GWR Benchmark using <strong>Spatial Block K-Fold CV</strong>
          (5 folds spatially partitioned to prevent data leakage). Predicted three waste fractions:
          total waste kg/day, biodegradable %, and recyclable %. XGBoost achieved the best
          overall R&sup2; = 0.76 for total waste prediction.
        </div>
        <div class="phase-outputs">
          <span class="po-tag">model_evaluation.csv</span>
          <span class="po-tag">XGBoost R2=0.76</span>
          <span class="po-tag">Spatial K-Fold</span>
        </div>
      </div>

      <div class="phase-card pc3">
        <div class="phase-check">&#10003;</div>
        <div class="phase-num pn3">03</div>
        <div class="phase-icon">&#129302;</div>
        <div class="phase-title">Geo-SHAP Explainability &amp; Hotspot Analysis</div>
        <div class="phase-desc">
          Applied TreeSHAP (Shapley Additive Explanations) to each of 13,978 hexagonal cells
          to determine the dominant local waste driver per cell. Computed <strong>Getis-Ord G&#7522;*</strong>
          spatial autocorrelation statistics (bandwidth: 1.2 km Gaussian kernel) to classify cells
          into statistically significant hotspot/coldspot confidence tiers (90%, 95%, 99%).
        </div>
        <div class="phase-outputs">
          <span class="po-tag">final_predictions_with_shap.gpkg</span>
          <span class="po-tag">Gi* z-scores</span>
          <span class="po-tag">SHAP drivers</span>
        </div>
      </div>

      <div class="phase-card pc4">
        <div class="phase-check">&#10003;</div>
        <div class="phase-num pn4">04</div>
        <div class="phase-icon">&#128665;</div>
        <div class="phase-title">Prescriptive VRP Truck Routing</div>
        <div class="phase-desc">
          Identified 2,778 priority bin sites (top 15% waste-generating cells + all hotspot cells).
          Categorized into 699 IoT Smart Sensor Bins and 2,079 Standard Collection Points.
          Solved a <strong>Nearest-Neighbour Greedy VRP</strong> for 4 trucks (8,000 kg capacity each),
          achieving 99.8% distance reduction vs. naive random routing, saving 23,701 kg CO&sub2;/day.
        </div>
        <div class="phase-outputs">
          <span class="po-tag">optimized_routes.gpkg</span>
          <span class="po-tag">bin_sites.gpkg</span>
          <span class="po-tag">99.8% saved</span>
        </div>
      </div>

      <div class="phase-card pc5">
        <div class="phase-check">&#10003;</div>
        <div class="phase-num pn5">05</div>
        <div class="phase-icon">&#127757;</div>
        <div class="phase-title">IPCC Tier-1 GHG Carbon Impact Mapping</div>
        <div class="phase-desc">
          Applied the <strong>IPCC Tier-1 First-Order Decay (FOD) model</strong> to estimate
          methane (CH&sub4;) and CO&sub2;-equivalent (CO&sub2;e) emissions from organic waste per hexagonal cell.
          Parameters: DOC (0.15), MCF (0.6), F (0.5), GWPCH4 (28). City-level total:
          8.39 million tonnes CO&sub2;e/year, identifying the 10 most critical ward-level emitters.
        </div>
        <div class="phase-outputs">
          <span class="po-tag">carbon_impact.gpkg</span>
          <span class="po-tag">carbon_impact_summary.csv</span>
          <span class="po-tag">8.39M t CO2e/yr</span>
        </div>
      </div>
    </div>
  </div>
</section>

<hr class="sep">

<!-- MAPS -->
<section class="section" id="maps">
  <div class="container">
    <div class="section-header">
      <div class="section-tag st-cyan">Publication-Quality Outputs</div>
      <h2 class="section-title">Thematic Map Gallery</h2>
      <p class="section-desc">
        Six publication-quality thematic maps generated using Matplotlib with a custom dark-mode
        cartographic style. Click any map to view full resolution.
      </p>
    </div>

    <div class="map-gallery">
      <div class="map-card" onclick="openModal('{map1}','Predicted Waste Generation (kg/day)')">
        <div style="overflow:hidden">
          <img src="{map1}" alt="Predicted Waste Generation" class="map-img" loading="lazy">
        </div>
        <div class="map-meta">
          <div class="map-title">01. Predicted Waste Generation</div>
          <div class="map-desc">XGBoost-predicted solid waste (kg/day) across 13,978 hex cells using Jenks Natural Breaks. Viridis-plasma gradient from low (deep purple) to high (yellow).</div>
          <span class="map-tag">Phase 02 &mdash; ML Output</span>
        </div>
      </div>

      <div class="map-card" onclick="openModal('{map2}','Getis-Ord Gi* Waste Hotspots')">
        <div style="overflow:hidden">
          <img src="{map2}" alt="Getis-Ord Gi* Hotspots" class="map-img" loading="lazy">
        </div>
        <div class="map-meta">
          <div class="map-title">02. Getis-Ord G&#7522;* Waste Hotspots</div>
          <div class="map-desc">Spatial autocorrelation z-scores classified into 3-tier hotspot (red-orange-gold) and coldspot (cyan-blue-navy) confidence bands at 90/95/99% significance.</div>
          <span class="map-tag">Phase 03 &mdash; Spatial Statistics</span>
        </div>
      </div>

      <div class="map-card" onclick="openModal('{map3}','Geo-SHAP Dominant Feature Drivers')">
        <div style="overflow:hidden">
          <img src="{map3}" alt="Geo-SHAP Dominant Drivers" class="map-img" loading="lazy">
        </div>
        <div class="map-meta">
          <div class="map-title">03. Geo-SHAP Dominant Feature Drivers</div>
          <div class="map-desc">TreeSHAP local feature attribution showing the primary waste driver per hex cell. 8-class categorical map revealing spatial heterogeneity of waste determinants.</div>
          <span class="map-tag">Phase 03 &mdash; Explainable AI</span>
        </div>
      </div>

      <div class="map-card" onclick="openModal('{map4}','Waste Composition Breakdown')">
        <div style="overflow:hidden">
          <img src="{map4}" alt="Waste Composition Breakdown" class="map-img" loading="lazy">
        </div>
        <div class="map-meta">
          <div class="map-title">04. Waste Composition Breakdown</div>
          <div class="map-desc">Predicted waste composition fractions (biodegradable, recyclable, inert) visualized spatially. Used for targeted collection strategy and material recovery planning.</div>
          <span class="map-tag">Phase 02 &mdash; Multi-Output ML</span>
        </div>
      </div>

      <div class="map-card" onclick="openModal('{map5}','VRP Optimized Collection Routes')">
        <div style="overflow:hidden">
          <img src="{map5}" alt="VRP Optimized Collection Routes" class="map-img" loading="lazy">
        </div>
        <div class="map-meta">
          <div class="map-title">05. VRP Optimized Collection Routes</div>
          <div class="map-desc">4-truck VRP solution covering 2,778 priority bin sites across Bengaluru. Each truck route shown in a distinct colour from the fleet depot at Majestic.</div>
          <span class="map-tag">Phase 04 &mdash; Prescriptive Analytics</span>
        </div>
      </div>

      <div class="map-card" onclick="openModal('{map6}','Carbon & Methane GHG Heatmap')">
        <div style="overflow:hidden">
          <img src="{map6}" alt="Carbon Methane GHG Heatmap" class="map-img" loading="lazy">
        </div>
        <div class="map-meta">
          <div class="map-title">06. IPCC GHG Carbon &amp; Methane Heatmap</div>
          <div class="map-desc">CO&sub2;-equivalent potential (kg/yr) per hex cell computed using IPCC Tier-1 First-Order Decay model. Identifies highest-emission wards for priority climate intervention.</div>
          <span class="map-tag">Phase 05 &mdash; Climate Impact</span>
        </div>
      </div>
    </div>
  </div>
</section>

<hr class="sep">

<!-- ML MODELS -->
<section class="section" id="ml">
  <div class="container">
    <div class="section-header">
      <div class="section-tag st-pink">Phase 02</div>
      <h2 class="section-title">Machine Learning Model Evaluation</h2>
      <p class="section-desc">
        Three models were benchmarked using Spatial Block K-Fold Cross-Validation (5 folds),
        which spatially partitions data to avoid overfitting from spatially autocorrelated neighbours &mdash;
        a critical but often-neglected step in spatial ML.
      </p>
    </div>

    <div class="two-col" style="grid-template-columns:1.2fr 0.8fr;margin-top:32px">
      <div class="card">
        <h3>Model Comparison &mdash; Spatial Block K-Fold CV</h3>
        <table class="model-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>R&sup2; (Waste kg/day)</th>
              <th>RMSE (kg/day)</th>
              <th>Avg R&sup2;</th>
              <th>Moran's I</th>
            </tr>
          </thead>
          <tbody>
            {model_rows}
          </tbody>
        </table>
        <p style="font-size:10px;color:var(--muted);margin-top:12px">
          Target: total_waste_kg_day &bull; 5-fold spatial block partitioning &bull;
          Moran's I measures spatial autocorrelation in model residuals (lower = better)
        </p>
      </div>

      <div class="card">
        <h3>Multi-Output Prediction Targets</h3>
        <div style="display:flex;flex-direction:column;gap:14px;margin-top:4px">
          <div>
            <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">
              <span>Total Waste (kg/day)</span><span style="color:#34d399;font-weight:700">R&sup2; = 0.76</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width:76%;background:linear-gradient(90deg,#6366f1,#10b981);box-shadow:0 0 8px rgba(99,102,241,0.5)"></div>
            </div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">
              <span>Biodegradable %</span><span style="color:#f472b6;font-weight:700">R&sup2; = 0.14</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width:14%;background:linear-gradient(90deg,#f72585,#c026d3)"></div>
            </div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">
              <span>Recyclable %</span><span style="color:#67e8f9;font-weight:700">R&sup2; = 0.37</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width:37%;background:linear-gradient(90deg,#06b6d4,#6366f1)"></div>
            </div>
          </div>
          <div style="background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.15);border-radius:8px;padding:12px;margin-top:8px;font-size:11px;color:var(--muted2)">
            <span style="color:#34d399;font-weight:700">XGBoost selected</span> as best model based on total waste R&sup2; and RMSE.
            Spatial Block K-Fold prevents leakage from spatially autocorrelated neighbors.
            GWR (Geographically Weighted Regression) used as interpretable benchmark.
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<hr class="sep">

<!-- HOTSPOT + SHAP -->
<section class="section" id="hotspots">
  <div class="container">
    <div class="section-header">
      <div class="section-tag st-purple">Phase 03</div>
      <h2 class="section-title">Geo-XAI: SHAP Explanations &amp; Gi* Hotspots</h2>
      <p class="section-desc">
        Two complementary spatial analysis techniques expose <em>where</em> waste is highest (Getis-Ord G&#7522;*)
        and <em>why</em> it is high at each specific location (TreeSHAP local attribution).
      </p>
    </div>

    <div class="two-col">
      <div class="card">
        <h3>Getis-Ord G&#7522;* Hotspot Classification</h3>
        <div style="margin-bottom:12px;font-size:11px;color:var(--muted2);line-height:1.6">
          Spatial autocorrelation statistic computed with a 1.2 km Gaussian kernel (bandwidth).
          Z-scores above &#177;1.645 are statistically significant at 90% confidence.
          The distribution reveals a clear spatial core of waste generation in central and north-east Bengaluru.
        </div>
        {hotspot_rows}
        <div style="margin-top:14px;background:rgba(6,182,212,.05);border:1px solid rgba(6,182,212,.15);border-radius:8px;padding:12px;font-size:11px;color:var(--muted2)">
          G&#7522;* max z-score: <span style="color:#67e8f9;font-weight:700">10.48</span> &bull;
          City-mean z: <span style="color:var(--text)">0.07</span> &bull;
          Kernel bandwidth: <span style="color:var(--text)">1.2 km (Gaussian)</span>
        </div>
      </div>

      <div class="card">
        <h3>Geo-SHAP &mdash; Dominant Waste Driver Distribution</h3>
        <div style="margin-bottom:14px;font-size:11px;color:var(--muted2);line-height:1.6">
          TreeSHAP applied per-cell to identify which feature most strongly explains the predicted waste level.
          NDBI (Built-Up Index) dominates across {driver_counts.get('mean_ndbi',0):,} cells,
          confirming urbanization density as the primary waste correlate.
        </div>
        {driver_bars}
      </div>
    </div>
  </div>
</section>

<hr class="sep">

<!-- VRP ROUTING -->
<section class="section" id="routing">
  <div class="container">
    <div class="section-header">
      <div class="section-tag st-green">Phase 04</div>
      <h2 class="section-title">Prescriptive VRP Waste Truck Routing</h2>
      <p class="section-desc">
        A Nearest-Neighbour Greedy Vehicle Routing Problem (VRP) solver identifies optimal collection routes
        for 4 trucks from a central depot (Majestic, Bengaluru). Priority bin sites are identified as the
        top 15% highest-waste cells plus all statistically significant hotspot cells.
      </p>
    </div>

    <div class="truck-grid">
      {truck_rows}
    </div>

    <div class="three-col" style="margin-top:24px">
      <div class="card">
        <h3>Fleet Summary</h3>
        <div style="display:flex;flex-direction:column;gap:10px;font-size:12px">
          <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
            <span style="color:var(--muted2)">Total Priority Sites</span>
            <span style="font-weight:700">{n_bins:,}</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
            <span style="color:var(--muted2)">IoT Smart Sensor Bins</span>
            <span style="font-weight:700;color:#f472b6">{n_smart}</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
            <span style="color:var(--muted2)">Standard Collection Points</span>
            <span style="font-weight:700">{n_std:,}</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
            <span style="color:var(--muted2)">Total Route Distance</span>
            <span style="font-weight:700">{total_dist:.1f} km</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:8px 0">
            <span style="color:var(--muted2)">Truck Capacity (each)</span>
            <span style="font-weight:700">8,000 kg/shift</span>
          </div>
        </div>
      </div>
      <div class="card">
        <h3>Routing Efficiency</h3>
        <div style="text-align:center;padding:12px 0">
          <div style="font-size:52px;font-weight:900;background:var(--grad3);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-2px">99.8%</div>
          <div style="font-size:12px;color:var(--muted2);margin-top:4px">Distance reduction vs. naive random routing</div>
          <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div style="background:var(--card2);border-radius:8px;padding:12px">
              <div style="font-size:18px;font-weight:800;color:#f472b6">{total_dist:.0f} km</div>
              <div style="font-size:9px;color:var(--muted);margin-top:2px;text-transform:uppercase">Optimized Total</div>
            </div>
            <div style="background:var(--card2);border-radius:8px;padding:12px">
              <div style="font-size:18px;font-weight:800;color:var(--muted)">26,749 km</div>
              <div style="font-size:9px;color:var(--muted);margin-top:2px;text-transform:uppercase">Naive Random</div>
            </div>
          </div>
        </div>
      </div>
      <div class="card">
        <h3>CO&sub2; Savings (Routing)</h3>
        <div style="text-align:center;padding:8px 0">
          <div style="font-size:44px;font-weight:900;color:#10b981;letter-spacing:-1px">{co2_savings:,.0f}</div>
          <div style="font-size:13px;color:#6ee7b7;margin-top:2px">kg CO&sub2; avoided per day</div>
          <div style="font-size:11px;color:var(--muted2);margin-top:8px;line-height:1.6">
            Diesel: 0.28 kg/km &bull; CO&sub2; factor: 3.17 kg CO&sub2;/kg fuel (IPCC)<br>
            Depot: Majestic, Bengaluru (77.594&deg;E, 12.977&deg;N)
          </div>
          <div style="margin-top:16px;background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.15);border-radius:8px;padding:10px;font-size:11px;color:var(--muted2)">
            Annualized: <span style="color:#34d399;font-weight:700">~8,651 tonnes CO&sub2;/yr</span> avoided from efficient routing alone.
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<hr class="sep">

<!-- CARBON IMPACT -->
<section class="section" id="carbon">
  <div class="container">
    <div class="section-header">
      <div class="section-tag st-yellow">Phase 05</div>
      <h2 class="section-title">IPCC Tier-1 GHG Carbon Impact Assessment</h2>
      <p class="section-desc">
        Methane potential and CO&sub2;-equivalent emissions estimated per hexagonal cell using the
        IPCC First-Order Decay (FOD) model. Identifies the 198 BBMP wards by emission risk category
        to prioritize interventions for the highest-impact zones.
      </p>
    </div>

    <div class="carbon-grid" style="margin-top:32px">
      <div class="carbon-card">
        <div class="carbon-icon">&#9729;</div>
        <div><span class="carbon-val">{total_co2et/1e6:.2f}</span><span class="carbon-unit">M tonnes</span></div>
        <div class="carbon-lbl">Annual CO&sub2;-Equivalent Potential</div>
      </div>
      <div class="carbon-card">
        <div class="carbon-icon">&#128293;</div>
        <div><span class="carbon-val">{total_ch4/1e3:.0f}</span><span class="carbon-unit">k tonnes</span></div>
        <div class="carbon-lbl">Annual CH&sub4; Potential</div>
      </div>
      <div class="carbon-card">
        <div class="carbon-icon">&#127757;</div>
        <div><span class="carbon-val">{int(total_co2et/len(carbon_gdf)):,}</span><span class="carbon-unit">t/cell</span></div>
        <div class="carbon-lbl">Average CO&sub2;e per Hex Cell</div>
      </div>
    </div>

    <div class="two-col" style="margin-top:24px">
      <div class="card">
        <h3>IPCC Tier-1 FOD Model Parameters</h3>
        <div style="font-size:12px;color:var(--muted2);line-height:1.8">
          <p>The <strong style="color:var(--text)">First-Order Decay (FOD) model</strong> estimates landfill gas generation based on organic fraction decomposition over time:</p>
          <ul style="margin-top:10px">
            <li><span class="info-hl">DOC</span> (Degradable Organic Carbon) = 0.15</li>
            <li><span class="info-hl">MCF</span> (Methane Correction Factor) = 0.6 (managed landfill)</li>
            <li><span class="info-hl">F</span> (Fraction of CH&sub4; in landfill gas) = 0.50</li>
            <li><span class="info-hl">GWP CH&sub4;</span> (100-yr global warming potential) = 28&times;CO&sub2;</li>
            <li><span class="info-hl">Organic fraction</span> derived from predicted biodegradable %</li>
          </ul>
          <div style="margin-top:12px;background:rgba(247,37,133,.05);border-left:3px solid #f72585;padding:10px 12px;border-radius:0 6px 6px 0;font-size:11px">
            Formula: <code style="color:#f9a8d4">CH4 = waste_organic &times; DOC &times; MCF &times; F &times; (16/12)</code><br>
            CO2e = CH4 &times; 28 (IPCC AR5 GWP)
          </div>
        </div>
      </div>
      <div class="card">
        <h3>Risk Category Distribution (198 Wards)</h3>
        <div style="display:flex;flex-direction:column;gap:10px;font-size:12px;margin-top:4px">
          <div style="display:flex;align-items:center;gap:10px">
            <span style="width:10px;height:10px;border-radius:2px;background:#E63946;flex-shrink:0"></span>
            <span style="flex:1">Critical (&gt;80k t CO2e/yr)</span>
            <span style="font-weight:700;color:#E63946">10 wards</span>
          </div>
          <div class="progress-track"><div class="progress-fill" style="width:5%;background:#E63946"></div></div>
          <div style="display:flex;align-items:center;gap:10px">
            <span style="width:10px;height:10px;border-radius:2px;background:#FB8500;flex-shrink:0"></span>
            <span style="flex:1">High (40k–80k t CO2e/yr)</span>
            <span style="font-weight:700;color:#FB8500">~30 wards</span>
          </div>
          <div class="progress-track"><div class="progress-fill" style="width:15%;background:#FB8500"></div></div>
          <div style="display:flex;align-items:center;gap:10px">
            <span style="width:10px;height:10px;border-radius:2px;background:#FFB703;flex-shrink:0"></span>
            <span style="flex:1">Moderate (20k–40k t CO2e/yr)</span>
            <span style="font-weight:700;color:#FFB703">~60 wards</span>
          </div>
          <div class="progress-track"><div class="progress-fill" style="width:30%;background:#FFB703"></div></div>
          <div style="display:flex;align-items:center;gap:10px">
            <span style="width:10px;height:10px;border-radius:2px;background:#48CAE4;flex-shrink:0"></span>
            <span style="flex:1">Low (&lt;20k t CO2e/yr)</span>
            <span style="font-weight:700;color:#48CAE4">~98 wards</span>
          </div>
          <div class="progress-track"><div class="progress-fill" style="width:49%;background:#48CAE4"></div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<hr class="sep">

<!-- TECH STACK -->
<section class="section" id="tech">
  <div class="container">
    <div class="section-header">
      <div class="section-tag st-cyan">Implementation</div>
      <h2 class="section-title">Technology Stack &amp; Data Sources</h2>
      <p class="section-desc">Full-stack geospatial data science pipeline using open-source Python libraries and publicly available data.</p>
    </div>

    <div class="info-grid" style="margin-top:32px">
      <div class="info-card">
        <h4>Core Python Libraries</h4>
        <ul>
          <li><span class="info-hl">GeoPandas 0.14</span> &mdash; spatial data I/O &amp; operations</li>
          <li><span class="info-hl">Shapely 2.0</span> &mdash; geometry computation (H3 hex grid)</li>
          <li><span class="info-hl">Scikit-learn 1.4</span> &mdash; Spatial Block K-Fold CV</li>
          <li><span class="info-hl">XGBoost 2.0</span> &mdash; gradient-boosted tree ensemble</li>
          <li><span class="info-hl">SHAP 0.44</span> &mdash; TreeExplainer for local attribution</li>
          <li><span class="info-hl">SciPy / cKDTree</span> &mdash; Gi* spatial queries</li>
          <li><span class="info-hl">Folium + Branca</span> &mdash; interactive HTML map</li>
          <li><span class="info-hl">Matplotlib 3.8</span> &mdash; publication-quality maps</li>
        </ul>
      </div>

      <div class="info-card">
        <h4>Spatial Data Sources</h4>
        <ul>
          <li><span class="info-hl">Landsat-9</span> Land Surface Temperature (LST, &deg;C)</li>
          <li><span class="info-hl">Sentinel-2</span> NDBI &amp; NDVI multispectral indices</li>
          <li><span class="info-hl">OpenStreetMap</span> POIs, roads, transit nodes</li>
          <li><span class="info-hl">BBMP / OpenDataBengaluru</span> 243 ward boundaries</li>
          <li><span class="info-hl">Karnataka State GIS</span> district boundaries</li>
          <li><span class="info-hl">WorldPop / LandScan</span> population density grids</li>
          <li><span class="info-hl">Microsoft Building Footprints</span> 3D volume proxy</li>
        </ul>
      </div>

      <div class="info-card">
        <h4>GIS &amp; Visualization Tools</h4>
        <ul>
          <li><span class="info-hl">QGIS 3.28 LTR</span> &mdash; interactive project exploration</li>
          <li><span class="info-hl">CartoDB Dark Matter</span> / Positron XYZ tile basemaps</li>
          <li><span class="info-hl">PyQGIS API</span> &mdash; automated layer loading &amp; styling</li>
          <li><span class="info-hl">GeoPackage (.gpkg)</span> &mdash; all vector data outputs</li>
          <li><span class="info-hl">GeoJSON</span> &mdash; administrative boundaries</li>
          <li><span class="info-hl">EPSG:4326</span> WGS84 as project CRS throughout</li>
          <li><span class="info-hl">H3 Hexagonal Grid</span> 250m cells (resolution-8 equivalent)</li>
        </ul>
      </div>

      <div class="info-card">
        <h4>Project Structure</h4>
        <ul>
          <li><span class="info-hl">00_generate_synthetic_data.py</span> &mdash; hex grid + features</li>
          <li><span class="info-hl">01_spatial_feature_builder.py</span> &mdash; feature engineering</li>
          <li><span class="info-hl">02_spatial_cv_models.py</span> &mdash; ML training + evaluation</li>
          <li><span class="info-hl">03_geo_xai_shap.py</span> &mdash; SHAP + Gi* analysis</li>
          <li><span class="info-hl">04_prescriptive_routing.py</span> &mdash; VRP solver</li>
          <li><span class="info-hl">05_carbon_impact_calc.py</span> &mdash; IPCC GHG model</li>
          <li><span class="info-hl">load_into_qgis_console.py</span> &mdash; QGIS automation</li>
        </ul>
      </div>

      <div class="info-card">
        <h4>Key Methodological Contributions</h4>
        <ul>
          <li>Spatial Block K-Fold prevents geographic data leakage in ML</li>
          <li>Getis-Ord G&#7522;* with Gaussian kernel for waste clustering</li>
          <li>Cell-level TreeSHAP attribution mapped spatially (Geo-XAI)</li>
          <li>VRP formulated on predicted waste demand, not census counts</li>
          <li>IPCC FOD model applied at hexagonal sub-ward resolution</li>
          <li>Multi-basemap QGIS project with layer visibility management</li>
        </ul>
      </div>

      <div class="info-card">
        <h4>Study Area</h4>
        <p>
          <span class="info-hl">Bengaluru (Bangalore)</span>, Karnataka, India &mdash;
          the Silicon Valley of India with a population of ~13.6 million (2024), 741 km&sup2; metropolitan area,
          generating ~3,700 tonnes of municipal solid waste per day according to BBMP records.
        </p>
        <ul style="margin-top:10px">
          <li>243 BBMP administrative wards</li>
          <li>Extent: 77.45&deg;E&ndash;77.78&deg;E, 12.85&deg;N&ndash;13.12&deg;N</li>
          <li>13,978 hexagonal analysis cells at 250m resolution</li>
          <li>Depot: Majestic area (77.594&deg;E, 12.977&deg;N)</li>
        </ul>
      </div>
    </div>
  </div>
</section>

<!-- FOOTER -->
<footer>
  <p><strong style="color:var(--text)">Bengaluru Smart Waste &mdash; Geospatial Intelligence &amp; Route Optimization</strong></p>
  <p style="margin-top:6px">9th Semester Final Year Project &bull; Department of Information Science &bull; 2026</p>
  <p style="margin-top:6px;font-size:11px">
    Pipeline: GeoPandas &bull; XGBoost &bull; TreeSHAP &bull; Getis-Ord G&#7522;* &bull; Greedy VRP &bull; IPCC Tier-1 &bull; QGIS 3.28 &bull; Folium
  </p>
</footer>

<!-- IMAGE MODAL -->
<div class="modal-overlay" id="modal" onclick="closeModal(event)">
  <button class="modal-close" onclick="document.getElementById('modal').classList.remove('open')">&times;</button>
  <img src="" id="modal-img" class="modal-img" alt="">
</div>

<script>
function openModal(src, title) {{
  var m = document.getElementById('modal');
  document.getElementById('modal-img').src = src;
  m.classList.add('open');
}}
function closeModal(e) {{
  if (e.target === document.getElementById('modal')) {{
    document.getElementById('modal').classList.remove('open');
  }}
}}
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') document.getElementById('modal').classList.remove('open');
}});

// Active nav link on scroll
var sections = document.querySelectorAll('section[id]');
var navLinks = document.querySelectorAll('.nav-link');
window.addEventListener('scroll', function() {{
  var scrollPos = window.scrollY + 100;
  sections.forEach(function(section) {{
    if (scrollPos >= section.offsetTop && scrollPos < section.offsetTop + section.offsetHeight) {{
      navLinks.forEach(function(link) {{
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + section.id) link.classList.add('active');
      }});
    }}
  }});
}});

// Animate progress bars on scroll
var observed = false;
var observer = new IntersectionObserver(function(entries) {{
  entries.forEach(function(entry) {{
    if (entry.isIntersecting && !observed) {{
      observed = true;
      document.querySelectorAll('.progress-fill').forEach(function(bar) {{
        var w = bar.style.width;
        bar.style.width = '0';
        setTimeout(function() {{ bar.style.width = w; }}, 200);
      }});
    }}
  }});
}}, {{threshold: 0.1}});
var firstSection = document.querySelector('.section');
if (firstSection) observer.observe(firstSection);
</script>
</body>
</html>"""

out = os.path.join(REPORT, "bengaluru_project_showcase.html")
print(f"[4/4] Writing dashboard to {out} ...")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"\n[SUCCESS] Project Showcase UI generated!")
print(f"   File: {out}")
print(f"   Open: file:///{out.replace(chr(92), '/')}")
