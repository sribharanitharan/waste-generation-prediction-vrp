"""
generate_html_dashboard.py  (optimized — uses bulk GeoJson, not row-by-row)
─────────────────────────────────────────────────────────────────────────────
Run:    python scripts/generate_html_dashboard.py
Output: report/bengaluru_smart_waste_dashboard.html
─────────────────────────────────────────────────────────────────────────────
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm

warnings.filterwarnings("ignore")
# Ensure emoji/unicode prints safely on Windows
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC   = os.path.join(BASE, "data", "processed")
REPORT = os.path.join(BASE, "report")
os.makedirs(REPORT, exist_ok=True)

# ─── 1. LOAD ─────────────────────────────────────────────────────────────────
print("[1/5] Loading spatial data ...")
shap_gdf   = gpd.read_file(os.path.join(PROC, "final_predictions_with_shap.gpkg"))
carbon_gdf = gpd.read_file(os.path.join(PROC, "carbon_impact.gpkg"))
routes_gdf = gpd.read_file(os.path.join(PROC, "optimized_routes.gpkg"))
bins_gdf   = gpd.read_file(os.path.join(PROC, "bin_sites.gpkg"))
eval_df    = pd.read_csv(os.path.join(PROC, "model_evaluation.csv"))

for gdf in [shap_gdf, carbon_gdf, routes_gdf, bins_gdf]:
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf.to_crs(epsg=4326, inplace=True)

# Keep only needed columns to reduce GeoJSON size
shap_slim = shap_gdf[["hex_id","pred_waste_kg_day","hotspot_class","primary_driver",
                       "gi_z_score","shap_driver_1","shap_driver_2","geometry"]].copy()
carbon_slim = carbon_gdf[["hex_id","co2e_kg_yr","ch4_potential_kg_yr","co2e_tonnes_yr","geometry"]].copy()

print(f"  {len(shap_gdf):,} hex cells | {len(bins_gdf):,} bin sites | {len(routes_gdf)} truck routes")

# ─── 2. BUILD MAP ────────────────────────────────────────────────────────────
print("[2/5] Building interactive map ...")

m = folium.Map(location=[12.97, 77.59], zoom_start=11, tiles=None, prefer_canvas=True)

folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
    name="Esri Dark Canvas (Clean Dark)",
    max_zoom=16,
).add_to(m)
folium.TileLayer(
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="&copy; OpenStreetMap contributors",
    name="OpenStreetMap Standard",
    max_zoom=19,
).add_to(m)
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri",
    name="Esri Satellite",
    max_zoom=19,
).add_to(m)

# ── Color helpers ─────────────────────────────────────────────────────────────
hotspot_colors = {
    "Hotspot 99%":    "#FF1744",
    "Hotspot 95%":    "#FF6D00",
    "Hotspot 90%":    "#FFD600",
    "Not Significant":"#546E7A",
    "Coldspot 90%":   "#40C4FF",
    "Coldspot 95%":   "#0091EA",
    "Coldspot 99%":   "#01579B",
}
driver_colors = {
    "pop_density":          "#E74C3C",
    "mean_ndbi":            "#8E44AD",
    "mean_ndvi":            "#27AE60",
    "mean_lst":             "#E67E22",
    "building_vol_m3":      "#2980B9",
    "weighted_poi_density": "#F39C12",
    "transit_centrality":   "#16A085",
    "road_density":         "#7F8C8D",
}
truck_colors = {
    "TRUCK_01":"#FF1744","TRUCK_02":"#00E676","TRUCK_03":"#2979FF","TRUCK_04":"#FFD600"
}

# Graduated color for waste
vmin = float(shap_slim["pred_waste_kg_day"].quantile(0.05))
vmax = float(shap_slim["pred_waste_kg_day"].quantile(0.95))

def waste_color(v):
    t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    if t < 0.33:
        r,g,b = int(13+t/0.33*(58-13)), int(2+t/0.33*(12-2)), int(33+t/0.33*(163-33))
    elif t < 0.66:
        tt=(t-0.33)/0.33
        r,g,b = int(58+tt*(247-58)), int(12+tt*(37-12)), int(163+tt*(133-163))
    else:
        tt=(t-0.66)/0.34
        r,g,b = int(247+tt*(255-247)), int(37+tt*(214-37)), int(133+tt*(0-133))
    return f"#{r:02x}{g:02x}{b:02x}"

# Graduated color for CO2e
co2_min = float(carbon_slim["co2e_kg_yr"].quantile(0.05))
co2_max = float(carbon_slim["co2e_kg_yr"].quantile(0.95))
def co2_color(v):
    t = max(0.0, min(1.0, (v - co2_min) / (co2_max - co2_min)))
    palettes = [(26,5,51),(106,5,114),(171,131,161),(246,211,101),(253,160,133)]
    idx = min(int(t * 4), 3)
    tt = t * 4 - idx
    r1,g1,b1 = palettes[idx]
    r2,g2,b2 = palettes[idx+1]
    return f"#{int(r1+tt*(r2-r1)):02x}{int(g1+tt*(g2-g1)):02x}{int(b1+tt*(b2-b1)):02x}"

# ── LAYER A: Predicted Waste — BULK GeoJson ───────────────────────────────────
print("  [A] Predicted Waste layer ...")
shap_slim["_fill"] = shap_slim["pred_waste_kg_day"].apply(waste_color)
shap_slim["_tip"]  = (
    "<b>Hex:</b> " + shap_slim["hex_id"].astype(str) + "<br>" +
    "<b>Waste:</b> " + shap_slim["pred_waste_kg_day"].round(0).astype(int).astype(str) + " kg/day<br>" +
    "<b>Hotspot:</b> " + shap_slim["hotspot_class"].astype(str) + "<br>" +
    "<b>Driver:</b> " + shap_slim["primary_driver"].astype(str).str.replace("_"," ").str.title()
)
lg_waste = folium.FeatureGroup(name="Predicted Waste (kg/day)", show=True)
folium.GeoJson(
    shap_slim[["_fill","_tip","geometry"]].__geo_interface__,
    style_function=lambda f: {
        "fillColor": f["properties"]["_fill"],
        "color": "none", "fillOpacity": 0.72, "weight": 0,
    },
    tooltip=folium.GeoJsonTooltip(fields=["_tip"], labels=False,
                                   style="background:#1a2035;color:#e2e8f0;border:none;font-size:12px"),
).add_to(lg_waste)
lg_waste.add_to(m)

# ── LAYER B: Hotspots — BULK GeoJson (significant only) ──────────────────────
print("  [B] Gi* Hotspots layer ...")
hot_sub = shap_slim[shap_slim["hotspot_class"] != "Not Significant"].copy()
hot_sub["_fill"] = hot_sub["hotspot_class"].map(hotspot_colors).fillna("#546E7A")
hot_sub["_tip2"] = (
    "<b>" + hot_sub["hotspot_class"].astype(str) + "</b><br>" +
    "Z-score: " + hot_sub["gi_z_score"].round(2).astype(str) + "<br>" +
    "Waste: " + hot_sub["pred_waste_kg_day"].round(0).astype(int).astype(str) + " kg/day"
)
lg_hot = folium.FeatureGroup(name="Gi* Waste Hotspots", show=True)
folium.GeoJson(
    hot_sub[["_fill","_tip2","geometry"]].__geo_interface__,
    style_function=lambda f: {
        "fillColor": f["properties"]["_fill"],
        "color": "none", "fillOpacity": 0.82, "weight": 0,
    },
    tooltip=folium.GeoJsonTooltip(fields=["_tip2"], labels=False,
                                   style="background:#1a2035;color:#e2e8f0;border:none;font-size:12px"),
).add_to(lg_hot)
lg_hot.add_to(m)

# ── LAYER C: Geo-SHAP Drivers — BULK GeoJson ─────────────────────────────────
print("  [C] Geo-SHAP Drivers layer ...")
shap_slim["_dfill"] = shap_slim["primary_driver"].map(driver_colors).fillna("#455A64")
shap_slim["_dtip"]  = (
    "<b>Driver:</b> " + shap_slim["primary_driver"].astype(str).str.replace("_"," ").str.title() + "<br>" +
    "<b>SHAP #1:</b> " + shap_slim["shap_driver_1"].astype(str).str.replace("_"," ").str.title() + "<br>" +
    "<b>SHAP #2:</b> " + shap_slim["shap_driver_2"].astype(str).str.replace("_"," ").str.title()
)
lg_shap = folium.FeatureGroup(name="Geo-SHAP Dominant Drivers", show=False)
folium.GeoJson(
    shap_slim[["_dfill","_dtip","geometry"]].__geo_interface__,
    style_function=lambda f: {
        "fillColor": f["properties"]["_dfill"],
        "color": "none", "fillOpacity": 0.80, "weight": 0,
    },
    tooltip=folium.GeoJsonTooltip(fields=["_dtip"], labels=False,
                                   style="background:#1a2035;color:#e2e8f0;border:none;font-size:12px"),
).add_to(lg_shap)
lg_shap.add_to(m)

# ── LAYER D: Carbon Impact — BULK GeoJson ─────────────────────────────────────
print("  [D] Carbon Impact layer ...")
carbon_slim["_cfill"] = carbon_slim["co2e_kg_yr"].apply(co2_color)
carbon_slim["_ctip"]  = (
    "<b>CO2e:</b> " + carbon_slim["co2e_kg_yr"].round(0).astype(int).astype(str) + " kg/yr<br>" +
    "<b>CH4:</b> "  + carbon_slim["ch4_potential_kg_yr"].round(1).astype(str) + " kg/yr<br>" +
    "<b>CO2e:</b> " + carbon_slim["co2e_tonnes_yr"].round(2).astype(str) + " t/yr"
)
lg_carb = folium.FeatureGroup(name="IPCC Carbon Impact (CO2e)", show=False)
folium.GeoJson(
    carbon_slim[["_cfill","_ctip","geometry"]].__geo_interface__,
    style_function=lambda f: {
        "fillColor": f["properties"]["_cfill"],
        "color": "none", "fillOpacity": 0.78, "weight": 0,
    },
    tooltip=folium.GeoJsonTooltip(fields=["_ctip"], labels=False,
                                   style="background:#1a2035;color:#e2e8f0;border:none;font-size:12px"),
).add_to(lg_carb)
lg_carb.add_to(m)

# ── LAYER E: VRP Routes ────────────────────────────────────────────────────────
print("  [E] VRP Routes layer ...")
lg_routes = folium.FeatureGroup(name="VRP Optimized Routes", show=True)
routes_gdf["_rcol"] = routes_gdf["truck_id"].map(truck_colors).fillna("#FF1744")
routes_gdf["_rtip"] = (
    "<b>" + routes_gdf["truck_id"].astype(str) + "</b><br>" +
    "Stops: " + routes_gdf["stops"].astype(str) + "<br>" +
    "Distance: " + routes_gdf["distance_km"].round(1).astype(str) + " km<br>" +
    "Waste: " + routes_gdf["waste_collected_kg"].round(0).astype(int).astype(str) + " kg"
)
folium.GeoJson(
    routes_gdf[["_rcol","_rtip","geometry"]].__geo_interface__,
    style_function=lambda f: {
        "color": f["properties"]["_rcol"],
        "weight": 4.5, "opacity": 0.92,
    },
    tooltip=folium.GeoJsonTooltip(fields=["_rtip"], labels=False,
                                   style="background:#1a2035;color:#e2e8f0;border:none;font-size:12px"),
).add_to(lg_routes)
lg_routes.add_to(m)

# ── LAYER F: Priority Bin Sites ────────────────────────────────────────────────
print("  [F] Priority Bin Sites layer ...")
lg_bins = folium.FeatureGroup(name="Priority Bin Sites (Top 15%)", show=True)
cluster = MarkerCluster(name="Bin Cluster").add_to(lg_bins)
for _, row in bins_gdf.iterrows():
    try:
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4, color="#F39C12", fill=True,
            fill_color="#F39C12", fill_opacity=0.85, weight=1.2,
            tooltip=folium.Tooltip(
                f"<b>{row.get('bin_type','Bin')}</b><br>"
                f"Waste: {row.get('pred_waste_kg_day',0):.0f} kg/day<br>"
                f"Capacity: {row.get('recommended_capacity_kg',0):.0f} kg",
                sticky=True),
        ).add_to(cluster)
    except Exception:
        pass
lg_bins.add_to(m)

folium.LayerControl(collapsed=False, position="topright").add_to(m)
map_html = m._repr_html_()
print("  Map rendered OK")

# ─── 3. STATS PANEL ──────────────────────────────────────────────────────────
print("[3/5] Computing stats ...")

def sfmt(v, fmt):
    try: return fmt.format(float(v))
    except: return str(v)

eval_rows = ""
for _, r in eval_df.iterrows():
    best = r["model"] == "XGBoost"
    cls = ' class="best-model"' if best else ""
    eval_rows += (
        f"<tr><td{cls}>{r['model']}</td>"
        f"<td{cls}>{sfmt(r.get('r2_total_waste_kg_day',''), '{:.4f}')}</td>"
        f"<td{cls}>{sfmt(r.get('rmse_total_waste_kg_day',''), '{:.1f}')}</td>"
        f"<td{cls}>{sfmt(r.get('avg_r2',''), '{:.4f}')}</td></tr>"
    )

total_co2et = carbon_gdf["co2e_tonnes_yr"].sum()
total_ch4   = carbon_gdf["ch4_potential_kg_yr"].sum()
avg_waste   = shap_gdf["pred_waste_kg_day"].mean()
n_hot99  = (shap_gdf["hotspot_class"]=="Hotspot 99%").sum()
n_hot95  = (shap_gdf["hotspot_class"]=="Hotspot 95%").sum()
n_hot90  = (shap_gdf["hotspot_class"]=="Hotspot 90%").sum()
n_total_hot = n_hot99 + n_hot95 + n_hot90

truck_rows_html = ""
for _, row in routes_gdf.iterrows():
    tid   = str(row["truck_id"])
    color = truck_colors.get(tid, "#fff")
    truck_rows_html += (
        f'<div class="truck-row">'
        f'<span class="truck-dot" style="background:{color}"></span>'
        f'<span class="truck-name">{tid}</span>'
        f'<div class="truck-stats">'
        f'<span>{int(row.get("stops",0))} stops &middot; {float(row.get("distance_km",0)):.1f} km</span>'
        f'<span>{float(row.get("waste_collected_kg",0)):,.0f} kg waste</span>'
        f'</div></div>'
    )

driver_labels = {
    "pop_density":          ("Population Density",    "#E74C3C"),
    "mean_ndbi":            ("Built-Up Index (NDBI)", "#8E44AD"),
    "mean_ndvi":            ("Vegetation (NDVI)",     "#27AE60"),
    "mean_lst":             ("Land Surface Temp",     "#E67E22"),
    "building_vol_m3":      ("3D Building Volume",    "#2980B9"),
    "weighted_poi_density": ("POI Density",           "#F39C12"),
    "transit_centrality":   ("Transit Centrality",    "#16A085"),
    "road_density":         ("Road Density",          "#7F8C8D"),
}
driver_legend_html = ""
for key, (label, color) in driver_labels.items():
    cnt = (shap_gdf["primary_driver"] == key).sum()
    pct = cnt / len(shap_gdf) * 100
    driver_legend_html += (
        f'<div class="driver-row">'
        f'<span class="driver-dot" style="background:{color}"></span>'
        f'<span class="driver-label">{label}</span>'
        f'<div class="driver-bar-wrap"><div class="driver-bar" style="width:{pct:.1f}%;background:{color}"></div></div>'
        f'<span class="driver-pct">{pct:.0f}%</span>'
        f'</div>'
    )

# ─── 4. ASSEMBLE HTML ─────────────────────────────────────────────────────────
print("[4/5] Assembling dashboard HTML ...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Bengaluru Smart Waste - Geospatial Intelligence Dashboard</title>
<meta name="description" content="ML waste prediction, Geo-SHAP, VRP routing and IPCC GHG impact across 13,978 hexagonal cells of Bengaluru.">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#080c18;--panel:#0f172a;--card:#141e30;--border:#1e293b;--accent:#6366f1;--accent2:#f72585;--text:#e2e8f0;--muted:#64748b;--muted2:#94a3b8;--success:#10b981}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}}
.header{{background:linear-gradient(135deg,#080c18 0%,#0c1628 60%,#080c18 100%);border-bottom:1px solid var(--border);padding:14px 26px;display:flex;align-items:center;gap:14px;position:relative;overflow:hidden}}
.header::before{{content:'';position:absolute;top:-80px;left:40%;width:600px;height:200px;background:radial-gradient(ellipse,rgba(99,102,241,0.12) 0%,transparent 70%)}}
.h-icon{{font-size:36px;filter:drop-shadow(0 0 12px rgba(99,102,241,0.5));flex-shrink:0}}
.h-title{{font-size:18px;font-weight:700;background:linear-gradient(90deg,#818cf8,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-.3px}}
.h-sub{{font-size:11px;color:var(--muted);margin-top:2px}}
.badges{{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
.badge{{padding:3px 9px;border-radius:20px;font-size:10px;font-weight:600;letter-spacing:.5px}}
.bp{{background:rgba(99,102,241,.18);color:#a5b4fc;border:1px solid rgba(99,102,241,.3)}}
.bk{{background:rgba(247,37,133,.18);color:#f9a8d4;border:1px solid rgba(247,37,133,.3)}}
.bg{{background:rgba(16,185,129,.18);color:#6ee7b7;border:1px solid rgba(16,185,129,.3)}}
.bc{{background:rgba(6,182,212,.18);color:#67e8f9;border:1px solid rgba(6,182,212,.3)}}
.stats-row{{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--border);border-bottom:1px solid var(--border)}}
.stat{{background:var(--panel);padding:12px 16px;cursor:default;transition:background .2s}}
.stat:hover{{background:var(--card)}}
.stat-icon{{font-size:18px;margin-bottom:4px}}
.stat-val{{font-size:19px;font-weight:700;color:#fff;line-height:1}}
.stat-lbl{{font-size:9px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.6px}}
.main{{display:grid;grid-template-columns:1fr 295px;height:calc(100vh - 108px)}}
.map-wrap{{overflow:hidden}}
.map-wrap>div{{width:100%!important;height:100%!important}}
.side{{background:var(--panel);border-left:1px solid var(--border);overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:14px}}
.side::-webkit-scrollbar{{width:3px}}
.side::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px}}
.sec h3{{font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:9px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
.tbl{{width:100%;border-collapse:collapse;font-size:11px}}
.tbl th{{text-align:left;color:var(--muted);font-weight:500;padding:3px 4px;font-size:9px;text-transform:uppercase}}
.tbl td{{padding:5px 4px;border-top:1px solid var(--border);color:var(--text)}}
.tbl tr:hover td{{background:rgba(99,102,241,.05)}}
.best-model{{color:#34d399!important;font-weight:600}}
.truck-row{{display:flex;align-items:flex-start;gap:7px;padding:6px 0;border-bottom:1px solid var(--border);font-size:11px}}
.truck-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:3px}}
.truck-name{{font-weight:600;min-width:70px;color:#fff}}
.truck-stats{{display:flex;flex-direction:column;gap:1px;color:var(--muted2)}}
.driver-row{{display:grid;grid-template-columns:8px 1fr 64px 26px;align-items:center;gap:6px;padding:4px 0;font-size:10px}}
.driver-dot{{width:8px;height:8px;border-radius:50%}}
.driver-label{{color:var(--text)}}
.driver-bar-wrap{{background:var(--border);border-radius:4px;height:4px;overflow:hidden}}
.driver-bar{{height:100%;border-radius:4px}}
.driver-pct{{color:var(--muted);text-align:right;font-size:9px}}
.c-grid{{display:grid;grid-template-columns:1fr 1fr;gap:7px}}
.c-card{{background:rgba(247,37,133,.07);border:1px solid rgba(247,37,133,.18);border-radius:8px;padding:10px;text-align:center}}
.c-val{{font-size:15px;font-weight:700;color:#f472b6;line-height:1}}
.c-lbl{{font-size:9px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.5px}}
.hs-row{{display:flex;align-items:center;gap:7px;font-size:11px;padding:3px 0}}
.hs-swatch{{width:20px;height:10px;border-radius:2px;flex-shrink:0}}
.hs-count{{margin-left:auto;color:var(--muted);font-size:9px}}
.phase{{display:flex;align-items:center;gap:7px;padding:4px 0;font-size:11px;border-bottom:1px solid var(--border)}}
.phase-num{{width:18px;height:18px;border-radius:50%;background:var(--accent);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;flex-shrink:0;color:#fff}}
.phase-done{{margin-left:auto;color:var(--success)}}
</style>
</head>
<body>

<header class="header">
  <div class="h-icon">&#127758;</div>
  <div>
    <div class="h-title">Bengaluru Smart Waste &mdash; Geospatial Intelligence &amp; Route Optimization</div>
    <div class="h-sub">Machine Learning &middot; Geo-SHAP &middot; VRP Routing &middot; IPCC Tier-1 GHG &middot; Spatial Block CV &middot; Bengaluru, Karnataka</div>
  </div>
  <div class="badges">
    <span class="badge bp">EPSG:4326</span>
    <span class="badge bk">XGBoost + RF</span>
    <span class="badge bg">Spatial K-Fold</span>
    <span class="badge bc">IPCC Tier-1</span>
  </div>
</header>

<div class="stats-row">
  <div class="stat"><div class="stat-icon">&#128311;</div><div class="stat-val">{len(shap_gdf):,}</div><div class="stat-lbl">Hex Grid Cells</div></div>
  <div class="stat"><div class="stat-icon">&#128293;</div><div class="stat-val">{n_total_hot:,}</div><div class="stat-lbl">Waste Hotspots</div></div>
  <div class="stat"><div class="stat-icon">&#128205;</div><div class="stat-val">{len(bins_gdf):,}</div><div class="stat-lbl">Priority Bin Sites</div></div>
  <div class="stat"><div class="stat-icon">&#128665;</div><div class="stat-val">4</div><div class="stat-lbl">Optimised Trucks</div></div>
  <div class="stat"><div class="stat-icon">&#9729;</div><div class="stat-val">{total_co2et/1000:.1f}k</div><div class="stat-lbl">tonnes CO2e / yr</div></div>
  <div class="stat"><div class="stat-icon">&#127774;</div><div class="stat-val">{avg_waste:,.0f}</div><div class="stat-lbl">avg kg/day/cell</div></div>
</div>

<div class="main">
  <div class="map-wrap" id="mapContainer">
    {map_html}
  </div>

  <aside class="side">

    <div class="sec">
      <h3>ML Model Evaluation (Spatial CV)</h3>
      <table class="tbl">
        <thead><tr><th>Model</th><th>R2</th><th>RMSE</th><th>Avg R2</th></tr></thead>
        <tbody>{eval_rows}</tbody>
      </table>
      <p style="font-size:9px;color:var(--muted);margin-top:5px">Spatial Block K-Fold &middot; target: total_waste_kg_day</p>
    </div>

    <div class="sec">
      <h3>IPCC Tier-1 GHG Impact</h3>
      <div class="c-grid">
        <div class="c-card"><div class="c-val">{total_co2et/1000:.1f}k</div><div class="c-lbl">tonnes CO2e/yr</div></div>
        <div class="c-card"><div class="c-val">{total_ch4/1000:.1f}k</div><div class="c-lbl">kg CH4/yr</div></div>
      </div>
    </div>

    <div class="sec">
      <h3>VRP Greedy Routing</h3>
      {truck_rows_html}
    </div>

    <div class="sec">
      <h3>Geo-SHAP Feature Dominance</h3>
      {driver_legend_html}
    </div>

    <div class="sec">
      <h3>Getis-Ord Gi* Legend</h3>
      <div class="hs-row"><div class="hs-swatch" style="background:#FF1744"></div><span>Hotspot 99%</span><span class="hs-count">{n_hot99:,} cells</span></div>
      <div class="hs-row"><div class="hs-swatch" style="background:#FF6D00"></div><span>Hotspot 95%</span><span class="hs-count">{n_hot95:,} cells</span></div>
      <div class="hs-row"><div class="hs-swatch" style="background:#FFD600"></div><span>Hotspot 90%</span><span class="hs-count">{n_hot90:,} cells</span></div>
      <div class="hs-row"><div class="hs-swatch" style="background:#546E7A"></div><span>Not Significant</span></div>
      <div class="hs-row"><div class="hs-swatch" style="background:#40C4FF"></div><span>Coldspot 90%</span></div>
      <div class="hs-row"><div class="hs-swatch" style="background:#0091EA"></div><span>Coldspot 95%</span></div>
      <div class="hs-row"><div class="hs-swatch" style="background:#01579B"></div><span>Coldspot 99%</span></div>
    </div>

    <div class="sec">
      <h3>Pipeline Phases</h3>
      <div class="phase"><div class="phase-num">1</div><span>Synthetic Data + Hex Grid</span><span class="phase-done">&#10003;</span></div>
      <div class="phase"><div class="phase-num">2</div><span>Spatial Block K-Fold ML</span><span class="phase-done">&#10003;</span></div>
      <div class="phase"><div class="phase-num">3</div><span>Geo-SHAP + Gi* Hotspots</span><span class="phase-done">&#10003;</span></div>
      <div class="phase"><div class="phase-num">4</div><span>Greedy VRP Routing</span><span class="phase-done">&#10003;</span></div>
      <div class="phase" style="border:none"><div class="phase-num">5</div><span>IPCC GHG Carbon Impact</span><span class="phase-done">&#10003;</span></div>
    </div>

  </aside>
</div>

<script>
(function(){{
  var w=document.getElementById('mapContainer');
  if(w){{[].forEach.call(w.children,function(c){{c.style.width='100%';c.style.height='100%';}})}}
}})();
</script>
</body>
</html>"""

out = os.path.join(REPORT, "bengaluru_smart_waste_dashboard.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[5/5] Dashboard saved OK")
print(f"      {out}")
print(f"\nOpen: file:///{out.replace(chr(92),'/')}")
