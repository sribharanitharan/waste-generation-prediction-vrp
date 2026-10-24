"""
load_into_qgis_console.py
─────────────────────────────────────────────────────────────────────────────
PyQGIS Automation Script for Interactive Output Exploration
High-Aesthetics Cartography Edition (Cyber-Dark & Clean GIS Styling)
─────────────────────────────────────────────────────────────────────────────
How to use in QGIS:
  1. Open QGIS.
  2. Press Ctrl+Alt+P (or go to Plugins -> Python Console).
  3. Click the "Show Editor" icon in the console toolbar.
  4. Open or paste this file (load_into_qgis_console.py) and click the green "Run Script" button.
  5. All project layers will load automatically with professional thematic styling!
"""

import os
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsGraduatedSymbolRenderer,
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsRendererRange,
    QgsSymbol,
    QgsSimpleFillSymbolLayer,
    QgsSimpleLineSymbolLayer,
    QgsSimpleMarkerSymbolLayer,
    QgsClassificationQuantile,
    QgsClassificationJenks,
    QgsColorRampShader,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsCoordinateReferenceSystem,
    QgsRectangle,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from qgis.utils import iface

# Base directory for the project
BASE_DIR = r"c:\clg\9thsem\projects\Spatial"

def run():
    proj = QgsProject.instance()
    root = proj.layerTreeRoot()
    root.clear()

    print("Loading Bengaluru Smart Waste Intelligence layers with high-aesthetics cartography...")

    # ─────────────────────────────────────────────────────────────
    # GROUP 1: Prescriptive Operations
    # ─────────────────────────────────────────────────────────────
    grp_presc = root.addGroup("1. Prescriptive Waste Routing & Bin Siting")

    # Layer 01: VRP Collection Routes (Glowing Neon)
    routes_path = os.path.join(BASE_DIR, "data", "processed", "optimized_routes.gpkg")
    if os.path.exists(routes_path):
        vlayer_routes = QgsVectorLayer(routes_path, "01. VRP Optimized Collection Routes", "ogr")
        if vlayer_routes.isValid():
            route_styles = [
                ("TRUCK_01", QColor("#FF2A6D"), "Truck 01 (Route A - 12.2 km)"),
                ("TRUCK_02", QColor("#00E676"), "Truck 02 (Route B - 11.3 km)"),
                ("TRUCK_03", QColor("#00F0FF"), "Truck 03 (Route C - 12.3 km)"),
                ("TRUCK_04", QColor("#FFD600"), "Truck 04 (Route D - 10.4 km)"),
            ]
            categories = []
            for t_id, col, label in route_styles:
                sym = QgsSymbol.defaultSymbol(vlayer_routes.geometryType())
                sym.setColor(col)
                sym.setWidth(1.8)
                if sym.symbolLayerCount() > 0:
                    sym.symbolLayer(0).setPenCapStyle(Qt.RoundCap)
                    sym.symbolLayer(0).setPenJoinStyle(Qt.RoundJoin)
                cat = QgsRendererCategory(t_id, sym, label)
                categories.append(cat)

            vlayer_routes.setRenderer(QgsCategorizedSymbolRenderer("truck_id", categories))
            proj.addMapLayer(vlayer_routes, False)
            grp_presc.addLayer(vlayer_routes)
            print("  [+] Added: 01. VRP Optimized Routes (Cyber-Neon Styling)")

    # Layer 02: Priority Bin Sites (Smart Sensors vs Standard Points)
    bins_path = os.path.join(BASE_DIR, "data", "processed", "bin_sites.gpkg")
    if os.path.exists(bins_path):
        vlayer_bins = QgsVectorLayer(bins_path, "02. Priority Smart Bin Sites (Top 15%)", "ogr")
        if vlayer_bins.isValid():
            bin_cats = []
            # Smart Sensor Bin (IoT)
            sym_smart = QgsSymbol.defaultSymbol(vlayer_bins.geometryType())
            sym_smart.setColor(QColor("#FF007F"))
            sym_smart.setSize(2.6)
            if sym_smart.symbolLayerCount() > 0:
                sym_smart.symbolLayer(0).setStrokeColor(QColor("#FFFFFF"))
                sym_smart.symbolLayer(0).setStrokeWidth(0.6)
            bin_cats.append(QgsRendererCategory("Smart Sensor Bin", sym_smart, "Smart Sensor Bin (IoT)"))

            # Standard Collection Point
            sym_std = QgsSymbol.defaultSymbol(vlayer_bins.geometryType())
            sym_std.setColor(QColor("#FFB300"))
            sym_std.setSize(1.6)
            if sym_std.symbolLayerCount() > 0:
                sym_std.symbolLayer(0).setStrokeColor(QColor("#000000"))
                sym_std.symbolLayer(0).setStrokeWidth(0.3)
            bin_cats.append(QgsRendererCategory("Standard Collection Point", sym_std, "Standard Collection Point"))

            vlayer_bins.setRenderer(QgsCategorizedSymbolRenderer("bin_type", bin_cats))
            proj.addMapLayer(vlayer_bins, False)
            node_bins = grp_presc.addLayer(vlayer_bins)
            node_bins.setItemVisibilityChecked(False)
            print("  [+] Added: 02. Priority Bin Sites (unchecked by default)")

    # ─────────────────────────────────────────────────────────────
    # GROUP 2: Geo-XAI & Hotspots
    # ─────────────────────────────────────────────────────────────
    grp_xai = root.addGroup("2. Spatial Explainable AI (Geo-SHAP) & Hotspots")

    shap_path = os.path.join(BASE_DIR, "data", "processed", "final_predictions_with_shap.gpkg")
    if os.path.exists(shap_path):
        # Layer 03: Multi-tier Getis-Ord Gi* Hotspot Confidence
        vlayer_hotspots = QgsVectorLayer(shap_path, "03. Getis-Ord Gi* Waste Hotspots", "ogr")
        if vlayer_hotspots.isValid():
            hot_config = [
                ("Hotspot 99%", QColor("#E63946"), 0.90, "Hotspot 99% Confidence (Critical)"),
                ("Hotspot 95%", QColor("#FB8500"), 0.85, "Hotspot 95% Confidence (High)"),
                ("Hotspot 90%", QColor("#FFB703"), 0.80, "Hotspot 90% Confidence (Moderate)"),
                ("Not Significant", QColor(0, 0, 0, 0), 0.0, "Not Significant (Transparent)"),
                ("Coldspot 90%", QColor("#48CAE4"), 0.75, "Coldspot 90% Confidence"),
                ("Coldspot 95%", QColor("#0077B6"), 0.80, "Coldspot 95% Confidence"),
                ("Coldspot 99%", QColor("#03045E"), 0.85, "Coldspot 99% Confidence"),
            ]
            cats = []
            for name, col, opacity, label in hot_config:
                sym = QgsSymbol.defaultSymbol(vlayer_hotspots.geometryType())
                sym.setColor(col)
                sym.setOpacity(opacity)
                if sym.symbolLayerCount() > 0:
                    sym.symbolLayer(0).setStrokeStyle(Qt.NoPen)
                    sym.symbolLayer(0).setStrokeColor(QColor(0, 0, 0, 0))
                cat = QgsRendererCategory(name, sym, label)
                cats.append(cat)

            vlayer_hotspots.setRenderer(QgsCategorizedSymbolRenderer("hotspot_class", cats))
            proj.addMapLayer(vlayer_hotspots, False)
            grp_xai.addLayer(vlayer_hotspots)
            print("  [+] Added: 03. Getis-Ord Gi* Hotspots (Transparent Neutral, Glowing Tiers)")

        # Layer 04: Geo-SHAP Driver Layer
        vlayer_drivers = QgsVectorLayer(shap_path, "04. Geo-SHAP Dominant Feature Drivers", "ogr")
        if vlayer_drivers.isValid():
            driver_colors = {
                "pop_density":           (QColor("#E74C3C"), "Population Density"),
                "mean_ndbi":             (QColor("#8E44AD"), "Built-up Index (NDBI)"),
                "mean_ndvi":             (QColor("#27AE60"), "Vegetation Cover (NDVI)"),
                "mean_lst":              (QColor("#E67E22"), "Land Surface Temperature"),
                "building_vol_m3":       (QColor("#2980B9"), "3D Building Volume"),
                "weighted_poi_density":  (QColor("#F39C12"), "Commercial POI Density"),
                "transit_centrality":    (QColor("#16A085"), "Transit Hub Centrality"),
                "road_density":          (QColor("#7F8C8D"), "Road Density"),
            }
            dcats = []
            for dname, (col, dlabel) in driver_colors.items():
                sym = QgsSymbol.defaultSymbol(vlayer_drivers.geometryType())
                sym.setColor(col)
                sym.setOpacity(0.80)
                if sym.symbolLayerCount() > 0:
                    sym.symbolLayer(0).setStrokeStyle(Qt.NoPen)
                    sym.symbolLayer(0).setStrokeColor(QColor(0, 0, 0, 0))
                cat = QgsRendererCategory(dname, sym, dlabel)
                dcats.append(cat)

            vlayer_drivers.setRenderer(QgsCategorizedSymbolRenderer("primary_driver", dcats))
            proj.addMapLayer(vlayer_drivers, False)
            node_drivers = grp_xai.addLayer(vlayer_drivers)
            node_drivers.setItemVisibilityChecked(False)
            print("  [+] Added: 04. Geo-SHAP Dominant Drivers (unchecked by default)")

    # ─────────────────────────────────────────────────────────────
    # GROUP 3: ML Predictions & Carbon Impact
    # ─────────────────────────────────────────────────────────────
    grp_preds = root.addGroup("3. Machine Learning Predictions & Carbon Impact")
    grp_preds.setItemVisibilityChecked(False)

    if os.path.exists(shap_path):
        vlayer_waste = QgsVectorLayer(shap_path, "05. Predicted Total Waste (kg/day)", "ogr")
        if vlayer_waste.isValid():
            renderer = QgsGraduatedSymbolRenderer.createRenderer(
                vlayer_waste, "pred_waste_kg_day", 5,
                QgsGraduatedSymbolRenderer.GraduatedColor,
                QgsSymbol.defaultSymbol(vlayer_waste.geometryType()),
                QgsClassificationJenks()
            )
            vlayer_waste.setRenderer(renderer)
            proj.addMapLayer(vlayer_waste, False)
            node_w = grp_preds.addLayer(vlayer_waste)
            node_w.setItemVisibilityChecked(False)
            print("  [+] Added: 05. Predicted Waste Volume (unchecked by default)")

    carbon_path = os.path.join(BASE_DIR, "data", "processed", "carbon_impact.gpkg")
    if os.path.exists(carbon_path):
        vlayer_carb = QgsVectorLayer(carbon_path, "06. IPCC GHG — Methane & CO2e (kg/yr)", "ogr")
        if vlayer_carb.isValid():
            renderer = QgsGraduatedSymbolRenderer.createRenderer(
                vlayer_carb, "co2e_kg_yr", 5,
                QgsGraduatedSymbolRenderer.GraduatedColor,
                QgsSymbol.defaultSymbol(vlayer_carb.geometryType()),
                QgsClassificationQuantile()
            )
            vlayer_carb.setRenderer(renderer)
            proj.addMapLayer(vlayer_carb, False)
            node_c = grp_preds.addLayer(vlayer_carb)
            node_c.setItemVisibilityChecked(False)
            print("  [+] Added: 06. Carbon Impact Heatmap (unchecked by default)")

    # ─────────────────────────────────────────────────────────────
    # GROUP 4: Administrative Boundaries & Base Maps
    # ─────────────────────────────────────────────────────────────
    grp_base = root.addGroup("4. Administrative Boundaries & Base Maps")

    # 07. Bengaluru BBMP Ward Boundaries (243 Wards)
    wards_path = os.path.join(BASE_DIR, "data", "raw", "bengaluru_wards.geojson")
    vlayer_wards = None
    if os.path.exists(wards_path):
        vlayer_wards = QgsVectorLayer(wards_path, "07. Bengaluru BBMP Ward Boundaries (243 Wards)", "ogr")
        if vlayer_wards.isValid():
            sym = QgsSymbol.defaultSymbol(vlayer_wards.geometryType())
            sym.setColor(QColor(0, 0, 0, 0))  # transparent fill
            if sym.symbolLayerCount() > 0:
                sym.symbolLayer(0).setStrokeColor(QColor("#64748B"))
                sym.symbolLayer(0).setStrokeWidth(0.45)
            vlayer_wards.renderer().setSymbol(sym)
            proj.addMapLayer(vlayer_wards, False)
            grp_base.addLayer(vlayer_wards)
            print("  [+] Added: 07. Bengaluru BBMP Ward Boundaries (243 Wards)")

    # 08. Karnataka State District Boundaries
    kar_path = os.path.join(BASE_DIR, "data", "raw", "karnataka_districts.geojson")
    if os.path.exists(kar_path):
        vlayer_kar = QgsVectorLayer(kar_path, "08. Karnataka State District Boundaries", "ogr")
        if vlayer_kar.isValid():
            sym = QgsSymbol.defaultSymbol(vlayer_kar.geometryType())
            sym.setColor(QColor(0, 0, 0, 0))
            if sym.symbolLayerCount() > 0:
                sym.symbolLayer(0).setStrokeColor(QColor("#475569"))
                sym.symbolLayer(0).setStrokeWidth(0.3)
                sym.symbolLayer(0).setStrokeStyle(Qt.DashLine)
            vlayer_kar.renderer().setSymbol(sym)
            proj.addMapLayer(vlayer_kar, False)
            node_kar = grp_base.addLayer(vlayer_kar)
            node_kar.setItemVisibilityChecked(False)
            print("  [+] Added: 08. Karnataka State District Boundaries (unchecked)")

    # 09. Hex Grid (250m)
    hex_path = os.path.join(BASE_DIR, "data", "processed", "hex_grid_250m.gpkg")
    if os.path.exists(hex_path):
        vlayer_hex = QgsVectorLayer(hex_path, "09. Hexagonal Analysis Grid (250m)", "ogr")
        if vlayer_hex.isValid():
            sym = QgsSymbol.defaultSymbol(vlayer_hex.geometryType())
            sym.setColor(QColor(0, 0, 0, 0))
            if sym.symbolLayerCount() > 0:
                sym.symbolLayer(0).setStrokeColor(QColor(74, 85, 104, 110))
                sym.symbolLayer(0).setStrokeWidth(0.12)
            vlayer_hex.renderer().setSymbol(sym)
            proj.addMapLayer(vlayer_hex, False)
            node_hex = grp_base.addLayer(vlayer_hex)
            node_hex.setItemVisibilityChecked(False)
            print("  [+] Added: 09. Hex Grid (250m) (unchecked)")

    # 10. Esri Dark Canvas Basemap (Recommended Dark Theme - No Watermark/Key)
    url_dark = "type=xyz&url=https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}&zmax=16&zmin=0"
    rlayer_dark = QgsRasterLayer(url_dark, "10. Esri Dark Canvas Basemap (Active)", "wms")
    if rlayer_dark.isValid():
        proj.addMapLayer(rlayer_dark, False)
        grp_base.addLayer(rlayer_dark)
        print("  [+] Added: 10. Esri Dark Canvas Basemap (Active)")

    # 11. Esri World Imagery (Satellite Alternative)
    url_sat = "type=xyz&url=https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}&zmax=19&zmin=0"
    rlayer_sat = QgsRasterLayer(url_sat, "11. Esri Satellite Basemap (unchecked)", "wms")
    if rlayer_sat.isValid():
        proj.addMapLayer(rlayer_sat, False)
        node_sat = grp_base.addLayer(rlayer_sat)
        node_sat.setItemVisibilityChecked(False)
        print("  [+] Added: 11. Esri Satellite Basemap (unchecked)")

    # 12. OpenStreetMap Standard Basemap
    url_osm = "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0"
    rlayer_osm = QgsRasterLayer(url_osm, "12. OpenStreetMap Standard Basemap", "wms")
    if rlayer_osm.isValid():
        proj.addMapLayer(rlayer_osm, False)
        node_osm = grp_base.addLayer(rlayer_osm)
        node_osm.setItemVisibilityChecked(False)
        print("  [+] Added: 12. OpenStreetMap Standard Basemap (unchecked)")

    # 13-15. Environmental Rasters
    rasters = [
        ("13. Landsat-9 Land Surface Temperature (°C)", os.path.join(BASE_DIR, "data", "raw", "landsat_lst", "lst_celsius.tif")),
        ("14. Sentinel-2 NDBI Built-up Index",          os.path.join(BASE_DIR, "data", "raw", "sentinel2_indices", "ndbi.tif")),
        ("15. Sentinel-2 NDVI Vegetation Index",        os.path.join(BASE_DIR, "data", "raw", "sentinel2_indices", "ndvi.tif")),
    ]
    for rname, rpath in rasters:
        if os.path.exists(rpath):
            rlayer = QgsRasterLayer(rpath, rname)
            if rlayer.isValid():
                proj.addMapLayer(rlayer, False)
                node_r = grp_base.addLayer(rlayer)
                node_r.setItemVisibilityChecked(False)
                print(f"  [+] Added: {rname} (unchecked)")

    # ─────────────────────────────────────────────────────────────
    # Set Project CRS and Extent
    # ─────────────────────────────────────────────────────────────
    crs = QgsCoordinateReferenceSystem("EPSG:4326")
    proj.setCrs(crs)
    bengaluru_extent = QgsRectangle(77.45, 12.85, 77.78, 13.12)

    try:
        canvas = iface.mapCanvas()
        if vlayer_wards and vlayer_wards.isValid():
            canvas.setExtent(vlayer_wards.extent())
        else:
            canvas.setExtent(bengaluru_extent)
        canvas.refresh()
    except Exception as e:
        print(f"  [!] Note: Could not set canvas extent automatically ({e})")

    # Save to native project file
    save_path = os.path.join(BASE_DIR, "bengaluru_smart_waste.qgs")
    proj.write(save_path)

    print("\n[SUCCESS] All layers loaded with high-aesthetics cartography!")
    print(f"  CRS set to: {crs.authid()}")
    print(f"  Project saved to: {save_path}")

if __name__ == "__main__":
    run()
