"""
create_qgis_project.py
─────────────────────────────────────────────────────────────────────────────
Generates a complete, standalone QGIS Project file: bengaluru_smart_waste.qgs
─────────────────────────────────────────────────────────────────────────────
This file can be double-clicked directly on Windows to open all project outputs
in QGIS with hierarchical layer groups, color styling, and basemap support.
"""

import os
import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom


def make_layer_id(name: str) -> str:
    clean = name.replace(" ", "_").replace("-", "_").lower()
    return f"{clean}_{uuid.uuid4().hex[:8]}"


def create_qgis_project():
    project_path = "bengaluru_smart_waste.qgs"
    base_dir = os.path.abspath(".")

    # Vector layers definitions
    # (id, name, rel_path, geometry_type, color, stroke_color, size_or_width)
    layers = [
        # Prescriptive routes & bins
        {
            "id": "optimized_routes",
            "name": "01. Optimized Waste Truck Routes",
            "path": "data/processed/optimized_routes.gpkg",
            "type": "vector",
            "geom": "Line",
            "color": "231,76,60,255",
            "width": "1.2",
            "group": "1. Prescriptive Operations",
        },
        {
            "id": "bin_sites",
            "name": "02. Priority Smart Bin Siting",
            "path": "data/processed/bin_sites.gpkg",
            "type": "vector",
            "geom": "Point",
            "color": "243,156,18,255",
            "width": "2.5",
            "group": "1. Prescriptive Operations",
        },
        # Hotspots & Explainable AI
        {
            "id": "getis_ord_hotspots",
            "name": "03. Getis-Ord Gi* Waste Hotspots",
            "path": "data/processed/final_predictions_with_shap.gpkg",
            "type": "vector",
            "geom": "Polygon",
            "color": "255,23,68,180",
            "width": "0.1",
            "group": "2. Geo-XAI & Hotspots",
        },
        {
            "id": "geo_shap_drivers",
            "name": "04. Geo-SHAP Dominant Drivers",
            "path": "data/processed/final_predictions_with_shap.gpkg",
            "type": "vector",
            "geom": "Polygon",
            "color": "88,166,255,180",
            "width": "0.1",
            "group": "2. Geo-XAI & Hotspots",
        },
        # ML Predictions & Carbon
        {
            "id": "pred_waste_density",
            "name": "05. Predicted Waste Generation (kg/day)",
            "path": "data/processed/final_predictions_with_shap.gpkg",
            "type": "vector",
            "geom": "Polygon",
            "color": "230,126,34,180",
            "width": "0.1",
            "group": "3. Predictions & Environmental Impact",
        },
        {
            "id": "carbon_impact",
            "name": "06. IPCC Carbon & Methane Potential (t CO2e/yr)",
            "path": "data/processed/carbon_impact.gpkg",
            "type": "vector",
            "geom": "Polygon",
            "color": "155,89,182,180",
            "width": "0.1",
            "group": "3. Predictions & Environmental Impact",
        },
        # Base grids and POIs
        {
            "id": "osm_pois",
            "name": "07. OpenStreetMap POIs & Activity Centers",
            "path": "data/raw/osm_pois.gpkg",
            "type": "vector",
            "geom": "Point",
            "color": "46,204,113,255",
            "width": "1.5",
            "group": "4. Basemap & Raw Covariates",
        },
        {
            "id": "hex_grid",
            "name": "08. Hexagonal Grid Units (250m)",
            "path": "data/processed/hex_grid_250m.gpkg",
            "type": "vector",
            "geom": "Polygon",
            "color": "0,0,0,0",
            "stroke": "120,120,120,200",
            "width": "0.2",
            "group": "4. Basemap & Raw Covariates",
        },
        # Raster layers
        {
            "id": "lst_thermal",
            "name": "09. Landsat-9 Land Surface Temp (°C)",
            "path": "data/raw/landsat_lst/lst_celsius.tif",
            "type": "raster",
            "group": "4. Basemap & Raw Covariates",
        },
        {
            "id": "ndbi_builtup",
            "name": "10. Sentinel-2 NDBI Built-up Index",
            "path": "data/raw/sentinel2_indices/ndbi.tif",
            "type": "raster",
            "group": "4. Basemap & Raw Covariates",
        },
        {
            "id": "ndvi_vegetation",
            "name": "11. Sentinel-2 NDVI Vegetation Index",
            "path": "data/raw/sentinel2_indices/ndvi.tif",
            "type": "raster",
            "group": "4. Basemap & Raw Covariates",
        },
    ]

    # Build XML
    qgis_elem = ET.Element("qgis", {
        "projectname": "Bengaluru Smart Waste Geospatial Intelligence",
        "version": "3.28.0",
    })

    title_elem = ET.SubElement(qgis_elem, "title")
    title_elem.text = "Bengaluru Smart Waste Geospatial Intelligence & Route Optimization"

    # Project CRS (WGS 84 / EPSG:4326)
    crs_elem = ET.SubElement(qgis_elem, "projectCrs")
    srs_elem = ET.SubElement(crs_elem, "spatialrefsys")
    ET.SubElement(srs_elem, "authid").text = "EPSG:4326"
    ET.SubElement(srs_elem, "description").text = "WGS 84"
    ET.SubElement(srs_elem, "projectionacronym").text = "longlat"
    ET.SubElement(srs_elem, "ellipsoidacronym").text = "EPSG:7030"
    ET.SubElement(srs_elem, "geographicflag").text = "true"

    # Layer tree group
    tree_elem = ET.SubElement(qgis_elem, "layer-tree-group")
    ET.SubElement(tree_elem, "customproperties")

    # Map groups
    groups = [
        "1. Prescriptive Operations",
        "2. Geo-XAI & Hotspots",
        "3. Predictions & Environmental Impact",
        "4. Basemap & Raw Covariates",
    ]
    group_nodes = {}
    for g_name in groups:
        g_elem = ET.SubElement(tree_elem, "layer-tree-group", {
            "name": g_name,
            "checked": "Qt::Checked",
            "expanded": "1",
        })
        group_nodes[g_name] = g_elem

    # Maplayers root
    maplayers_elem = ET.SubElement(qgis_elem, "maplayers")

    for l in layers:
        layer_uuid = make_layer_id(l["id"])
        rel_path = l["path"].replace("\\", "/")
        abs_path = os.path.join(base_dir, rel_path).replace("\\", "/")

        # Add to tree
        g_elem = group_nodes[l["group"]]
        ET.SubElement(g_elem, "layer-tree-layer", {
            "id": layer_uuid,
            "name": l["name"],
            "source": abs_path,
            "providerKey": "ogr" if l["type"] == "vector" else "gdal",
            "checked": "Qt::Checked",
            "expanded": "0",
        })

        # Add maplayer element
        if l["type"] == "vector":
            ml = ET.SubElement(maplayers_elem, "maplayer", {
                "type": "vector",
                "geometry": l["geom"],
                "wktType": l["geom"],
                "refreshOnNotifyEnabled": "0",
                "autoRefreshTime": "0",
                "hasScaleBasedVisibilityFlag": "0",
                "maxScale": "0",
                "minScale": "1e+08",
            })
            ET.SubElement(ml, "id").text = layer_uuid
            ET.SubElement(ml, "datasource").text = abs_path
            ET.SubElement(ml, "layername").text = l["name"]
            ET.SubElement(ml, "srs")
            srs = ml.find("srs")
            spatialref = ET.SubElement(srs, "spatialrefsys")
            ET.SubElement(spatialref, "authid").text = "EPSG:4326"
            ET.SubElement(spatialref, "description").text = "WGS 84"

            ET.SubElement(ml, "provider", {"encoding": "UTF-8"}).text = "ogr"

            # Simple default renderer
            renderer = ET.SubElement(ml, "renderer-v2", {
                "type": "singleSymbol",
                "forceraster": "0",
                "symbollevels": "0",
            })
            symbols = ET.SubElement(renderer, "symbols")
            sym = ET.SubElement(symbols, "symbol", {
                "name": "0",
                "type": "marker" if l["geom"] == "Point" else ("line" if l["geom"] == "Line" else "fill"),
                "alpha": "1",
                "clip_to_extent": "1",
            })
            layer_sym = ET.SubElement(sym, "layer", {
                "class": "SimpleMarker" if l["geom"] == "Point" else ("SimpleLine" if l["geom"] == "Line" else "SimpleFill"),
                "pass": "0",
                "locked": "0",
            })
            color = l.get("color", "100,100,100,255")
            stroke = l.get("stroke", "50,50,50,255")
            width = l.get("width", "0.26")
            ET.SubElement(layer_sym, "prop", {"k": "color", "v": color})
            ET.SubElement(layer_sym, "prop", {"k": "outline_color", "v": stroke})
            ET.SubElement(layer_sym, "prop", {"k": "outline_width" if l["geom"] == "Polygon" else "line_width", "v": width})
            if l["geom"] == "Point":
                ET.SubElement(layer_sym, "prop", {"k": "size", "v": width})
        else:
            # Raster
            ml = ET.SubElement(maplayers_elem, "maplayer", {
                "type": "raster",
                "autoRefreshTime": "0",
                "hasScaleBasedVisibilityFlag": "0",
                "maxScale": "0",
                "minScale": "1e+08",
            })
            ET.SubElement(ml, "id").text = layer_uuid
            ET.SubElement(ml, "datasource").text = abs_path
            ET.SubElement(ml, "layername").text = l["name"]
            ET.SubElement(ml, "provider").text = "gdal"

            pipe = ET.SubElement(ml, "pipe")
            ET.SubElement(pipe, "provider").text = "gdal"
            raster_renderer = ET.SubElement(pipe, "rasterrenderer", {
                "type": "singlebandpseudocolor",
                "band": "1",
                "opacity": "0.75",
            })
            raster_shader = ET.SubElement(raster_renderer, "rastershader")
            ET.SubElement(raster_shader, "colorrampshader", {
                "colorRampType": "INTERPOLATED",
                "clip": "0",
            })

    # Save formatted XML
    xml_str = ET.tostring(qgis_elem, encoding="utf-8")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    with open(project_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"Generated QGIS Project file: {project_path}")
    print(f"Contains {len(layers)} mapped layers across 4 structured categories.")


if __name__ == "__main__":
    create_qgis_project()
