"""
00_generate_synthetic_data.py
─────────────────────────────────────────────────────────────────────────────
Synthetic Data Generator for Smart Waste Prediction Pipeline
─────────────────────────────────────────────────────────────────────────────
Produces realistic GeoTIFFs, GeoPackages, and CSVs that mimic:
  * Sentinel-2 NDBI / NDVI rasters
  * Landsat-9 Land Surface Temperature raster
  * WorldPop population density raster
  * OSM POIs GeoPackage
  * Municipal waste ground-truth CSV

All outputs are centered on Bengaluru (Bangalore), India.
Run this script FIRST before any other pipeline step.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from shapely.geometry import Point, box
import os

# ─── Configuration ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# Bengaluru approximate bounding box (WGS-84)
BBOX = (77.45, 12.85, 77.75, 13.10)   # (west, south, east, north)
CITY_NAME = "Bengaluru"

RAW_DIR = "data/raw"
GT_DIR  = "data/ground_truth"

# Raster resolution for synthetic data
RASTER_ROWS, RASTER_COLS = 300, 300


# ─── Helper: write single-band GeoTIFF ────────────────────────────────────────
def write_geotiff(path: str, data: np.ndarray, nodata: float = -9999.0):
    west, south, east, north = BBOX
    transform = from_bounds(west, south, east, north,
                            width=data.shape[1], height=data.shape[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(data.dtype), 1)
    print(f"  Written: {path}")


# ─── 1. Raster: NDBI ──────────────────────────────────────────────────────────
def generate_ndbi():
    """
    NDBI ranges from -1 to +1. High values (>0.1) indicate dense built-up areas.
    CBD has high NDBI; peri-urban / green areas have negative values.
    """
    base = np.random.uniform(-0.3, 0.15, (RASTER_ROWS, RASTER_COLS)).astype(np.float32)
    cy, cx = RASTER_ROWS // 2, RASTER_COLS // 2
    yy, xx = np.ogrid[:RASTER_ROWS, :RASTER_COLS]
    core_mask = (yy - cy) ** 2 + (xx - cx) ** 2 < (RASTER_ROWS * 0.18) ** 2
    base[core_mask] += np.random.uniform(0.2, 0.45, core_mask.sum()).astype(np.float32)
    base = np.clip(base, -1.0, 1.0)
    write_geotiff("data/raw/sentinel2_indices/ndbi.tif", base)
    return base


# ─── 2. Raster: NDVI ──────────────────────────────────────────────────────────
def generate_ndvi(ndbi: np.ndarray):
    """NDVI inversely correlated with NDBI."""
    ndvi = -0.55 * ndbi + np.random.uniform(-0.1, 0.1, ndbi.shape).astype(np.float32)
    ndvi = np.clip(ndvi, -0.2, 0.85)
    write_geotiff("data/raw/sentinel2_indices/ndvi.tif", ndvi)
    return ndvi


# ─── 3. Raster: Land Surface Temperature (LST) ────────────────────────────────
def generate_lst(ndbi: np.ndarray):
    """LST in Celsius. Urban heat island -> higher LST in built-up zones."""
    lst = 28.0 + 8.0 * ndbi + np.random.normal(0, 1.2, ndbi.shape).astype(np.float32)
    lst = np.clip(lst, 22.0, 45.0)
    write_geotiff("data/raw/landsat_lst/lst_celsius.tif", lst)
    return lst


# ─── 4. Raster: WorldPop Population Density ───────────────────────────────────
def generate_worldpop(ndbi: np.ndarray):
    """Population density (persons/km2). Denser where NDBI is high."""
    pop = 3000 * (0.6 * ndbi + 0.4) + np.random.exponential(800, ndbi.shape).astype(np.float32)
    pop = np.clip(pop, 100, 35000).astype(np.float32)
    write_geotiff("data/raw/worldpop.tif", pop)
    return pop


# ─── 5. Vector: OSM POIs ──────────────────────────────────────────────────────
WASTE_WEIGHTS = {
    "food_market": 5.0, "restaurant": 4.0, "supermarket": 3.0,
    "hotel": 3.0,       "hospital": 2.5,   "office": 1.5,
    "school": 1.2,      "park": 0.5,       "bus_stop": 0.3,
    "atm": 0.1,
}

def generate_osm_pois(n_pois: int = 1800):
    west, south, east, north = BBOX
    amenities = list(WASTE_WEIGHTS.keys())

    lons = np.random.uniform(west,  east,  n_pois)
    lats = np.random.uniform(south, north, n_pois)

    # Cluster more POIs toward CBD
    cluster_mask = np.random.rand(n_pois) < 0.55
    lons[cluster_mask] = np.random.normal((west + east) / 2,  0.04, cluster_mask.sum())
    lats[cluster_mask] = np.random.normal((south + north) / 2, 0.03, cluster_mask.sum())
    lons = np.clip(lons, west, east)
    lats = np.clip(lats, south, north)

    amenity_types = np.random.choice(
        amenities, n_pois,
        p=[0.12, 0.18, 0.10, 0.08, 0.07, 0.15, 0.10, 0.08, 0.08, 0.04]
    )

    gdf = gpd.GeoDataFrame({
        "amenity":      amenity_types,
        "waste_weight": [WASTE_WEIGHTS[a] for a in amenity_types],
        "name":         [f"{a.replace('_', ' ').title()} {i}" for i, a in enumerate(amenity_types)],
        "geometry":     [Point(lon, lat) for lon, lat in zip(lons, lats)],
    }, crs="EPSG:4326")

    os.makedirs(RAW_DIR, exist_ok=True)
    gdf.to_file("data/raw/osm_pois.gpkg", driver="GPKG")
    print(f"  Written: data/raw/osm_pois.gpkg  ({len(gdf)} POIs)")
    return gdf


# ─── 6. Vector: City Boundary ─────────────────────────────────────────────────
def generate_boundary():
    west, south, east, north = BBOX
    boundary = gpd.GeoDataFrame({
        "city":     [CITY_NAME],
        "geometry": [box(west, south, east, north)],
    }, crs="EPSG:4326")
    os.makedirs(RAW_DIR, exist_ok=True)
    boundary.to_file("data/raw/boundary.geojson", driver="GeoJSON")
    print("  Written: data/raw/boundary.geojson")
    return boundary


# ─── 7. CSV: Municipal Waste Ground Truth ─────────────────────────────────────
def generate_ground_truth(n_wards: int = 198):
    """Simulates ward-level waste records for Bengaluru's ~198 wards."""
    west, south, east, north = BBOX

    ward_ids  = [f"WARD_{i:03d}" for i in range(1, n_wards + 1)]
    ward_lons = np.random.uniform(west,  east,  n_wards)
    ward_lats = np.random.uniform(south, north, n_wards)

    base_waste = np.random.lognormal(mean=8.0, sigma=0.6, size=n_wards)
    total_waste = np.clip(base_waste, 800, 8000)

    organic_frac     = np.random.dirichlet([3, 2, 1], n_wards)
    biodegradable_pct = organic_frac[:, 0] * 100
    recyclable_pct    = organic_frac[:, 1] * 100
    inert_pct         = organic_frac[:, 2] * 100

    df = pd.DataFrame({
        "ward_id":               ward_ids,
        "longitude":             ward_lons,
        "latitude":              ward_lats,
        "total_waste_kg_day":    total_waste.astype(np.float32),
        "biodegradable_pct":     biodegradable_pct.astype(np.float32),
        "recyclable_pct":        recyclable_pct.astype(np.float32),
        "inert_pct":             inert_pct.astype(np.float32),
        "collection_freq_per_wk": np.random.randint(5, 15, n_wards),
        "sensor_installed":      np.random.choice([0, 1], n_wards, p=[0.7, 0.3]),
    })

    os.makedirs(GT_DIR, exist_ok=True)
    df.to_csv("data/ground_truth/municipal_waste_data.csv", index=False)
    print(f"  Written: data/ground_truth/municipal_waste_data.csv  ({len(df)} ward records)")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Synthetic Data Generator - Smart Waste Prediction System")
    print(f"  Study Area : {CITY_NAME}, India")
    print("=" * 60)

    print("\n[1/7] Generating NDBI raster ...")
    ndbi = generate_ndbi()

    print("\n[2/7] Generating NDVI raster ...")
    ndvi = generate_ndvi(ndbi)

    print("\n[3/7] Generating LST (Land Surface Temperature) raster ...")
    lst = generate_lst(ndbi)

    print("\n[4/7] Generating WorldPop population density raster ...")
    pop = generate_worldpop(ndbi)

    print("\n[5/7] Generating OSM POI vector layer ...")
    pois = generate_osm_pois()

    print("\n[6/7] Generating municipal boundary ...")
    boundary = generate_boundary()

    print("\n[7/7] Generating ward-level waste ground truth ...")
    gt = generate_ground_truth()

    print("\n" + "=" * 60)
    print("  All synthetic data generated successfully!")
    print("  Next step -> run: python scripts/01_spatial_feature_builder.py")
    print("=" * 60)
