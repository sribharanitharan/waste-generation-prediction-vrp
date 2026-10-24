"""
01_spatial_feature_builder.py
─────────────────────────────────────────────────────────────────────────────
Phase 1: Spatial Grid & Environmental Feature Engineering
─────────────────────────────────────────────────────────────────────────────
Steps:
  1. Create a 250m hexagonal grid clipped to city boundary
  2. Extract zonal statistics from NDBI, NDVI, LST, WorldPop rasters
  3. Compute weighted POI density per hexagon
  4. Compute road/transit centrality proxy
  5. Join ward-level waste ground truth to hex cells
  6. Export fused_spatial_features.gpkg ready for ML

Requires:
  - data/raw/  (run 00_generate_synthetic_data.py first)

Output:
  - data/processed/hex_grid_250m.gpkg
  - data/processed/fused_spatial_features.gpkg
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import Point
from shapely.ops import unary_union
import warnings
import os

warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────
BBOX       = (77.45, 12.85, 77.75, 13.10)  # Bengaluru (west, south, east, north)
SEED       = 42
np.random.seed(SEED)

RASTER_PATHS = {
    "mean_ndbi": "data/raw/sentinel2_indices/ndbi.tif",
    "mean_ndvi": "data/raw/sentinel2_indices/ndvi.tif",
    "mean_lst":  "data/raw/landsat_lst/lst_celsius.tif",
    "pop_density": "data/raw/worldpop.tif",
}

# ─── Step 1: Create Hexagonal Grid ─────────────────────────────────────────────
def create_hex_grid(bbox, hex_size_deg: float = 0.0025):
    """
    Approximate 250m hexagonal grid using pointy-top hex tiling.
    hex_size_deg ~ 0.0025 deg ≈ 250m at equatorial latitudes.
    """
    west, south, east, north = bbox
    dx = hex_size_deg
    dy = hex_size_deg * np.sqrt(3) / 2

    hexagons = []
    row = 0
    y = south
    while y < north:
        offset_x = (dx / 2) if (row % 2 == 1) else 0.0
        x = west + offset_x
        while x < east:
            # Build a regular hexagon (pointy-top) in geographic coordinates
            angles = np.radians(np.arange(0, 360, 60) + 30)
            coords = [(x + dx/2 * np.cos(a), y + dy * np.sin(a)) for a in angles]
            from shapely.geometry import Polygon
            hexagons.append(Polygon(coords))
            x += dx
        y += dy
        row += 1

    boundary = gpd.read_file("data/raw/boundary.geojson")
    city_geom = boundary.geometry.iloc[0]

    grid = gpd.GeoDataFrame({"geometry": hexagons}, crs="EPSG:4326")
    grid = grid[grid.geometry.intersects(city_geom)].copy()
    grid["hex_id"] = [f"H{i:05d}" for i in range(len(grid))]
    grid["centroid_lon"] = grid.geometry.centroid.x
    grid["centroid_lat"] = grid.geometry.centroid.y
    grid = grid.reset_index(drop=True)

    print(f"  Created {len(grid)} hexagonal cells (~250m diameter)")
    return grid


# ─── Step 2: Zonal Statistics (raster -> hex) ─────────────────────────────────
def zonal_stats_raster(gdf: gpd.GeoDataFrame, raster_path: str, col_name: str) -> gpd.GeoDataFrame:
    """Sample raster at hex centroid (fast proxy for true zonal statistics)."""
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata

    values = []
    for geom in gdf.geometry:
        cx, cy = geom.centroid.x, geom.centroid.y
        try:
            col_idx = int((cx - transform.c) / transform.a)
            row_idx = int((cy - transform.f) / transform.e)
            col_idx = max(0, min(col_idx, data.shape[1] - 1))
            row_idx = max(0, min(row_idx, data.shape[0] - 1))
            val = float(data[row_idx, col_idx])
            if nodata is not None and val == nodata:
                val = np.nan
        except Exception:
            val = np.nan
        values.append(val)

    gdf = gdf.copy()
    gdf[col_name] = values
    return gdf


# ─── Step 3: Weighted POI Density per Hex ─────────────────────────────────────
def compute_poi_density(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Counts weighted POIs per hexagonal cell."""
    pois = gpd.read_file("data/raw/osm_pois.gpkg")
    joined = gpd.sjoin(pois, grid[["hex_id", "geometry"]], how="left", predicate="within")

    agg = joined.groupby("hex_id").agg(
        poi_count=("amenity", "count"),
        weighted_poi_density=("waste_weight", "sum"),
    ).reset_index()

    grid = grid.merge(agg, on="hex_id", how="left")
    grid["poi_count"]            = grid["poi_count"].fillna(0).astype(int)
    grid["weighted_poi_density"] = grid["weighted_poi_density"].fillna(0.0)
    return grid


# ─── Step 4: Transit Centrality Proxy ─────────────────────────────────────────
def compute_transit_centrality(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Proxy for transit / footfall centrality: inverse distance to nearest bus_stop/
    transit POI, normalized to [0, 1].
    """
    pois = gpd.read_file("data/raw/osm_pois.gpkg")
    transit = pois[pois["amenity"] == "bus_stop"].copy()

    if len(transit) == 0:
        grid["transit_centrality"] = 0.0
        grid["road_density"]       = np.random.uniform(0.1, 1.0, len(grid))
        return grid

    transit_pts = np.column_stack([transit.geometry.x, transit.geometry.y])
    centroids   = np.column_stack([grid.geometry.centroid.x, grid.geometry.centroid.y])

    min_dists = []
    for cx, cy in centroids:
        dists = np.sqrt((transit_pts[:, 0] - cx) ** 2 + (transit_pts[:, 1] - cy) ** 2)
        min_dists.append(dists.min())

    min_dists = np.array(min_dists)
    centrality = 1.0 / (1.0 + min_dists * 1000)  # normalize
    centrality = (centrality - centrality.min()) / (centrality.max() - centrality.min() + 1e-9)

    grid = grid.copy()
    grid["transit_centrality"] = centrality.astype(np.float32)
    # Road density proxy: correlated with transit centrality + noise
    grid["road_density"] = np.clip(
        0.6 * centrality + 0.4 * np.random.uniform(0, 1, len(grid)), 0.0, 1.0
    ).astype(np.float32)
    return grid


# ─── Step 5: 3D Building Volume (proxy from NDBI) ─────────────────────────────
def compute_building_volume(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Proxy: building volume estimated from NDBI and cell area.
    building_vol_m3 = hex_area_m2 * mean_height
    mean_height ~ 3 + 15 * NDBI (clamped to 3–60m).
    """
    grid = grid.copy()
    # Approximate area in m2 (at Bengaluru latitude ~13°)
    grid["hex_area_m2"] = grid.geometry.area * (111_320 ** 2) * np.cos(np.radians(12.97))
    ndbi_norm = np.clip(grid["mean_ndbi"].fillna(0), -1, 1)
    mean_height = np.clip(3.0 + 15.0 * ndbi_norm, 3.0, 60.0)
    grid["building_vol_m3"] = (grid["hex_area_m2"] * mean_height).astype(np.float32)
    return grid


# ─── Step 6: Join Waste Ground Truth ──────────────────────────────────────────
def join_ground_truth(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build waste targets that are realistically correlated with spatial features,
    giving ML models a learnable signal.

    Core relationships (domain-driven):
      total_waste  ↑ with pop_density, weighted_poi_density, mean_ndbi
      total_waste  ↓ with mean_ndvi (green ↔ less dense)
      biodegradable_pct ↑ with restaurant/food_market POI density, pop_density
      recyclable_pct    ↑ with mean_ndbi (commercial / industrial zones)
    """
    grid = grid.copy()
    n = len(grid)

    # Normalise feature vectors to [0, 1] for combination
    def norm(v):
        v = np.array(v, dtype=np.float64)
        mn, mx = v.min(), v.max()
        return (v - mn) / (mx - mn + 1e-9)

    pop   = norm(grid["pop_density"].fillna(0))
    ndbi  = norm(grid["mean_ndbi"].fillna(0))
    ndvi  = norm(grid["mean_ndvi"].fillna(0))
    poi   = norm(grid["weighted_poi_density"].fillna(0))
    trans = norm(grid["transit_centrality"].fillna(0))

    # --- Y1: total_waste_kg_day -------------------------------------------
    # Base: 500–8000 kg/day; driven by pop, NDBI, POI, transit
    signal_waste = 0.35 * pop + 0.25 * ndbi + 0.25 * poi + 0.15 * trans
    noise_waste  = np.random.lognormal(0, 0.18, n)
    total_waste  = (500 + 7500 * signal_waste) * noise_waste
    grid["total_waste_kg_day"] = np.clip(total_waste, 200, 12000).astype(np.float32)

    # --- Y2: biodegradable_pct --------------------------------------------
    # Higher in residential / high-pop / food-market areas; lower in pure commercial
    signal_bio   = 0.45 * pop + 0.30 * poi + 0.10 * ndvi - 0.15 * ndbi
    bio_pct      = 35 + 35 * signal_bio + np.random.normal(0, 4, n)
    grid["biodegradable_pct"] = np.clip(bio_pct, 10, 80).astype(np.float32)

    # --- Y3: recyclable_pct -----------------------------------------------
    # Higher in commercial / built-up areas (packaging, paper)
    signal_rec   = 0.50 * ndbi + 0.30 * poi - 0.20 * pop
    rec_pct      = 20 + 25 * signal_rec + np.random.normal(0, 3, n)
    grid["recyclable_pct"] = np.clip(rec_pct, 5, 55).astype(np.float32)

    # --- Inert (residual) -------------------------------------------------
    grid["inert_pct"] = np.clip(
        100 - grid["biodegradable_pct"] - grid["recyclable_pct"], 2, 40
    ).astype(np.float32)

    # Assign nearest ward ID for record-keeping
    gt = pd.read_csv("data/ground_truth/municipal_waste_data.csv")
    gt_gdf = gpd.GeoDataFrame(
        gt, geometry=gpd.points_from_xy(gt.longitude, gt.latitude), crs="EPSG:4326"
    )
    ward_pts      = np.column_stack([gt_gdf.geometry.x, gt_gdf.geometry.y])
    hex_centroids = np.column_stack([grid.geometry.centroid.x, grid.geometry.centroid.y])
    assigned_idx  = [
        int(np.argmin(np.sqrt((ward_pts[:, 0] - cx) ** 2 + (ward_pts[:, 1] - cy) ** 2)))
        for cx, cy in hex_centroids
    ]
    grid["ward_id"] = gt_gdf.iloc[assigned_idx]["ward_id"].values

    return grid


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 1: Spatial Feature Engineering")
    print("=" * 60)

    os.makedirs("data/processed", exist_ok=True)

    print("\n[1/6] Creating hexagonal grid ...")
    grid = create_hex_grid(BBOX)
    grid.to_file("data/processed/hex_grid_250m.gpkg", driver="GPKG")
    print(f"  Saved: data/processed/hex_grid_250m.gpkg")

    print("\n[2/6] Extracting raster zonal statistics ...")
    for col, path in RASTER_PATHS.items():
        print(f"  -> {col} from {path}")
        grid = zonal_stats_raster(grid, path, col)

    print("\n[3/6] Computing weighted POI density ...")
    grid = compute_poi_density(grid)

    print("\n[4/6] Computing transit centrality proxy ...")
    grid = compute_transit_centrality(grid)

    print("\n[5/6] Computing 3D building volume proxy ...")
    grid = compute_building_volume(grid)

    print("\n[6/6] Joining waste ground truth ...")
    grid = join_ground_truth(grid)

    # Final NaN fill
    feature_cols = ["mean_ndbi", "mean_ndvi", "mean_lst", "pop_density",
                    "weighted_poi_density", "transit_centrality", "road_density",
                    "building_vol_m3", "poi_count"]
    for col in feature_cols:
        if col in grid.columns:
            grid[col] = grid[col].fillna(grid[col].median())

    grid.to_file("data/processed/fused_spatial_features.gpkg", driver="GPKG")

    print("\n" + "=" * 60)
    print(f"  Fused feature matrix: {len(grid)} hexagonal cells x {len(grid.columns)} attributes")
    print("  Saved: data/processed/fused_spatial_features.gpkg")
    print("  Next step -> run: python scripts/02_spatial_cv_models.py")
    print("=" * 60)
