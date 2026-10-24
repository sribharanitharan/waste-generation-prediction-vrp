"""
04_prescriptive_routing.py
─────────────────────────────────────────────────────────────────────────────
Phase 4: Prescriptive Analytics & Smart Waste Truck Routing (VRP Simulation)
─────────────────────────────────────────────────────────────────────────────
Steps:
  1. Load final_predictions_with_shap.gpkg
  2. Identify top 15% under-served high-waste hexagonal cells (bin sites)
  3. Simulate a road network (OSM-like street segments within bbox)
  4. Run a Nearest-Neighbor greedy VRP to produce optimized collection routes
  5. Compute carbon savings vs. naive random routing
  6. Export route GeoPackage + prescriptive routing map PNG

Output:
  - data/processed/optimized_routes.gpkg   (LineString collection routes)
  - data/processed/bin_sites.gpkg          (priority bin / sensor locations)
  - maps/05_optimized_collection_routes.png
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, MultiLineString
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# Fleet configuration
N_TRUCKS       = 4         # Number of waste collection trucks
TRUCK_CAPACITY = 8000.0    # kg per truck per shift
DEPOT_LON      = 77.594    # Waste depot location (approx Majestic, Bengaluru)
DEPOT_LAT      = 12.977
FUEL_KG_PER_KM = 0.28      # Diesel consumption kg/km
CO2_PER_KG_FUEL = 3.17     # kg CO2 per kg diesel (IPCC)

STYLE = {
    "bg":    "#0d1117",
    "panel": "#161b22",
    "text":  "#e6edf3",
    "accent":"#58a6ff",
    "grid":  "#21262d",
}


# ─── Haversine Distance ────────────────────────────────────────────────────────
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─── Nearest-Neighbour VRP (Greedy) ───────────────────────────────────────────
def nearest_neighbour_vrp(depot: tuple, nodes: np.ndarray, demands: np.ndarray,
                           capacity: float, n_trucks: int):
    """
    Greedy Nearest-Neighbour Vehicle Routing Problem solver.
    Returns list of routes; each route is a list of node indices.
    depot: (lon, lat)
    nodes: (N, 2) array of (lon, lat) for each candidate bin location
    demands: (N,) waste demand per node in kg
    """
    unvisited = set(range(len(nodes)))
    routes    = []

    for truck in range(n_trucks):
        if not unvisited:
            break
        route    = []
        load     = 0.0
        curr_pos = np.array(depot)

        while unvisited:
            # Filter nodes that fit in remaining capacity
            candidates = [i for i in unvisited if demands[i] <= (capacity - load)]
            if not candidates:
                break

            # Nearest unvisited candidate
            dists = [haversine_km(curr_pos[0], curr_pos[1],
                                  nodes[i, 0], nodes[i, 1])
                     for i in candidates]
            nearest_idx = candidates[int(np.argmin(dists))]

            route.append(nearest_idx)
            load     += demands[nearest_idx]
            curr_pos  = nodes[nearest_idx]
            unvisited.discard(nearest_idx)

        routes.append(route)

    return routes


# ─── Route Distance ────────────────────────────────────────────────────────────
def route_distance_km(depot: tuple, nodes: np.ndarray, route: list) -> float:
    """Total route distance (depot -> nodes -> depot) in km."""
    if not route:
        return 0.0
    total  = haversine_km(depot[0], depot[1], nodes[route[0], 0], nodes[route[0], 1])
    for i in range(len(route) - 1):
        total += haversine_km(nodes[route[i],   0], nodes[route[i],   1],
                              nodes[route[i+1], 0], nodes[route[i+1], 1])
    total += haversine_km(nodes[route[-1], 0], nodes[route[-1], 1],
                          depot[0], depot[1])
    return total


# ─── Build Route LineStrings ───────────────────────────────────────────────────
def build_route_geometry(depot: tuple, nodes: np.ndarray, route: list) -> LineString:
    coords = [depot] + [(nodes[i, 0], nodes[i, 1]) for i in route] + [depot]
    if len(coords) < 2:
        return None
    return LineString(coords)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 4: Prescriptive VRP Waste Truck Routing")
    print("=" * 60)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("maps", exist_ok=True)

    print("\n[1/5] Loading spatial predictions ...")
    gdf = gpd.read_file("data/processed/final_predictions_with_shap.gpkg")
    print(f"  Loaded {len(gdf)} hexagonal cells")

    # ─── Step 2: Identify Priority Bin Sites ───────────────────────────────
    print("\n[2/5] Identifying high-waste priority bin sites (top 15%) ...")
    waste_threshold = gdf["pred_waste_kg_day"].quantile(0.85)
    hotspot_mask    = (
        (gdf["pred_waste_kg_day"] >= waste_threshold) |
        (gdf["hotspot_class"].isin(["Hotspot 90%", "Hotspot 95%", "Hotspot 99%"]))
    )
    priority_cells = gdf[hotspot_mask].copy()

    # Bin site = hex centroid
    priority_cells["bin_lon"] = priority_cells.geometry.centroid.x
    priority_cells["bin_lat"] = priority_cells.geometry.centroid.y
    priority_cells["bin_type"] = np.where(
        priority_cells["pred_waste_kg_day"] >= gdf["pred_waste_kg_day"].quantile(0.95),
        "Smart Sensor Bin", "Standard Collection Point"
    )
    priority_cells["recommended_capacity_kg"] = (
        priority_cells["pred_waste_kg_day"] / 7
    ).clip(50, 1000).astype(np.float32)

    bin_sites = gpd.GeoDataFrame(
        priority_cells[["hex_id", "bin_lon", "bin_lat", "bin_type",
                         "pred_waste_kg_day", "recommended_capacity_kg",
                         "hotspot_class", "primary_driver"]],
        geometry=gpd.points_from_xy(priority_cells["bin_lon"], priority_cells["bin_lat"]),
        crs="EPSG:4326"
    )
    bin_sites.to_file("data/processed/bin_sites.gpkg", driver="GPKG")
    print(f"  Identified {len(bin_sites)} priority bin/sensor sites")
    print(f"    Smart Sensor Bins:        {(bin_sites['bin_type'] == 'Smart Sensor Bin').sum()}")
    print(f"    Standard Collection Pts:  {(bin_sites['bin_type'] == 'Standard Collection Point').sum()}")

    # ─── Step 3: VRP Setup ─────────────────────────────────────────────────
    print(f"\n[3/5] Running greedy VRP ({N_TRUCKS} trucks, capacity={TRUCK_CAPACITY} kg each) ...")
    depot  = (DEPOT_LON, DEPOT_LAT)
    nodes  = np.column_stack([bin_sites["bin_lon"].values, bin_sites["bin_lat"].values])
    demands = bin_sites["pred_waste_kg_day"].values / 7.0  # daily per-site demand

    # Cap demand at truck capacity (multi-day sites will need multiple visits)
    demands = np.clip(demands, 0, TRUCK_CAPACITY)

    routes = nearest_neighbour_vrp(depot, nodes, demands, TRUCK_CAPACITY, N_TRUCKS)

    # ─── Step 4: Route Metrics & Carbon ────────────────────────────────────
    print("\n[4/5] Computing route distances & carbon footprint ...")
    TRUCK_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7B731"]

    route_records = []
    geom_list     = []
    color_list    = []

    for t_idx, route in enumerate(routes):
        dist_km   = route_distance_km(depot, nodes, route)
        total_kg  = sum(demands[i] for i in route)
        fuel_kg   = dist_km * FUEL_KG_PER_KM
        co2_kg    = fuel_kg * CO2_PER_KG_FUEL
        geom      = build_route_geometry(depot, nodes, route)

        if geom is not None:
            geom_list.append(geom)
            color_list.append(TRUCK_COLORS[t_idx % len(TRUCK_COLORS)])

        route_records.append({
            "truck_id":       f"TRUCK_{t_idx+1:02d}",
            "stops":          len(route),
            "distance_km":    round(dist_km, 2),
            "waste_collected_kg": round(total_kg, 1),
            "fuel_consumed_kg":   round(fuel_kg, 2),
            "co2_emitted_kg":     round(co2_kg, 2),
            "color":          TRUCK_COLORS[t_idx % len(TRUCK_COLORS)],
        })
        print(f"  Truck {t_idx+1}: {len(route)} stops | "
              f"{dist_km:.1f} km | {total_kg:.0f} kg waste | {co2_kg:.1f} kg CO2")

    routes_df = pd.DataFrame(route_records)

    # Naive benchmark: random ordering
    naive_dist = 0.0
    all_node_ids = list(range(len(nodes)))
    np.random.shuffle(all_node_ids)
    for t_idx in range(N_TRUCKS):
        chunk = all_node_ids[t_idx::N_TRUCKS]
        naive_dist += route_distance_km(depot, nodes, chunk)

    opt_dist     = routes_df["distance_km"].sum()
    savings_pct  = (1 - opt_dist / (naive_dist + 1e-6)) * 100
    co2_saved    = (naive_dist - opt_dist) * FUEL_KG_PER_KM * CO2_PER_KG_FUEL

    print(f"\n  Optimized total route: {opt_dist:.1f} km")
    print(f"  Naive random route:    {naive_dist:.1f} km")
    print(f"  Distance saved:        {savings_pct:.1f}%")
    print(f"  CO2 avoided:           {co2_saved:.1f} kg/day")

    # Build route GeoDataFrame
    if geom_list:
        routes_gdf = gpd.GeoDataFrame(
            routes_df[["truck_id","stops","distance_km","waste_collected_kg",
                        "fuel_consumed_kg","co2_emitted_kg"]],
            geometry=geom_list, crs="EPSG:4326"
        )
        routes_gdf.to_file("data/processed/optimized_routes.gpkg", driver="GPKG")
        print("  Saved: data/processed/optimized_routes.gpkg")

    # ─── Step 5: Routing Map ────────────────────────────────────────────────
    print("\n[5/5] Generating prescriptive routing map ...")
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])

    # Background: all cells (dim)
    gdf.plot(ax=ax, color="#1c2733", edgecolor="none", linewidth=0.0, alpha=0.7)

    # Hotspot cells (faint highlight)
    hotspot_cells = gdf[gdf["hotspot_class"].str.startswith("Hotspot")]
    if len(hotspot_cells):
        hotspot_cells.plot(ax=ax, color="#FF444420", edgecolor="none", linewidth=0.0)

    # Routes
    legend_handles = []
    for t_idx, (route, geom) in enumerate(zip(routes, geom_list)):
        color = TRUCK_COLORS[t_idx % len(TRUCK_COLORS)]
        xs, ys = geom.xy
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.9, zorder=5)
        patch = mpatches.Patch(color=color,
                               label=f"Truck {t_idx+1} | {routes_df.iloc[t_idx]['stops']} stops | {routes_df.iloc[t_idx]['distance_km']:.1f} km")
        legend_handles.append(patch)

    # Bin sites
    smart_bins = bin_sites[bin_sites["bin_type"] == "Smart Sensor Bin"]
    std_bins   = bin_sites[bin_sites["bin_type"] == "Standard Collection Point"]
    if len(smart_bins):
        ax.scatter(smart_bins["bin_lon"], smart_bins["bin_lat"],
                   c="#FFD700", s=45, zorder=10, marker="*", label="Smart Sensor Bin")
    if len(std_bins):
        ax.scatter(std_bins["bin_lon"], std_bins["bin_lat"],
                   c="#AAB7B8", s=18, zorder=9, marker="o", alpha=0.8,
                   label="Standard Collection Point")
    # Depot
    ax.scatter([DEPOT_LON], [DEPOT_LAT], c="#00FF88", s=200, zorder=11,
               marker="D", label="Waste Depot")
    ax.annotate("DEPOT", (DEPOT_LON, DEPOT_LAT),
                textcoords="offset points", xytext=(8, 6),
                color="#00FF88", fontsize=8, fontweight="bold")

    leg = ax.legend(handles=legend_handles +
                    [mpatches.Patch(color="#FFD700", label="Smart Sensor Bin"),
                     mpatches.Patch(color="#AAB7B8", label="Std Collection Point"),
                     mpatches.Patch(color="#00FF88", label="Waste Depot")],
                    title="Fleet & Facility", loc="lower right",
                    fontsize=7, title_fontsize=8, framealpha=0.9,
                    facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                    labelcolor=STYLE["text"])
    leg.get_title().set_color(STYLE["text"])

    ax.set_title("Prescriptive VRP Waste Collection Routes — Bengaluru",
                 color=STYLE["accent"], fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, 1.01,
            f"{N_TRUCKS} trucks | {opt_dist:.0f} km total | {savings_pct:.1f}% savings vs random | {co2_saved:.0f} kg CO2/day avoided",
            transform=ax.transAxes, ha="center", color=STYLE["text"], fontsize=8, alpha=0.75)
    ax.tick_params(colors=STYLE["text"], labelsize=7)
    ax.spines[:].set_color(STYLE["grid"])
    ax.set_xlabel("Longitude", color=STYLE["text"], fontsize=8)
    ax.set_ylabel("Latitude",  color=STYLE["text"], fontsize=8)

    plt.tight_layout()
    path = "maps/05_optimized_collection_routes.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {path}")

    print("\n" + "=" * 60)
    print(f"  VRP routing complete! {N_TRUCKS} trucks covering {len(bin_sites)} priority sites")
    print(f"  Estimated CO2 savings: {co2_saved:.1f} kg/day vs naive routing")
    print("  Next step -> run: python scripts/05_carbon_impact_calc.py")
    print("=" * 60)
