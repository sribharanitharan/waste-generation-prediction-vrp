"""
03_geo_xai_shap.py
─────────────────────────────────────────────────────────────────────────────
Phase 3: Spatial Explainable AI (Geo-XAI via TreeSHAP)
─────────────────────────────────────────────────────────────────────────────
Steps:
  1. Load best model and predictions GeoPackage
  2. Compute TreeSHAP values for every hexagonal cell
  3. Identify the dominant local driver per cell (primary_driver)
  4. Compute Getis-Ord Gi* hotspot statistics on predicted waste
  5. Export enriched GeoPackage + 4 publication-quality map PNGs

Output:
  - data/processed/final_predictions_with_shap.gpkg
  - maps/01_predicted_waste_generation.png
  - maps/02_getis_ord_hotspots.png
  - maps/03_geo_shap_drivers.png
  - maps/04_waste_composition_breakdown.png
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import pickle
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from scipy.stats import zscore

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

DRIVER_COLORS = {
    "pop_density":           "#E74C3C",
    "mean_ndbi":             "#8E44AD",
    "mean_ndvi":             "#27AE60",
    "mean_lst":              "#E67E22",
    "building_vol_m3":       "#2980B9",
    "weighted_poi_density":  "#F39C12",
    "transit_centrality":    "#16A085",
    "road_density":          "#7F8C8D",
    "poi_count":             "#C0392B",
}
DRIVER_LABELS = {
    "pop_density":           "Population Density",
    "mean_ndbi":             "Built-up Index (NDBI)",
    "mean_ndvi":             "Vegetation Cover (NDVI)",
    "mean_lst":              "Land Surface Temp (LST)",
    "building_vol_m3":       "3D Building Volume",
    "weighted_poi_density":  "Weighted POI Density",
    "transit_centrality":    "Transit Centrality",
    "road_density":          "Road Density",
    "poi_count":             "POI Count",
}


# ─── Compute SHAP Values ───────────────────────────────────────────────────────
def compute_shap(model_pkg: dict, X: np.ndarray):
    """
    Compute SHAP values using shap.TreeExplainer if available,
    fallback to permutation importance as proxy.
    """
    try:
        import shap
        model = model_pkg["model"]
        # MultiOutputRegressor wraps individual estimators
        estimator = model.estimators_[0] if hasattr(model, "estimators_") else model
        explainer   = shap.TreeExplainer(estimator)
        try:
            shap_values = explainer.shap_values(X, check_additivity=False, approximate=True)
        except TypeError:
            shap_values = explainer.shap_values(X)
        print("  Using TreeSHAP for local feature attribution")
        return shap_values
    except Exception as e:
        print(f"  TreeSHAP unavailable ({e}). Using permutation proxy ...")
        # Proxy: feature importance * X value (local attribution approximation)
        model = model_pkg["model"]
        est   = model.estimators_[0] if hasattr(model, "estimators_") else model
        fi    = getattr(est, "feature_importances_", np.ones(X.shape[1]) / X.shape[1])
        shap_proxy = X * fi[np.newaxis, :]
        return shap_proxy


from scipy.spatial import cKDTree

# ─── Getis-Ord Gi* Hotspot Analysis ──────────────────────────────────────────
def getis_ord_gi_star(values: np.ndarray, coords: np.ndarray,
                      bandwidth_deg: float = 0.012) -> np.ndarray:
    """
    Optimized Getis-Ord Gi* statistic using cKDTree spatial ball query.
    Returns z-scores: high positive = hotspot, high negative = coldspot.
    bandwidth_deg ~ 1.2km spatial kernel.
    """
    n = len(values)
    x_mean = float(values.mean())
    x_std  = float(values.std())
    if x_std < 1e-10:
        return np.zeros(n, dtype=np.float32)

    gi_z = np.zeros(n, dtype=np.float32)
    tree = cKDTree(coords)
    cutoff = 3.0 * bandwidth_deg
    neighbor_lists = tree.query_ball_point(coords, r=cutoff)
    two_bw_sq = 2.0 * (bandwidth_deg ** 2)
    s_div1 = n - 1
    s_div2 = (n - 1) ** 2

    for i in range(n):
        idx = np.array(neighbor_lists[i], dtype=int)
        dists_sq = np.sum((coords[idx] - coords[i]) ** 2, axis=1)
        w = np.exp(-dists_sq / two_bw_sq)

        w_sum = np.sum(w)
        w_sum2 = np.sum(w ** 2)
        numerator = np.sum(w * values[idx]) - x_mean * w_sum
        var_w = (w_sum2 / s_div1) - (w_sum ** 2 / s_div2)
        if var_w <= 0:
            gi_z[i] = 0.0
            continue
        S_term = np.sqrt(var_w)
        denom = x_std * np.sqrt(n * S_term + 1e-10)
        gi_z[i] = (numerator / denom) if denom > 1e-10 else 0.0

    return gi_z


def classify_hotspot(z_scores: np.ndarray) -> np.ndarray:
    """Classify Gi* z-scores into confidence bands."""
    labels = np.full(len(z_scores), "Not Significant", dtype=object)
    # Assign in ascending order so higher confidence overrides lower
    labels[z_scores >= 1.645] = "Hotspot 90%"
    labels[z_scores >= 1.960] = "Hotspot 95%"
    labels[z_scores >= 2.576] = "Hotspot 99%"
    labels[z_scores <= -1.645] = "Coldspot 90%"
    labels[z_scores <= -1.960] = "Coldspot 95%"
    labels[z_scores <= -2.576] = "Coldspot 99%"
    return labels


# ─── Map Plotting Utilities ───────────────────────────────────────────────────
STYLE = {
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "text":   "#e6edf3",
    "accent": "#58a6ff",
    "grid":   "#21262d",
}

def apply_dark_style(fig, ax, title: str, subtitle: str = ""):
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])
    ax.tick_params(colors=STYLE["text"], labelsize=7)
    ax.spines[:].set_color(STYLE["grid"])
    ax.set_xlabel("Longitude", color=STYLE["text"], fontsize=8)
    ax.set_ylabel("Latitude", color=STYLE["text"], fontsize=8)
    ax.set_title(title, color=STYLE["accent"], fontsize=13, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha="center", color=STYLE["text"], fontsize=8, alpha=0.7)


# ─── Map 1: Predicted Waste Generation ────────────────────────────────────────
def plot_waste_generation(gdf):
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_plot = gdf.copy()
    gdf_plot.plot(
        column="pred_waste_kg_day", ax=ax,
        cmap="YlOrRd", scheme="natural_breaks", k=7,
        legend=True,
        legend_kwds={"title": "kg/day", "fontsize": 8,
                     "title_fontsize": 9, "labelcolor": STYLE["text"],
                     "facecolor": STYLE["panel"], "edgecolor": STYLE["grid"]},
        edgecolor="none", linewidth=0.0,
    )
    apply_dark_style(fig, ax,
                     "Predicted Waste Generation Density",
                     "Multi-Output ML | Spatial Block K-Fold | Bengaluru, India")
    plt.tight_layout()
    path = "maps/01_predicted_waste_generation.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {path}")


# ─── Map 2: Getis-Ord Gi* Hotspots ───────────────────────────────────────────
def plot_hotspots(gdf):
    HOTSPOT_COLORS = {
        "Hotspot 99%":     "#FF1744",
        "Hotspot 95%":     "#FF6D00",
        "Hotspot 90%":     "#FFD600",
        "Not Significant": "#37474F",
        "Coldspot 90%":    "#40C4FF",
        "Coldspot 95%":    "#0091EA",
        "Coldspot 99%":    "#01579B",
    }
    ordered = ["Hotspot 99%","Hotspot 95%","Hotspot 90%",
               "Not Significant","Coldspot 90%","Coldspot 95%","Coldspot 99%"]

    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in ordered:
        subset = gdf[gdf["hotspot_class"] == cat]
        if len(subset):
            subset.plot(ax=ax, color=HOTSPOT_COLORS[cat], edgecolor="none",
                        linewidth=0.0, label=cat)

    patches = [mpatches.Patch(color=HOTSPOT_COLORS[c], label=c) for c in ordered
               if c in gdf["hotspot_class"].values]
    leg = ax.legend(handles=patches, title="Confidence Level", loc="lower right",
                    fontsize=7, title_fontsize=8, framealpha=0.85,
                    facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                    labelcolor=STYLE["text"])
    leg.get_title().set_color(STYLE["text"])

    apply_dark_style(fig, ax,
                     "Getis-Ord Gi* Hotspot & Coldspot Analysis",
                     "Statistical confidence clustering | Bengaluru, India")
    plt.tight_layout()
    path = "maps/02_getis_ord_hotspots.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {path}")


# ─── Map 3: Geo-SHAP Dominant Drivers ────────────────────────────────────────
def plot_shap_drivers(gdf):
    fig, ax = plt.subplots(figsize=(10, 8))
    drivers = gdf["primary_driver"].unique()

    for driver in drivers:
        subset = gdf[gdf["primary_driver"] == driver]
        color  = DRIVER_COLORS.get(driver, "#95A5A6")
        subset.plot(ax=ax, color=color, edgecolor="none", linewidth=0.0,
                    label=DRIVER_LABELS.get(driver, driver))

    patches = [
        mpatches.Patch(color=DRIVER_COLORS.get(d, "#95A5A6"),
                       label=DRIVER_LABELS.get(d, d))
        for d in drivers if d in DRIVER_COLORS
    ]
    leg = ax.legend(handles=patches, title="Primary Waste Driver",
                    loc="lower right", fontsize=7, title_fontsize=8,
                    framealpha=0.85, facecolor=STYLE["panel"],
                    edgecolor=STYLE["grid"], labelcolor=STYLE["text"])
    leg.get_title().set_color(STYLE["text"])

    apply_dark_style(fig, ax,
                     "Geo-XAI: Spatial SHAP Dominant Drivers",
                     "Local feature attribution per hex cell | TreeSHAP | Bengaluru, India")
    plt.tight_layout()
    path = "maps/03_geo_shap_drivers.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {path}")


# ─── Map 4: Waste Stream Composition ─────────────────────────────────────────
def plot_waste_composition(gdf):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(STYLE["bg"])

    pairs = [
        ("pred_biodegradable_pct", "Biodegradable / Organic (%)", "Greens"),
        ("pred_recyclable_pct",    "Recyclables / Dry (%)",       "Blues"),
        ("pred_inert_pct",         "Inert / Commercial (%)",      "Oranges"),
    ]

    for ax, (col, title, cmap) in zip(axes, pairs):
        gdf.plot(column=col, ax=ax, cmap=cmap, vmin=0, vmax=100,
                 edgecolor="none", linewidth=0.0)
        ax.set_facecolor(STYLE["panel"])
        ax.tick_params(colors=STYLE["text"], labelsize=6)
        ax.spines[:].set_color(STYLE["grid"])
        ax.set_title(title, color=STYLE["accent"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Longitude", color=STYLE["text"], fontsize=7)
        ax.set_ylabel("Latitude", color=STYLE["text"], fontsize=7)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=STYLE["text"], labelsize=7)
        cb.outline.set_edgecolor(STYLE["grid"])

    fig.suptitle("Predicted Waste Stream Composition Breakdown — Bengaluru",
                 color=STYLE["accent"], fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = "maps/04_waste_composition_breakdown.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 3: Spatial Explainable AI (Geo-XAI / Spatial SHAP)")
    print("=" * 60)

    os.makedirs("maps", exist_ok=True)

    print("\n[1/5] Loading model and predictions ...")
    gdf = gpd.read_file("data/processed/predictions.gpkg")
    with open("data/processed/best_model.pkl", "rb") as f:
        model_pkg = pickle.load(f)

    features = model_pkg["features"]
    X = gdf[features].fillna(0).values
    coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])

    print("\n[2/5] Computing SHAP values ...")
    shap_values = compute_shap(model_pkg, X)

    # Dominant driver per cell
    dominant_driver_idx = np.argmax(np.abs(shap_values), axis=1)
    gdf["primary_driver"] = [features[i] for i in dominant_driver_idx]

    # Store top 3 SHAP values per cell
    for rank in range(3):
        sorted_idx = np.argsort(np.abs(shap_values), axis=1)[:, -(rank + 1)]
        gdf[f"shap_driver_{rank+1}"]     = [features[i] for i in sorted_idx]
        gdf[f"shap_importance_{rank+1}"] = np.abs(shap_values)[np.arange(len(shap_values)), sorted_idx].astype(np.float32)

    print("  Primary driver distribution:")
    print(gdf["primary_driver"].value_counts().to_string())

    print("\n[3/5] Computing Getis-Ord Gi* hotspot statistics ...")
    waste_values = gdf["pred_waste_kg_day"].fillna(0).values
    gi_z = getis_ord_gi_star(waste_values, coords)
    gdf["gi_z_score"]   = gi_z.astype(np.float32)
    gdf["hotspot_class"] = classify_hotspot(gi_z)
    print("  Hotspot classification:")
    print(gdf["hotspot_class"].value_counts().to_string())

    print("\n[4/5] Exporting enriched GeoPackage ...")
    gdf.to_file("data/processed/final_predictions_with_shap.gpkg", driver="GPKG")
    print("  Saved: data/processed/final_predictions_with_shap.gpkg")

    print("\n[5/5] Generating QGIS-ready maps ...")
    plot_waste_generation(gdf)
    plot_hotspots(gdf)
    plot_shap_drivers(gdf)
    plot_waste_composition(gdf)

    print("\n" + "=" * 60)
    print("  Geo-XAI analysis complete!")
    print("  Next step -> run: python scripts/04_prescriptive_routing.py")
    print("=" * 60)
