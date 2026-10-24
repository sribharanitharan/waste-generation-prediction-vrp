"""
05_carbon_impact_calc.py
─────────────────────────────────────────────────────────────────────────────
Phase 5: Environmental & Climate Impact Modeling
─────────────────────────────────────────────────────────────────────────────
Applies IPCC Tier 1 Solid Waste Methodology to compute per-hex-cell
greenhouse gas (CH4 / CO2-equivalent) emission potential from predicted
organic waste fractions.

IPCC First Order Decay (FOD) for Open Dumping (India context):
  CH4_potential (tonnes CH4/yr) =
      MSW_organic (tonnes) * DOC * DOCf * F * (16/12) * (1 - OX)
Where:
  DOC  = Degradable Organic Carbon fraction in organic waste  (0.17 default)
  DOCf = Fraction of DOC that decomposes                      (0.77 IPCC default)
  F    = Fraction of CH4 in landfill gas                      (0.5 IPCC default)
  OX   = Oxidation factor (0.1 for managed, 0.0 for open dump)
  16/12 = CH4/C molecular weight ratio

CO2-equivalent: CH4 GWP100 = 29.8 (IPCC AR6)

Outputs:
  - data/processed/carbon_impact.gpkg
  - maps/06_carbon_methane_heatmap.png
  - report/carbon_impact_summary.csv
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ─── IPCC Parameters ───────────────────────────────────────────────────────────
DOC      = 0.17    # Degradable Organic Carbon (mixed MSW, IPCC Table 3.3)
DOCf     = 0.77    # Fraction of DOC decomposing (IPCC default)
F        = 0.50    # CH4 fraction in landfill gas (IPCC default)
OX       = 0.00    # Oxidation factor (0 = open dump, no methane oxidation)
CH4_C_RATIO = 16.0 / 12.0
GWP_CH4  = 29.8   # IPCC AR6 GWP100 for CH4 (including climate-carbon feedback)

# India-specific adjustment (higher DOC for tropical food waste)
INDIA_DOC_FACTOR = 1.12

STYLE = {
    "bg":    "#0d1117",
    "panel": "#161b22",
    "text":  "#e6edf3",
    "accent":"#58a6ff",
    "grid":  "#21262d",
}


# ─── IPCC Tier 1 CH4 Potential ─────────────────────────────────────────────────
def compute_ch4_potential(organic_waste_kg_day: np.ndarray) -> np.ndarray:
    """
    Compute annual CH4 potential (kg CH4/yr) per spatial unit.
    Input: organic_waste_kg_day (kg/day)
    """
    # Convert to tonnes/year
    organic_tonnes_yr = organic_waste_kg_day * 365.0 / 1000.0

    ch4_kg_yr = (
        organic_tonnes_yr
        * DOC * INDIA_DOC_FACTOR
        * DOCf
        * F
        * CH4_C_RATIO
        * (1 - OX)
        * 1000.0  # convert back to kg
    )
    return ch4_kg_yr


def compute_co2_equivalent(ch4_kg_yr: np.ndarray) -> np.ndarray:
    """Convert CH4 (kg/yr) to CO2-equivalent (kg CO2e/yr) using GWP100."""
    return ch4_kg_yr * GWP_CH4


# ─── Ward/City Aggregation ─────────────────────────────────────────────────────
def summarize_by_ward(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    ward_summary = gdf.groupby("ward_id").agg(
        total_waste_kg_day=("total_waste_kg_day", "mean"),
        organic_waste_kg_day=("organic_waste_kg_day", "sum"),
        ch4_kg_yr=("ch4_potential_kg_yr", "sum"),
        co2e_kg_yr=("co2e_kg_yr", "sum"),
        hex_count=("hex_id", "count"),
    ).reset_index()
    ward_summary["co2e_tonnes_yr"]       = ward_summary["co2e_kg_yr"] / 1000
    ward_summary["ch4_kg_per_day"]       = ward_summary["ch4_kg_yr"] / 365
    ward_summary["risk_category"]        = pd.cut(
        ward_summary["co2e_tonnes_yr"],
        bins=[0, 50, 200, 600, float("inf")],
        labels=["Low", "Moderate", "High", "Critical"]
    )
    return ward_summary


# ─── Carbon Impact Map ─────────────────────────────────────────────────────────
def plot_carbon_heatmap(gdf: gpd.GeoDataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(STYLE["bg"])

    # Panel A: CH4 potential
    ax = axes[0]
    gdf.plot(column="ch4_potential_kg_yr", ax=ax, cmap="inferno",
             scheme="quantiles", k=6, edgecolor="none", linewidth=0.0)
    ax.set_facecolor(STYLE["panel"])
    ax.tick_params(colors=STYLE["text"], labelsize=7)
    ax.spines[:].set_color(STYLE["grid"])
    ax.set_title("CH4 Emission Potential (kg/yr per hex)", color="#FF6B35",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude", color=STYLE["text"], fontsize=8)
    ax.set_ylabel("Latitude",  color=STYLE["text"], fontsize=8)
    sm_a = plt.cm.ScalarMappable(
        cmap="inferno",
        norm=Normalize(vmin=gdf["ch4_potential_kg_yr"].min(),
                       vmax=gdf["ch4_potential_kg_yr"].max())
    )
    sm_a.set_array([])
    cb_a = plt.colorbar(sm_a, ax=ax, fraction=0.046, pad=0.04)
    cb_a.ax.tick_params(colors=STYLE["text"], labelsize=7)
    cb_a.set_label("kg CH4/yr", color=STYLE["text"], fontsize=8)
    cb_a.outline.set_edgecolor(STYLE["grid"])

    # Panel B: CO2-equivalent
    ax = axes[1]
    gdf.plot(column="co2e_kg_yr", ax=ax, cmap="plasma",
             scheme="quantiles", k=6, edgecolor="none", linewidth=0.0)
    ax.set_facecolor(STYLE["panel"])
    ax.tick_params(colors=STYLE["text"], labelsize=7)
    ax.spines[:].set_color(STYLE["grid"])
    ax.set_title("CO2-Equivalent Emissions (kg CO2e/yr per hex)",
                 color="#A855F7", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude", color=STYLE["text"], fontsize=8)
    ax.set_ylabel("Latitude",  color=STYLE["text"], fontsize=8)
    sm_b = plt.cm.ScalarMappable(
        cmap="plasma",
        norm=Normalize(vmin=gdf["co2e_kg_yr"].min(), vmax=gdf["co2e_kg_yr"].max())
    )
    sm_b.set_array([])
    cb_b = plt.colorbar(sm_b, ax=axes[1], fraction=0.046, pad=0.04)
    cb_b.ax.tick_params(colors=STYLE["text"], labelsize=7)
    cb_b.set_label("kg CO2e/yr", color=STYLE["text"], fontsize=8)
    cb_b.outline.set_edgecolor(STYLE["grid"])

    # Add IPCC annotation
    fig.text(0.5, 0.01,
             f"IPCC Tier 1 FOD | DOC={DOC} | DOCf={DOCf} | F={F} | GWP100(CH4)={GWP_CH4} | India DOC adjustment ×{INDIA_DOC_FACTOR}",
             ha="center", color=STYLE["text"], fontsize=7, alpha=0.65)

    fig.suptitle("Waste-Derived GHG Emission Potential — Bengaluru Hex Grid",
                 color=STYLE["accent"], fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = "maps/06_carbon_methane_heatmap.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 5: IPCC Tier 1 GHG Carbon Impact Mapping")
    print("=" * 60)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("maps", exist_ok=True)
    os.makedirs("report", exist_ok=True)

    print("\n[1/5] Loading spatial predictions ...")
    gdf = gpd.read_file("data/processed/final_predictions_with_shap.gpkg")
    print(f"  Loaded {len(gdf)} hex cells")

    print("\n[2/5] Computing organic waste per cell ...")
    # Use predicted biodegradable fraction to compute organic waste
    gdf["organic_fraction"] = gdf["pred_biodegradable_pct"].fillna(0) / 100.0
    gdf["organic_waste_kg_day"] = (
        gdf["pred_waste_kg_day"].fillna(0) * gdf["organic_fraction"]
    ).astype(np.float32)

    print("\n[3/5] Applying IPCC Tier 1 CH4 / CO2-e calculation ...")
    ch4 = compute_ch4_potential(gdf["organic_waste_kg_day"].values)
    co2e = compute_co2_equivalent(ch4)

    gdf["ch4_potential_kg_yr"]  = ch4.astype(np.float32)
    gdf["co2e_kg_yr"]           = co2e.astype(np.float32)
    gdf["co2e_tonnes_yr"]       = (co2e / 1000).astype(np.float32)
    gdf["ch4_kg_per_sqkm_yr"]   = (
        ch4 / (gdf.geometry.area * (111_320 ** 2) * 1e-6 + 1e-9)
    ).astype(np.float32)

    city_total_ch4  = ch4.sum() / 1000  # tonnes CH4/yr
    city_total_co2e = co2e.sum() / 1000  # tonnes CO2e/yr

    print(f"\n  City-level CH4 potential:   {city_total_ch4:,.1f} tonnes CH4/yr")
    print(f"  City-level CO2-equivalent:  {city_total_co2e:,.1f} tonnes CO2e/yr")
    print(f"  Top emitting hex:           {gdf['co2e_tonnes_yr'].max():.2f} t CO2e/yr")

    print("\n[4/5] Building ward-level carbon summary ...")
    ward_summary = summarize_by_ward(gdf)
    ward_summary.to_csv("report/carbon_impact_summary.csv", index=False)
    print(f"  Saved: report/carbon_impact_summary.csv  ({len(ward_summary)} wards)")

    print("\n  Top 10 highest-emission wards:")
    top10 = ward_summary.nlargest(10, "co2e_tonnes_yr")[
        ["ward_id", "co2e_tonnes_yr", "organic_waste_kg_day", "risk_category"]
    ]
    print(top10.to_string(index=False))

    print("\n[5/5] Exporting GeoPackage and heatmap ...")
    gdf.to_file("data/processed/carbon_impact.gpkg", driver="GPKG")
    print("  Saved: data/processed/carbon_impact.gpkg")

    plot_carbon_heatmap(gdf)

    # Final pipeline summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — All phases executed successfully!")
    print("=" * 60)
    print(f"\n  City: Bengaluru, India")
    print(f"  Spatial Units: {len(gdf)} hexagonal cells (~250m diameter)")
    print(f"  Annual CH4 Potential:  {city_total_ch4:,.0f} tonnes CH4/yr")
    print(f"  Annual CO2e Potential: {city_total_co2e:,.0f} tonnes CO2e/yr")
    print(f"\n  Output Maps (maps/ directory):")
    maps = [
        "01_predicted_waste_generation.png",
        "02_getis_ord_hotspots.png",
        "03_geo_shap_drivers.png",
        "04_waste_composition_breakdown.png",
        "05_optimized_collection_routes.png",
        "06_carbon_methane_heatmap.png",
    ]
    for m in maps:
        print(f"    - maps/{m}")
    print("\n  Load GeoPackages in QGIS for interactive exploration.")
    print("=" * 60)
