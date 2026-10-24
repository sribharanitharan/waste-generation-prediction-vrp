# Smart Waste Generation Prediction & Route Optimization
## Using Geospatial ML — Lab Report

**Author:** [Your Name] | **Institution:** [College Name]  
**Subject:** Spatial Analytics & GIS | **Semester:** 9th  
**Study Area:** Bengaluru Municipal Corporation (BBMP), India

---

## Abstract

This project implements an end-to-end **Geospatial Intelligence system** for predicting
municipal solid waste (MSW) generation at sub-ward resolution using a multi-modal fusion
of satellite Earth Observation (EO) data, 3D urban morphology metrics, OpenStreetMap
Points of Interest, and demographic indicators. The system employs Multi-Output XGBoost /
LightGBM / Random Forest models with Spatial Block K-Fold cross-validation, Spatial
Explainable AI (Geo-XAI via TreeSHAP), Getis-Ord Gi* hotspot detection, prescriptive
Vehicle Routing Problem (VRP) optimization, and IPCC Tier 1 greenhouse gas emission mapping.

---

## 1. Introduction & Motivation

Bengaluru generates **~5,000–6,000 tonnes of MSW per day**, yet collection efficiency
remains below 70% due to static routing, imprecise bin placement, and reactive (not
predictive) scheduling. This study transforms waste management from *reactive* to
*predictive-prescriptive* by:

1. Predicting **total waste volume** AND **composition** (organic / recyclable / inert)
   at hexagonal grid level using EO + ML.
2. Identifying **spatial hotspots** via Getis-Ord Gi* statistics.
3. Explaining *why* each zone generates its waste load via **Spatial SHAP**.
4. Prescribing **optimal truck routes** using Nearest-Neighbour VRP.
5. Quantifying **GHG impact** per zone via IPCC Tier 1 methodology.

---

## 2. Study Area

| Parameter         | Value                                     |
|:------------------|:------------------------------------------|
| City              | Bengaluru (Bangalore), Karnataka, India   |
| Bounding Box      | 77.45°E – 77.75°E, 12.85°N – 13.10°N     |
| Municipal Body    | BBMP (Bruhat Bengaluru Mahanagara Palike) |
| Administrative    | ~198 wards                                |
| Area              | ~709 km²                                  |
| Spatial Unit      | 250m Hexagonal Grid (~H3 resolution 9)   |

---

## 3. Data Sources & Feature Engineering

### 3.1 Remote Sensing Inputs

| Feature              | Source                    | Extraction Method             |
|:---------------------|:--------------------------|:------------------------------|
| NDBI                 | Sentinel-2 (SWIR/NIR)     | Raster Calculator: (B11-B8)/(B11+B8) |
| NDVI                 | Sentinel-2 (NIR/Red)      | Raster Calculator: (B8-B4)/(B8+B4) |
| Land Surface Temp    | Landsat-9 TIRS Band 10    | Split-Window Algorithm        |
| Population Density   | WorldPop 100m             | Zonal Statistics (mean)       |

### 3.2 Derived Features

| Feature                  | Derivation                                      |
|:-------------------------|:------------------------------------------------|
| 3D Building Volume       | `hex_area_m² × estimated_floor_height`          |
| Weighted POI Density     | `Σ(poi_waste_weight)` per hex via spatial join  |
| Transit Centrality       | Inverse distance to nearest transit hub         |
| Road Density             | Correlated proxy from OSM road segments         |

---

## 4. Machine Learning Methodology

### 4.1 Target Variables (Multi-Output Regression)

| Target | Symbol | Description                         |
|:-------|:-------|:------------------------------------|
| Y₁     | `total_waste_kg_day`     | Total waste generation rate (kg/day)   |
| Y₂     | `biodegradable_pct`      | Organic / biodegradable fraction (%)   |
| Y₃     | `recyclable_pct`         | Recyclable / dry fraction (%)          |

### 4.2 Model Architecture

```
MultiOutputRegressor(
    base_estimator = XGBoost / LightGBM / RandomForest
)
```

All models wrapped in `sklearn.multioutput.MultiOutputRegressor` to predict Y₁, Y₂, Y₃
simultaneously.

### 4.3 Spatial Block K-Fold Cross-Validation

Standard K-Fold leaks spatial autocorrelation across training/test folds.  
**Solution:** KMeans-partitioned spatial blocks (k=5) on hex centroid coordinates prevent
adjacent cells from appearing in both folds.

```python
coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
spatial_folds = KMeans(n_clusters=5).fit_predict(coords)
```

### 4.4 Residual Diagnostics — Moran's I

Global Moran's I computed on model residuals to verify spatial independence:

- **I ≈ 0:** Residuals are spatially random (good model fit)
- **I > 0:** Positive spatial autocorrelation in errors (model missing spatial pattern)
- **I < 0:** Negative spatial autocorrelation in errors

---

## 5. Spatial Explainable AI (Geo-XAI)

### 5.1 TreeSHAP Local Attribution

For each hexagonal cell *i*, TreeSHAP decomposes the prediction into feature contributions:

```
ŷᵢ = base_value + SHAP(pop_density)ᵢ + SHAP(ndbi)ᵢ + ... + SHAP(poi_density)ᵢ
```

### 5.2 Dominant Driver Mapping

The `primary_driver` for each cell is `argmax |SHAP values|`, revealing *which* covariate
most explains the local waste generation. This produces a categorical choropleth map
showing spatial variation in dominant waste drivers across Bengaluru.

---

## 6. Hotspot Analysis — Getis-Ord Gi*

Getis-Ord Gi* statistic identifies statistically significant spatial clusters:

$$G_i^* = \frac{\sum_j w_{ij} x_j - \bar{x}\sum_j w_{ij}}
               {S\sqrt{\frac{n\sum_j w_{ij}^2 - (\sum_j w_{ij})^2}{n-1}}}$$

| z-score threshold | Classification   |
|:------------------|:-----------------|
| z > 2.576         | Hotspot (99% CI) |
| z > 1.960         | Hotspot (95% CI) |
| z > 1.645         | Hotspot (90% CI) |
| z < -1.645        | Coldspot (90% CI)|

---

## 7. Prescriptive Routing — VRP

### 7.1 Fleet Configuration

| Parameter         | Value                          |
|:------------------|:-------------------------------|
| Trucks            | 4                              |
| Truck Capacity    | 8,000 kg/shift                 |
| Algorithm         | Nearest-Neighbour Greedy VRP   |
| Baseline          | Random ordering (naive)        |

### 7.2 Carbon Savings from Routing

```
CO2 saved = (naive_km - optimized_km) × 0.28 kg_fuel/km × 3.17 kg_CO2/kg_fuel
```

---

## 8. IPCC Tier 1 GHG Impact

Methane emission potential from organic waste fraction:

$$\text{CH}_4 \text{ (kg/yr)} = \text{MSW}_{\text{organic}} \times \text{DOC} \times \text{DOC}_f \times F \times \frac{16}{12} \times (1-\text{OX}) \times 1000$$

CO₂-equivalent = CH₄ × GWP₁₀₀(CH₄) = CH₄ × **29.8** (IPCC AR6)

| Parameter | Value  | Source              |
|:----------|:-------|:--------------------|
| DOC       | 0.17   | IPCC Table 3.3      |
| DOCf      | 0.77   | IPCC default        |
| F         | 0.50   | IPCC default        |
| OX        | 0.00   | Open dump assumption|
| GWP₁₀₀   | 29.8   | IPCC AR6            |
| India DOC | ×1.12  | Tropical adjustment |

---

## 9. Output Maps

| Map # | Filename                            | Content                                         |
|:------|:------------------------------------|:------------------------------------------------|
| 1     | `01_predicted_waste_generation.png` | Graduated choropleth — kg/day per hex cell       |
| 2     | `02_getis_ord_hotspots.png`         | Statistical hotspot/coldspot confidence clusters |
| 3     | `03_geo_shap_drivers.png`           | Geo-XAI dominant driver per cell (categorical)  |
| 4     | `04_waste_composition_breakdown.png`| Tri-panel: biodegradable / recyclable / inert %  |
| 5     | `05_optimized_collection_routes.png`| VRP truck routes + bin sites + depot             |
| 6     | `06_carbon_methane_heatmap.png`     | CH₄ potential & CO₂e dual heatmap               |

---

## 10. Pipeline Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run all stages
python run_pipeline.py

# Or run individual stages
python scripts/00_generate_synthetic_data.py
python scripts/01_spatial_feature_builder.py
python scripts/02_spatial_cv_models.py
python scripts/03_geo_xai_shap.py
python scripts/04_prescriptive_routing.py
python scripts/05_carbon_impact_calc.py

# Resume from a specific stage
python run_pipeline.py --from 3

# Run only one stage
python run_pipeline.py --only 2
```

---

## 11. QGIS Integration

All `.gpkg` outputs can be loaded directly into **QGIS 3.x**:

1. **`hex_grid_250m.gpkg`** → Base hex layer for styling
2. **`fused_spatial_features.gpkg`** → Feature matrix (use graduated symbology on `mean_ndbi`, `pop_density`)
3. **`final_predictions_with_shap.gpkg`** → Predictions + primary_driver + hotspot_class
4. **`optimized_routes.gpkg`** → LineString routes (use categorized style by `truck_id`)
5. **`bin_sites.gpkg`** → Priority bin locations (symbolize by `bin_type`)
6. **`carbon_impact.gpkg`** → GHG heatmap (`co2e_tonnes_yr` graduated)

For the **Print Layout Map Portfolio** (6 panels), load the `.gpkg` layers, apply graduated/categorized symbology, and compose the Print Layout with scale bars, north arrows, and legends.

---

## 12. References

1. IPCC (2006). *2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 5: Waste.* Chapter 3.
2. Getis, A., & Ord, J. K. (1992). The analysis of spatial association by use of distance statistics. *Geographical Analysis, 24*(3), 189–206.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
4. Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929.
5. ESA (2022). *Sentinel-2 User Handbook.* European Space Agency.
