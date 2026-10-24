"""
02_spatial_cv_models.py
─────────────────────────────────────────────────────────────────────────────
Phase 2: Disaggregated Multi-Output Machine Learning + Spatial Block K-Fold
─────────────────────────────────────────────────────────────────────────────
Steps:
  1. Load fused_spatial_features.gpkg
  2. Spatial Block K-Fold cross-validation (K=5 spatial clusters)
  3. Train Multi-Output XGBoost, LightGBM, and RandomForest models
  4. Evaluate: R2, RMSE, Spatial Moran's I on residuals
  5. Run Geographically Weighted Regression (GWR) as benchmark
  6. Select best model, compute full predictions
  7. Export predictions + spatial fold assignments back to GeoPackage

Targets:
  Y1: total_waste_kg_day
  Y2: biodegradable_pct
  Y3: recyclable_pct

Output:
  - data/processed/predictions.gpkg
  - data/processed/model_evaluation.csv
"""

import os
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Optional heavy dependencies - graceful fallback
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("  [WARN] xgboost not installed. XGBoost model will be skipped.")

# LightGBM exhibits known access violation crashes in MultiOutputRegressor on Windows Python 3.11
LGB_AVAILABLE = False

SEED = 42
np.random.seed(SEED)

FEATURES = [
    "pop_density", "mean_ndbi", "mean_ndvi", "mean_lst",
    "building_vol_m3", "weighted_poi_density", "transit_centrality",
    "road_density", "poi_count",
]
TARGETS = ["total_waste_kg_day", "biodegradable_pct", "recyclable_pct"]


# ─── Spatial Block K-Fold ─────────────────────────────────────────────────────
def spatial_block_kfold(gdf: gpd.GeoDataFrame, k: int = 5) -> np.ndarray:
    """Assign each hex cell to a spatial fold via KMeans on centroid coordinates."""
    coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    folds  = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(coords)
    return folds


# ─── Global Moran's I ─────────────────────────────────────────────────────────
def morans_i(values: np.ndarray, coords: np.ndarray, k_neighbors: int = 8) -> float:
    """
    Compute Global Moran's I statistic on a vector of residuals.
    Uses inverse-distance weights between k nearest neighbors.
    """
    n = len(values)
    if n < 10:
        return float("nan")

    v = values - values.mean()

    # Build k-NN weight matrix
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    W_sum = 0.0
    numerator = 0.0
    for i in range(n):
        for j_pos in range(1, k_neighbors + 1):
            j = indices[i, j_pos]
            d = distances[i, j_pos]
            w = 1.0 / d if d > 0 else 0.0
            numerator += w * v[i] * v[j]
            W_sum += w

    denominator = np.sum(v ** 2)
    if denominator == 0 or W_sum == 0:
        return 0.0
    moran = (n / W_sum) * (numerator / denominator)
    return float(moran)


# ─── GWR Benchmark (simplified local regression) ─────────────────────────────
def simple_gwr(X: np.ndarray, y: np.ndarray, coords: np.ndarray,
               bandwidth: float = 0.05, max_eval: int = 800) -> tuple:
    """
    Simplified Geographically Weighted Regression benchmark.
    Uses Gaussian kernel weights for local OLS at each observation.
    Optimized to compute vector-matrix products without allocating giant diagonal matrices.
    """
    from numpy.linalg import lstsq

    n = len(y)
    if n > max_eval:
        eval_indices = np.linspace(0, n - 1, max_eval, dtype=int)
    else:
        eval_indices = np.arange(n)

    preds_sub = np.zeros(len(eval_indices))
    X_aug = np.column_stack([np.ones(n), X])
    two_bw_sq = 2.0 * (bandwidth ** 2)

    for k, i in enumerate(eval_indices):
        dists_sq = np.sum((coords - coords[i]) ** 2, axis=1)
        weights = np.exp(-dists_sq / two_bw_sq)
        XtWX = X_aug.T @ (weights[:, None] * X_aug)
        XtWy = X_aug.T @ (weights * y)
        try:
            beta, _, _, _ = lstsq(XtWX, XtWy, rcond=None)
            preds_sub[k] = X_aug[i] @ beta
        except Exception:
            preds_sub[k] = y.mean()

    return preds_sub, eval_indices


# ─── Model Training & Evaluation ─────────────────────────────────────────────
def evaluate_model(name: str, model, X: np.ndarray, Y: np.ndarray,
                   folds: np.ndarray, coords: np.ndarray) -> dict:
    """Spatial Block K-Fold cross-validation metrics."""
    k = len(np.unique(folds))
    all_preds = np.zeros_like(Y)

    for fold in range(k):
        test_mask  = folds == fold
        train_mask = ~test_mask
        if train_mask.sum() < 10:
            continue
        model.fit(X[train_mask], Y[train_mask])
        all_preds[test_mask] = model.predict(X[test_mask])

    # Metrics per target
    results = {"model": name}
    for i, target in enumerate(TARGETS):
        y_true = Y[:, i]
        y_pred = all_preds[:, i]
        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        residuals = y_true - y_pred
        mi = morans_i(residuals, coords)
        results[f"r2_{target}"]   = round(r2, 4)
        results[f"rmse_{target}"] = round(rmse, 2)
        results[f"morans_i_{target}"] = round(mi, 4)

    avg_r2 = np.mean([results[f"r2_{t}"] for t in TARGETS])
    results["avg_r2"] = round(avg_r2, 4)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2: Multi-Output ML + Spatial Cross-Validation")
    print("=" * 60)

    os.makedirs("data/processed", exist_ok=True)

    print("\n[1/5] Loading fused spatial features ...")
    gdf = gpd.read_file("data/processed/fused_spatial_features.gpkg")
    print(f"  Loaded {len(gdf)} hexagonal cells with {len(gdf.columns)} attributes")

    X = gdf[FEATURES].fillna(0).values
    Y = gdf[TARGETS].fillna(0).values
    coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])

    print("\n[2/5] Assigning Spatial Block K-Fold (k=5) ...")
    folds = spatial_block_kfold(gdf, k=5)
    gdf["spatial_fold"] = folds
    print(f"  Fold distribution: {np.bincount(folds).tolist()}")

    # ─── Build Model Zoo ───────────────────────────────────────────────────
    models = {
        "RandomForest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=SEED, verbosity=0, n_jobs=-1)
        )

    print("\n[3/5] Evaluating models with Spatial Block K-Fold ...", flush=True)
    eval_results = []
    for name, model in models.items():
        print(f"  Training: {name} ...", flush=True)
        res = evaluate_model(name, model, X, Y, folds, coords)
        eval_results.append(res)
        print(f"    avg_R2={res['avg_r2']:.4f} | "
              f"R2_waste={res[f'r2_total_waste_kg_day']:.4f} | "
              f"Moran_I_waste={res[f'morans_i_total_waste_kg_day']:.4f}", flush=True)

    # GWR benchmark on total_waste only
    print("  Benchmark: GWR (Y1=total_waste) ...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gwr_preds_sub, eval_idx = simple_gwr(X_scaled, Y[:, 0], coords, bandwidth=0.04, max_eval=800)
    y_sub = Y[eval_idx, 0]
    coords_sub = coords[eval_idx]
    gwr_r2   = r2_score(y_sub, gwr_preds_sub)
    gwr_rmse = np.sqrt(mean_squared_error(y_sub, gwr_preds_sub))
    gwr_mi   = morans_i(y_sub - gwr_preds_sub, coords_sub)
    eval_results.append({
        "model": "GWR_Benchmark",
        f"r2_total_waste_kg_day": round(gwr_r2, 4),
        f"rmse_total_waste_kg_day": round(gwr_rmse, 2),
        f"morans_i_total_waste_kg_day": round(gwr_mi, 4),
        "avg_r2": round(gwr_r2, 4),
    })
    print(f"    GWR R2_waste={gwr_r2:.4f} | Moran_I={gwr_mi:.4f}")

    eval_df = pd.DataFrame(eval_results).fillna("N/A")
    eval_df.to_csv("data/processed/model_evaluation.csv", index=False)
    print("\n  Saved: data/processed/model_evaluation.csv")

    # ─── Best Model: Full Dataset Predictions ──────────────────────────────
    print("\n[4/5] Training best model on full dataset ...")
    best_model_name = max(eval_results, key=lambda r: r.get("avg_r2", -999))["model"]
    if best_model_name == "GWR_Benchmark":
        best_model_name = "RandomForest"
    best_model = models.get(best_model_name, models["RandomForest"])
    print(f"  Best model: {best_model_name}")

    best_model.fit(X, Y)
    full_preds = best_model.predict(X)

    gdf["pred_waste_kg_day"]       = full_preds[:, 0].astype(np.float32)
    gdf["pred_biodegradable_pct"]  = np.clip(full_preds[:, 1], 0, 100).astype(np.float32)
    gdf["pred_recyclable_pct"]     = np.clip(full_preds[:, 2], 0, 100).astype(np.float32)
    gdf["pred_inert_pct"]          = np.clip(100 - full_preds[:, 1] - full_preds[:, 2], 0, 100).astype(np.float32)
    gdf["residual_waste"]          = (gdf["total_waste_kg_day"] - gdf["pred_waste_kg_day"]).astype(np.float32)
    gdf["best_model"]              = best_model_name

    # Save model object for SHAP
    import pickle
    with open("data/processed/best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "features": FEATURES, "targets": TARGETS,
                     "model_name": best_model_name}, f)
    print(f"  Saved: data/processed/best_model.pkl")

    print("\n[5/5] Exporting predictions GeoPackage ...")
    gdf.to_file("data/processed/predictions.gpkg", driver="GPKG")
    print(f"  Saved: data/processed/predictions.gpkg")

    print("\n" + "=" * 60)
    print("  Model Evaluation Summary:")
    print(eval_df[["model", "avg_r2"]].to_string(index=False))
    print("\n  Next step -> run: python scripts/03_geo_xai_shap.py")
    print("=" * 60)
