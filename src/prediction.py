"""
Spatial Prediction & Risk Surface Engine
Generates 2D spatial flood risk rasters, risk categorization, and KPI metrics.
"""

import numpy as np
import pandas as pd
import config
from src.model import FloodRiskModel

def assign_risk_category(risk_score: float) -> str:
    if risk_score < 25.0:
        return "LOW"
    elif risk_score < 50.0:
        return "MODERATE"
    elif risk_score < 75.0:
        return "HIGH"
    else:
        return "VERY HIGH"

def generate_spatial_prediction(df_grid: pd.DataFrame, model: FloodRiskModel) -> dict:
    """
    Performs spatial inference on grid cells and returns risk scores and KPIs.
    """
    X = df_grid[config.FEATURE_COLUMNS]
    probs = model.predict_proba(X)
    risk_scores = np.round(probs * 100.0, 1)

    # Reconstruct 2D grids (50 rows x 75 cols)
    rows = config.STUDY_AREA["grid_rows"]
    cols = config.STUDY_AREA["grid_cols"]

    risk_grid = risk_scores.reshape((rows, cols))
    prob_grid = probs.reshape((rows, cols))

    # Mask permanent river channel from novel flood reporting
    perm_water_grid = df_grid["permanent_water"].values.reshape((rows, cols))
    gt_grid = df_grid["is_flooded"].values.reshape((rows, cols))

    # Area calculation:
    # Cell size is ~500m x 500m = 0.25 sq km per cell
    cell_area_km2 = (config.STUDY_AREA["approx_cell_size_m"] / 1000.0) ** 2
    high_risk_cells = np.sum((risk_grid >= 60.0) & (perm_water_grid == 0))
    high_risk_area_km2 = round(high_risk_cells * cell_area_km2, 1)

    # Calculate mean risk across flood-prone alluvial corridor (elevation < 75m, non-permanent river)
    elev_grid = df_grid["elevation"].values.reshape((rows, cols))
    floodplain_mask = (elev_grid < 75.0) & (perm_water_grid == 0)
    
    if np.sum(floodplain_mask) > 0:
        mean_risk = round(float(np.mean(risk_grid[floodplain_mask])), 1)
    else:
        mean_risk = round(float(np.mean(risk_grid[perm_water_grid == 0])), 1)
        
    domain_mean_risk = round(float(np.mean(risk_grid)), 1)
    max_risk = round(float(np.max(risk_scores)), 1)

    # Categories count
    categories = [assign_risk_category(r) for r in risk_scores]
    cat_counts = pd.Series(categories).value_counts().to_dict()

    return {
        "risk_grid": risk_grid,
        "prob_grid": prob_grid,
        "gt_grid": gt_grid,
        "perm_water_grid": perm_water_grid,
        "mean_risk": mean_risk,
        "domain_mean_risk": domain_mean_risk,
        "max_risk": max_risk,
        "high_risk_area_km2": high_risk_area_km2,
        "category_counts": cat_counts,
        "risk_scores": risk_scores
    }
