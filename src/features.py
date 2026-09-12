"""
Feature Engineering Module for FloodSense AI
Computes spectral indices (NDVI, NDWI, MNDWI), topographic slope,
distance to drainage channels, and antecedent rainfall aggregations.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

def compute_ndvi(b8: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index (NDVI) = (NIR - Red) / (NIR + Red)
    For Sentinel-2: (B8 - B4) / (B8 + B4)
    """
    denom = b8 + b4 + 1e-7
    return np.clip((b8 - b4) / denom, -1.0, 1.0)

def compute_ndwi(b3: np.ndarray, b8: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index (NDWI, McFeeters 1996) = (Green - NIR) / (Green + NIR)
    For Sentinel-2: (B3 - B8) / (B3 + B8)
    Positive values (>0) typically correspond to open water bodies.
    """
    denom = b3 + b8 + 1e-7
    return np.clip((b3 - b8) / denom, -1.0, 1.0)

def compute_mndwi(b3: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """
    Modified Normalized Difference Water Index (MNDWI, Xu 2006) = (Green - SWIR1) / (Green + SWIR1)
    For Sentinel-2: (B3 - B11) / (B3 + B11)
    Suppresses noise from bare soil, built-up surfaces, and suspended river sediment.
    """
    denom = b3 + b11 + 1e-7
    return np.clip((b3 - b11) / denom, -1.0, 1.0)

def compute_slope(elevation_grid: np.ndarray, cell_size_m: float = 500.0) -> np.ndarray:
    """
    Calculates topographic gradient / slope in degrees from a 2D DEM grid.
    Uses central differences for interior pixels and forward/backward differences for edges.
    """
    dy, dx = np.gradient(elevation_grid, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    return np.clip(slope_deg, 0.0, 90.0)

def compute_dist_to_drainage(water_mask: np.ndarray, cell_size_m: float = 500.0) -> np.ndarray:
    """
    Computes Euclidean distance to the nearest permanent water channel in meters.
    water_mask: 2D array where 1 = permanent water, 0 = land.
    """
    # Inverse distance transform calculates distance to zero-pixels
    # Invert so 0 = water, 1 = land
    inv_mask = 1 - (water_mask > 0).astype(np.int32)
    dist_m = distance_transform_edt(inv_mask) * cell_size_m
    return dist_m

def engineer_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures all derived features (NDVI, NDWI, MNDWI) exist in tabular DataFrame.
    """
    out = df.copy()
    if "ndvi" not in out.columns and "b8" in out.columns and "b4" in out.columns:
        out["ndvi"] = compute_ndvi(out["b8"].values, out["b4"].values)
    if "ndwi" not in out.columns and "b3" in out.columns and "b8" in out.columns:
        out["ndwi"] = compute_ndwi(out["b3"].values, out["b8"].values)
    if "mndwi" not in out.columns and "b3" in out.columns and "b11" in out.columns:
        out["mndwi"] = compute_mndwi(out["b3"].values, out["b11"].values)
    return out
