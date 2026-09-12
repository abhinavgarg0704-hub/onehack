"""
Preprocessing and Dataset Preparation Pipeline
Ensures zero temporal data leakage, validates feature schema,
and produces clean train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config

def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing values, infinite values, and clamps variables to physical ranges.
    """
    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    
    # Fill any NaNs with median values
    for col in config.FEATURE_COLUMNS:
        if col in cleaned.columns and cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            
    # Physical reflectance bounds [0.0, 1.0]
    for b in ["b3", "b4", "b8", "b11", "b12"]:
        if b in cleaned.columns:
            cleaned[b] = cleaned[b].clip(0.0, 1.0)
            
    # Index bounds [-1.0, 1.0]
    for idx in ["ndvi", "ndwi", "mndwi"]:
        if idx in cleaned.columns:
            cleaned[idx] = cleaned[idx].clip(-1.0, 1.0)
            
    return cleaned

def get_spatial_temporal_split(seed: int = 42):
    """
    Constructs train and test datasets strictly avoiding both temporal and spatial leakage:
    - Spatial Split:
        * Training Region [Area A]: West and Central Kaziranga/Nagaon alluvial plain (cols <= 52)
        * Holdout Test Region [Area B]: East Kaziranga/Bokakhat alluvial plain (cols > 52)
    - Temporal Split:
        * Training Dates: Antecedent conditions (T-7, T-5, T-3, T-2)
        * Test Date: Peak historical flood holdout (T, July 14, 2020)
    
    This evaluates whether the model trained on antecedent conditions in one geographic sector
    can accurately predict catastrophic flood inundation in an unseen geographic sector at peak flood.
    """
    from src.data_loader import load_data_for_tag

    train_dfs = []
    for tag in ["T-7", "T-5", "T-3", "T-2"]:
        train_dfs.append(load_data_for_tag(tag))
    df_train_raw = pd.concat(train_dfs, ignore_index=True)
    df_test_raw = load_data_for_tag("T")

    df_train = validate_features(df_train_raw)
    df_test = validate_features(df_test_raw)

    train_mask = df_train["col"] <= 52
    test_mask = df_test["col"] > 52

    X_train = df_train.loc[train_mask, config.FEATURE_COLUMNS]
    y_train = df_train.loc[train_mask, config.TARGET_COLUMN]

    X_test = df_test.loc[test_mask, config.FEATURE_COLUMNS]
    y_test = df_test.loc[test_mask, config.TARGET_COLUMN]

    return X_train, y_train, X_test, y_test, df_test.loc[test_mask]

def get_train_test_split(seed: int = 42):
    """Default split uses the rigorous Spatial-Temporal Block Split."""
    return get_spatial_temporal_split(seed=seed)

