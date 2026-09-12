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

def get_train_test_split(seed: int = 42):
    """
    Constructs train and test datasets strictly avoiding temporal leakage:
    - Train data: Antecedent conditions and baseline events (T-7, T-5, T-3, T-2)
    - Test / Validation data: Peak historical event holdout (T, July 14, 2020)
    This strictly evaluates whether the model can predict the severe peak flood
    using environmental and antecedent drivers learned before the peak.
    """
    from src.data_loader import load_data_for_tag

    train_dfs = []
    for tag in ["T-7", "T-5", "T-3", "T-2"]:
        train_dfs.append(load_data_for_tag(tag))
    df_train_raw = pd.concat(train_dfs, ignore_index=True)
    df_test_raw = load_data_for_tag("T")

    df_train = validate_features(df_train_raw)
    df_test = validate_features(df_test_raw)

    X_train = df_train[config.FEATURE_COLUMNS]
    y_train = df_train[config.TARGET_COLUMN]

    X_test = df_test[config.FEATURE_COLUMNS]
    y_test = df_test[config.TARGET_COLUMN]

    return X_train, y_train, X_test, y_test, df_test
