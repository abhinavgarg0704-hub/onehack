"""
Machine Learning Model Training & Comparative Architecture
Trains primary Random Forest Classifier and comparative XGBoost Classifier.
Calibrates probabilities and computes feature importances.
"""

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import config
from src.preprocessing import get_train_test_split

class FloodRiskModel:
    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "xgb":
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=2.5,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(Flood) in [0.0, 1.0]"""
        return self.model.predict_proba(X)[:, 1]

    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """Converts probability to 0-100 risk score"""
        proba = self.predict_proba(X)
        return np.round(proba * 100.0, 1)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model_type == "rf":
            importances = self.model.feature_importances_
        else:
            importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            "feature": config.FEATURE_COLUMNS,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp

    def save(self, path=None):
        if path is None:
            path = config.RF_MODEL_PATH if self.model_type == "rf" else config.XGB_MODEL_PATH
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        if path is None:
            path = config.RF_MODEL_PATH if self.model_type == "rf" else config.XGB_MODEL_PATH
        self.model = joblib.load(path)

def train_and_save_all_models():
    """
    Executes training of both Random Forest and XGBoost models and persists them.
    """
    X_train, y_train, X_test, y_test, df_test = get_train_test_split()
    print(f"Training dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features.")
    print(f"Holdout validation dataset (Peak Event T): {X_test.shape[0]} samples.")

    # 1. Random Forest (Primary)
    print("Training Random Forest Classifier...")
    rf = FloodRiskModel(model_type="rf")
    rf.train(X_train, y_train)
    rf.save(config.RF_MODEL_PATH)

    # 2. XGBoost (Comparison)
    print("Training XGBoost Classifier...")
    xgb_model = FloodRiskModel(model_type="xgb")
    xgb_model.train(X_train, y_train)
    xgb_model.save(config.XGB_MODEL_PATH)

    print("All models successfully trained and persisted.")

if __name__ == "__main__":
    train_and_save_all_models()
