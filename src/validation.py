"""
Model Validation & Quantitative Benchmark Suite
Computes Precision, Recall, F1-Score, IoU (Jaccard Index), ROC-AUC, and Confusion Matrix.
Compares Random Forest against XGBoost on the historical flood holdout.
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, roc_auc_score, confusion_matrix
import config
from src.model import FloodRiskModel
from src.preprocessing import get_train_test_split

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes standard and geospatial evaluation metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "iou": round(float(iou), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "threshold": threshold
    }

def run_comparative_evaluation():
    """
    Runs evaluation for both RF and XGBoost on holdout peak event T.
    Saves results to metrics/evaluation_metrics.json.
    """
    X_train, y_train, X_test, y_test, df_test = get_train_test_split()

    rf = FloodRiskModel("rf")
    rf.load()
    rf_probs = rf.predict_proba(X_test)
    rf_metrics = evaluate_predictions(y_test.values, rf_probs)

    xgb_model = FloodRiskModel("xgb")
    xgb_model.load()
    xgb_probs = xgb_model.predict_proba(X_test)
    xgb_metrics = evaluate_predictions(y_test.values, xgb_probs)

    comparison = {
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
        "evaluation_event": config.HISTORICAL_EVENTS["event_2020_07"]["name"],
        "target_date": config.HISTORICAL_EVENTS["event_2020_07"]["peak_date"],
        "total_test_samples": len(y_test),
        "flood_pixels_count": int(np.sum(y_test == 1)),
        "non_flood_pixels_count": int(np.sum(y_test == 0)),
    }

    with open(config.METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n--- Model Benchmark on July 2020 Assam Brahmaputra Flood (Peak Event T) ---")
    print(f"{'Model':<15} {'F1-Score':<10} {'IoU':<10} {'Recall':<10} {'Precision':<12} {'ROC-AUC':<10}")
    for name, m in [("Random Forest", rf_metrics), ("XGBoost", xgb_metrics)]:
        print(f"{name:<15} {m['f1']:<10.4f} {m['iou']:<10.4f} {m['recall']:<10.4f} {m['precision']:<12.4f} {m['roc_auc']:<10.4f}")

    return comparison

if __name__ == "__main__":
    run_comparative_evaluation()
