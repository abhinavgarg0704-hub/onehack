"""
FloodSense AI - Configuration Module
Defines study region bounding box, historical flood events, feature schemas,
sensor parameters, warning thresholds, and filesystem paths.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MAPS_DIR = OUTPUTS_DIR / "maps"
DOCS_DIR = BASE_DIR / "docs"

for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR, MAPS_DIR, DOCS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Study Area: Assam Brahmaputra River Floodplain (Kaziranga - Golaghat Corridor)
STUDY_AREA = {
    "name": "Assam Brahmaputra Floodplain (Kaziranga - Golaghat Corridor)",
    "region_code": "IN-AS-KG",
    "country": "India",
    "state": "Assam",
    "min_lat": 26.45,
    "max_lat": 26.85,
    "min_lon": 93.05,
    "max_lon": 93.65,
    "center_lat": 26.65,
    "center_lon": 93.35,
    "default_zoom": 11,
    "grid_rows": 50,
    "grid_cols": 75,
    "approx_cell_size_m": 500,  # ~500m resolution per cell for high speed and precision
    "total_area_sq_km": 2640.0,
}

# Historical Events
HISTORICAL_EVENTS = {
    "event_2020_07": {
        "id": "assam_july_2020",
        "name": "July 2020 Assam Brahmaputra Catastrophic Flood",
        "description": "Massive monsoon inundation in the Brahmaputra valley submerging over 85% of Kaziranga and surrounding villages.",
        "peak_date": "2020-07-14",
        "timeline": [
            {"tag": "T-7", "date": "2020-07-07", "desc": "Pre-surge baseline, early monsoon showers"},
            {"tag": "T-5", "date": "2020-07-09", "desc": "Heavy rainfall intensification upstream in catchment"},
            {"tag": "T-3", "date": "2020-07-11", "desc": "Continuous torrential precipitation, river approaching danger mark"},
            {"tag": "T-2", "date": "2020-07-12", "desc": "Overtopping embankments, low-lying drainage saturated"},
            {"tag": "T-1", "date": "2020-07-13", "desc": "Severe embankment breaches, floodwaters rushing into plains"},
            {"tag": "T",   "date": "2020-07-14", "desc": "Peak flood inundation across Kaziranga and Brahmaputra floodplain"},
        ],
        "rainfall_timeline_7d": [58.4, 94.2, 168.5, 224.0, 286.3, 342.8],
        "rainfall_timeline_1d": [12.2, 28.5, 54.0, 68.4, 76.2, 92.5],
        "ground_truth_source": "Global Flood Database (DFO Event 4924) and Copernicus EMS Verified Flood Masks",
        "documented_flooded_area_sq_km": 1120.0,
    },
    "event_2022_06": {
        "id": "assam_june_2022",
        "name": "June 2022 Pre-Peak Brahmaputra Monsoon Flood",
        "description": "Extreme pre-monsoon precipitation triggering widespread inundation across Lower and Central Assam.",
        "peak_date": "2022-06-20",
        "timeline": [
            {"tag": "T-7", "date": "2022-06-13", "desc": "Initial monsoon surge in Upper Assam"},
            {"tag": "T-5", "date": "2022-06-15", "desc": "Heavy cloudburst in Arunachal foothills"},
            {"tag": "T-3", "date": "2022-06-17", "desc": "River levels rising past alert warning levels"},
            {"tag": "T-2", "date": "2022-06-18", "desc": "Secondary drainage canals backed up"},
            {"tag": "T-1", "date": "2022-06-19", "desc": "Substantial inundation of agricultural paddies"},
            {"tag": "T",   "date": "2022-06-20", "desc": "Regional flood crest across lower floodplain"},
        ],
        "rainfall_timeline_7d": [44.0, 78.5, 142.0, 198.2, 252.0, 310.5],
        "rainfall_timeline_1d": [8.5, 22.0, 48.2, 59.0, 64.5, 81.0],
        "ground_truth_source": "Global Flood Database / Sentinel-1 SAR Rapid Mapping",
        "documented_flooded_area_sq_km": 940.0,
    }
}

# Features List for ML Model
FEATURE_COLUMNS = [
    "ndvi",
    "ndwi",
    "mndwi",
    "b3",
    "b4",
    "b8",
    "b11",
    "b12",
    "rainfall_1d",
    "rainfall_3d",
    "rainfall_7d",
    "rainfall_14d",
    "elevation",
    "slope",
    "dist_to_drainage",
    "permanent_water"
]

TARGET_COLUMN = "is_flooded"

# Risk Categories
RISK_CATEGORIES = {
    "LOW": {"min": 0, "max": 25, "color": "#2ECC71", "label": "Low Risk"},
    "MODERATE": {"min": 25, "max": 50, "color": "#F39C12", "label": "Moderate Risk"},
    "HIGH": {"min": 50, "max": 75, "color": "#E67E22", "label": "High Risk"},
    "VERY HIGH": {"min": 75, "max": 100, "color": "#E74C3C", "label": "Very High Risk"}
}

# Alert Thresholds (0 - 100 risk score)
ALERT_LEVELS = {
    "NORMAL": {
        "threshold": 0,
        "label": "NORMAL",
        "color": "#27AE60",
        "badge": "NORMAL",
        "icon": "🟢",
        "action": "Routine hydrometric observation; no immediate flood hazard detected."
    },
    "WATCH": {
        "threshold": 40,
        "label": "WATCH",
        "color": "#F1C40F",
        "badge": "WATCH",
        "icon": "🟡",
        "action": "Elevated soil saturation and upstream precipitation. Monitor river stages closely."
    },
    "WARNING": {
        "threshold": 60,
        "label": "WARNING",
        "color": "#E67E22",
        "badge": "WARNING",
        "icon": "🟠",
        "action": "Significant likelihood of overbank flooding within 24 to 48 hours. Stage flood mitigation assets."
    },
    "HIGH_RISK": {
        "threshold": 80,
        "label": "HIGH FLOOD RISK",
        "color": "#E74C3C",
        "badge": "HIGH FLOOD RISK",
        "icon": "🚨",
        "action": "Imminent high-magnitude inundation predicted across low-lying floodplain. Evacuation advisory."
    }
}

# Prediction Horizon Disclosure
PREDICTION_HORIZON = "24 to 72 Hours (Antecedent precipitation and pre-event optical synthesis)"

# Model Artifact Paths
RF_MODEL_PATH = MODELS_DIR / "rf_flood_model.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgb_flood_model.json"
METRICS_JSON_PATH = METRICS_DIR / "evaluation_metrics.json"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"
