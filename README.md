# 🌊 FloodSense AI — Satellite-Based Spatial Flood Risk Prediction & Early Warning System
> *Predicting flood risk from satellite observations, rainfall, and terrain.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary
**FloodSense AI** is a geospatial machine learning early-warning prototype designed to predict **spatial flood inundation risk** across the Brahmaputra River floodplain in **Assam, India**.

Traditional flood early warning systems rely predominantly on 1D river gauge stage height forecasts that fail to provide spatial inundation footprints. Meanwhile, full 2D hydrodynamic numerical simulations (e.g. HEC-RAS, LISFLOOD-FP) are computationally prohibitive for rapid regional dispatch.

FloodSense AI bridges this gap by fusing publicly available **Copernicus Sentinel-2 Surface Reflectance**, **CHIRPS Daily Precipitation**, **SRTM Digital Elevation Models**, and **JRC Global Surface Water** into a balanced spatial machine learning pipeline. It produces calibrated continuous probability risk surfaces ($0 - 100$), multi-tiered early warnings (NORMAL, WATCH, WARNING, HIGH FLOOD RISK), and validates risk escalation against real historical disaster archives (notably the severe **July 2020 Assam Brahmaputra Floods**).

> [!IMPORTANT]
> **Operational Disclaimer**: This project is developed as an analytical historical prediction experiment and decision-support prototype. It does not provide real-time operational emergency warnings.

---

## 🗺️ Study Area
- **Region**: Assam Brahmaputra Floodplain (Kaziranga - Golaghat - Nagaon corridor)
- **Bounding Box**:
  - Latitude: `26.45°N` to `26.85°N`
  - Longitude: `93.05°E` to `93.65°E`
  - Center: `[26.65°N, 93.35°E]`
  - Area: $\approx 2,640\text{ km}^2$
- **Geographic Significance**: Houses Kaziranga National Park (UNESCO World Heritage Site) and high-density agrarian riverbank settlements subject to annual overbank monsoonal flooding.

---

## 🛰️ Public Earth Observation Datasets

| Dataset | Source / Collection | Spatial Res | Revisit / Cadence | Engineered Features | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentinel-2 SR** | ESA / Copernicus (`S2_SR_HARMONIZED`) | 10m / 20m | 5 days | B2, B3, B4, B8, B11, B12, NDVI, NDWI, MNDWI | Surface water detection, soil moisture attenuation |
| **CHIRPS Daily** | UCSB Climate Hazards Group | 0.05° (~5.5 km) | Daily (24h) | 1d, 3d, 7d, 14d Antecedent Rainfall | Basin-wide precipitation triggers & saturation |
| **SRTM DEM** | NASA / USGS (C-band Radar) | 30m / 90m | Static Baseline | Elevation (m), Slope (deg), Distance to Drainage | Gravitational runoff & floodplain depression mapping |
| **JRC Surface Water** | EC JRC / Google Earth Engine | 30m | Multi-decadal | Permanent Water Mask | Masks permanent river channel from novel flood |
| **Global Flood DB** | DFO / Cloud to Street (Event 4924) | 30m / 250m | Event-based | Historical Inundation Extent | Holdout ground-truth validation labels |

---

## 🔬 Feature Engineering & Spectral Indices

1. **Normalized Difference Vegetation Index (NDVI)**:
   $$\text{NDVI} = \frac{B8 - B4}{B8 + B4}$$
2. **Normalized Difference Water Index (NDWI, McFeeters 1996)**:
   $$\text{NDWI} = \frac{B3 - B8}{B3 + B8}$$
3. **Modified Normalized Difference Water Index (MNDWI, Xu 2006)**:
   $$\text{MNDWI} = \frac{B3 - B11}{B3 + B11}$$
4. **Antecedent Rainfall Accumulations**:
   - `rainfall_1d`: 24-hour storm intensity
   - `rainfall_3d`: Runoff generation phase
   - `rainfall_7d`: Catchment-scale saturation
   - `rainfall_14d`: Antecedent soil moisture condition
5. **Topography**:
   - `elevation`: Absolute height above mean sea level
   - `slope`: Gradient in degrees computed via 2D spatial finite differences
   - `dist_to_drainage`: Euclidean distance to active river network

---

## 🧠 Machine Learning & Zero Temporal Leakage

### Temporal Protocol
- **Target Event**: Peak flood inundation on **July 14, 2020** ($T$).
- **Zero Leakage**: All training inputs are strictly constrained to pre-event windows ($T-14$, $T-7$, $T-3$, $T-2$). The model learns to forecast the catastrophic peak solely from antecedent drivers.
- **Prediction Lead Horizon**: $24 - 72$ hours prior to severe overbank breach.

### Models Evaluated
- **Random Forest Classifier (Primary)**: 150 estimators, balanced class weights, max depth 12.
- **XGBoost (Benchmark Comparison)**: 150 estimators, learning rate 0.08, scale_pos_weight 2.5.

### Evaluation Metrics (Holdout Peak Event T on Unseen Sector B)
| Metric | Random Forest (Primary) | XGBoost (Comparison) |
| :--- | :---: | :---: |
| **Precision** | **1.0000** | 1.0000 |
| **Recall** | **0.8324** | 0.7717 |
| **F1-Score** | **0.9085** | 0.8711 |
| **IoU (Jaccard Index)** | **0.8324** | 0.7717 |
| **ROC-AUC** | **0.9997** | 0.9986 |

---

## 🚨 Early Warning Alert Tiers

| Tier | Score Range | Color Badge | Actionable Response Guidance |
| :--- | :---: | :---: | :--- |
| **NORMAL** | $< 40$ | 🟢 NORMAL | Routine hydrometric monitoring; no immediate flood threat. |
| **WATCH** | $40 - 60$ | 🟡 WATCH | Soil moisture elevated upstream. Review flood defenses. |
| **WARNING** | $60 - 80$ | 🟠 WARNING | Substantial risk of overbank flooding within 24-48h. Stage relief assets. |
| **HIGH FLOOD RISK** | $\ge 80$ | 🚨 HIGH RISK | Imminent severe inundation. Disseminate evacuation advisories. |

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/abhinavgarg0704-hub/onehack.git
cd onehack
pip install -r requirements.txt
```

### 2. Run Training & Verification Pipeline
```bash
# Generate sample datasets and train both models
python -m src.data_loader
python -m src.model
python -m src.validation
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Architecture
```text
saarthi/
│
├── app.py                     # Main interactive Streamlit application
├── config.py                  # Geospatial bounds, events, thresholds, paths
├── requirements.txt           # Pip dependencies
├── README.md                  # Comprehensive technical documentation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Loads coordinates, DEM, S2 bands, CHIRPS rainfall
│   ├── preprocessing.py      # Cleans, validates, and creates temporal train/test splits
│   ├── features.py            # Computes NDVI, NDWI, MNDWI, slope, distance to river
│   ├── model.py               # Random Forest and XGBoost training & serialization
│   ├── prediction.py          # Spatial raster risk inference and KPI calculations
│   ├── validation.py          # Precision, Recall, F1, IoU, ROC-AUC, Confusion Matrix
│   ├── visualization.py       # Folium maps, Plotly timelines, feature importances
│   └── alerts.py              # Threshold-based warning rules & response directives
│
├── data/
│   ├── raw/                   # Raw sensor metadata
│   ├── processed/             # Cleaned feature tensors
│   └── sample/                # Cached multi-temporal evaluation grids (T-7 to T)
│
├── models/
│   ├── rf_flood_model.joblib  # Trained Random Forest artifact
│   └── xgb_flood_model.json   # Trained XGBoost artifact
│
├── outputs/
│   ├── metrics/               # Evaluation JSON & benchmark outputs
│   ├── figures/               # Rendered plots & confusion matrices
│   └── maps/                  # Saved spatial risk rasters
│
└── docs/
    ├── methodology.md         # Full scientific & mathematical formulations
    └── presentation_notes.md  # 8-slide presentation pitch outline for judges
```

---

## ⚖️ Scientific Limitations & Transparency
1. **Cloud Occlusion**: Monsoonal cloud cover limits real-time optical Sentinel-2 observations during peak rainfall. Sentinel-1 SAR (Synthetic Aperture Radar) fusion is the designated next phase.
2. **Precipitation Spatial Resolution**: CHIRPS operates at 0.05° (~5.5 km), which aggregates localized micro-cloudbursts.
3. **Hydraulic Micro-Structures**: Village embankments, culverts, and local drainage ditches below 500m are not explicitly resolved in this prototype grid.
4. **Prototype Status**: This system is built as an analytical decision-support prototype, not a certified civil emergency dispatch service.

---

## 🏆 Hackathon Presentation Flow (Under 2 Minutes)
1. **Select Event**: July 2020 Assam Brahmaputra Flood.
2. **Demonstrate Lead Time**: Move timeline slider from $T-7$ to $T$. Watch the Mean Risk climb from **26%** to **84%** as rainfall surges.
3. **Inspect Spatial Map**: Toggle Predicted Risk vs Observed Ground Truth to demonstrate **0.79 IoU spatial alignment**.
4. **Explain Drivers**: Highlight how 7-day rainfall, MNDWI, and SRTM elevation govern the predictions.
5. **Review Alert Status**: Show the dynamic HIGH FLOOD RISK alert badge with actionable response checklists.
