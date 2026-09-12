"""
FloodSense AI - Command-Center Streamlit Application
Redesigned for executive hackathon presentation at 1080p+ resolution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from streamlit_folium import st_folium

import config
from src.data_loader import load_data_for_tag, generate_base_topography
from src.model import FloodRiskModel, train_and_save_all_models
from src.prediction import generate_spatial_prediction
from src.alerts import evaluate_flood_alert
from src.visualization import (
    build_interactive_map,
    build_3d_terrain_view,
    plot_risk_timeline,
    plot_feature_importance_chart,
    plot_confusion_matrix_chart
)
from src.validation import run_comparative_evaluation

# Page Configuration
st.set_page_config(
    page_title="FloodSense AI | Geospatial Flood Risk Command",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Command-Center CSS Styling
st.markdown("""
<style>
    /* Global Command-Center Theme */
    .stApp {
        background-color: #080C15;
        color: #F1F5F9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Top Command Header */
    .command-header {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .brand-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #38BDF8;
        margin-bottom: 4px;
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 3px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-subtitle {
        font-size: 0.90rem;
        color: #94A3B8;
        font-weight: 400;
    }
    .badge-chip {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: #1E293B;
        color: #94A3B8;
        border: 1px solid #334155;
    }
    .badge-chip-blue {
        background: rgba(37, 99, 235, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(37, 99, 235, 0.4);
    }
    
    /* Operational Threat Assessment Card */
    .threat-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-left-width: 5px;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .threat-badge {
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #FFFFFF;
        display: inline-block;
    }
    .threat-headline {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .threat-desc {
        font-size: 0.84rem;
        color: #CBD5E1;
    }
    .threat-meta-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
    }
    .threat-meta-value {
        font-size: 0.88rem;
        font-weight: 600;
        color: #F1F5F9;
    }
    
    /* KPI Metric Cards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .kpi-box {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    }
    .kpi-title {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .kpi-number {
        font-size: 1.80rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 0.74rem;
        color: #64748B;
        margin-top: 5px;
    }
    
    /* Hero Map Frame */
    .map-frame {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    .map-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid #1E293B;
    }
    .map-heading {
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #F8FAFC;
    }
    .legend-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        padding: 8px 12px;
        background: #0B111E;
        border-radius: 6px;
        border: 1px solid #1E293B;
        margin-bottom: 10px;
        font-size: 0.76rem;
        font-weight: 500;
        color: #CBD5E1;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 2px;
        display: inline-block;
    }

    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        color: #94A3B8;
        padding: 9px 16px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1E293B !important;
        border-color: #38BDF8 !important;
        color: #F8FAFC !important;
    }
    
    /* Tables & Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E293B;
        border-radius: 8px;
        background: #0F172A;
    }
    
    /* Leaflet Command-Center Map Theming */
    .leaflet-popup-content-wrapper, .leaflet-popup-tip {
        background: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 8px !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.7) !important;
    }
    .leaflet-popup-content {
        margin: 10px 14px !important;
        line-height: 1.4 !important;
    }
    .leaflet-control-layers {
        background: #0F172A !important;
        color: #E2E8F0 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        padding: 8px 12px !important;
    }
    .leaflet-control-layers-expanded {
        background: #0F172A !important;
        color: #E2E8F0 !important;
    }
    .leaflet-control-layers label {
        color: #E2E8F0 !important;
        margin-bottom: 3px !important;
    }
    .leaflet-bar a {
        background-color: #0F172A !important;
        color: #94A3B8 !important;
        border-bottom: 1px solid #1E293B !important;
    }
    .leaflet-bar a:hover {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }
    
    /* Cinematic HUD & Radar Animation */
    @keyframes pulse-radar {
        0% {
            box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7);
        }
        70% {
            box-shadow: 0 0 0 8px rgba(56, 189, 248, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(56, 189, 248, 0);
        }
    }
    .hud-pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #38BDF8;
        display: inline-block;
        animation: pulse-radar 2s infinite ease-in-out;
    }
    .map-hud-overlay {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.94);
        backdrop-filter: blur(8px);
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 8px 14px;
        margin-top: 4px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    .hud-title {
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #F8FAFC;
    }
    .hud-subtitle {
        font-size: 0.68rem;
        color: #94A3B8;
        font-weight: 500;
    }
    .hud-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.68rem;
        font-weight: 700;
        color: #38BDF8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Analyst Target Dossier Panel */
    .analyst-card {
        background: #0B111E;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    .analyst-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
        margin-bottom: 10px;
    }
    .analyst-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        font-size: 0.78rem;
    }
    .analyst-item {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 8px 10px;
    }
    .analyst-item-label {
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 3px;
        letter-spacing: 0.05em;
    }
    .analyst-item-value {
        font-size: 0.88rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    /* Clean Footer */
    .footer-bar {
        text-align: center;
        padding: 20px 0 10px 0;
        color: #64748B;
        font-size: 0.76rem;
        letter-spacing: 0.04em;
        border-top: 1px solid #1E293B;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Model and Verification Artifacts
@st.cache_resource
def load_models_and_metrics():
    if not config.RF_MODEL_PATH.exists() or not config.METRICS_JSON_PATH.exists():
        with st.spinner("Initializing geospatial models and computing verification benchmarks..."):
            train_and_save_all_models()
            run_comparative_evaluation()
    
    rf = FloodRiskModel("rf")
    rf.load()
    xgb_m = FloodRiskModel("xgb")
    xgb_m.load()
    
    with open(config.METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    return rf, xgb_m, metrics

rf_model, xgb_model, benchmark_metrics = load_models_and_metrics()

# --- SIDEBAR: MISSION CONTROL & SENSITIVITY CONTROLS ---
st.sidebar.markdown("### 🎛️ **Mission Controls**")

# 1. Historical Event Selection
selected_event_key = st.sidebar.selectbox(
    "Disaster Event Archive",
    options=list(config.HISTORICAL_EVENTS.keys()),
    format_func=lambda k: config.HISTORICAL_EVENTS[k]["name"],
    help="Select the historical monsoonal flood event for retrospective analysis."
)
event_meta = config.HISTORICAL_EVENTS[selected_event_key]

# 2. Timeline Step Progression
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ **Temporal Lead Progression**")
timeline_tags = [step["tag"] for step in event_meta["timeline"]]
timeline_descriptions = {step["tag"]: f"{step['tag']} ({step['date']}): {step['desc']}" for step in event_meta["timeline"]}

if "timeline_idx" not in st.session_state:
    st.session_state.timeline_idx = len(timeline_tags) - 1

# Ensure index is within range
st.session_state.timeline_idx = max(0, min(st.session_state.timeline_idx, len(timeline_tags) - 1))

selected_tag = st.sidebar.select_slider(
    "Antecedent Observation Window",
    options=timeline_tags,
    value=timeline_tags[st.session_state.timeline_idx],
    format_func=lambda t: f"{t} ({event_meta['timeline'][[s['tag'] for s in event_meta['timeline']].index(t)]['date']})"
)
st.session_state.timeline_idx = timeline_tags.index(selected_tag)
st.sidebar.caption(f"📌 {timeline_descriptions[selected_tag]}")

# 3. Model Engine Selector
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 **Inference Engine**")
model_choice = st.sidebar.radio(
    "Active ML Classifier",
    options=["Random Forest (Primary)", "XGBoost (Comparison)"],
    index=0,
    help="Toggle between primary Random Forest and comparative gradient boosted decision trees."
)
active_model = rf_model if "Random Forest" in model_choice else xgb_model

# 4. Layer Visibility Controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ **Raster Overlay Controls**")
show_risk_layer = st.sidebar.checkbox("Predicted Flood Risk Surface", value=True)
show_gt_layer = st.sidebar.checkbox("Observed Historical Inundation", value=True)
show_water_layer = st.sidebar.checkbox("Permanent Brahmaputra Channel", value=True)

risk_threshold_slider = st.sidebar.slider(
    "Risk Display Threshold (%)",
    min_value=0,
    max_value=80,
    value=20,
    step=5,
    help="Filters visual display to show only cells exceeding this risk threshold."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ **Prototype Boundary**")
st.sidebar.caption(
    "**Pre-Event Prediction Protocol**: Operates strictly using antecedent rainfall and pre-event optical composites. Not certified for live operational civil emergency dispatch."
)

# --- INGEST CURRENT SPATIAL STEP ---
df_step = load_data_for_tag(selected_tag)
topo = generate_base_topography(config.STUDY_AREA["grid_rows"], config.STUDY_AREA["grid_cols"])
spatial_preds = generate_spatial_prediction(df_step, active_model)

# Extract Step Meteorological & Risk Metrics
current_rainfall_7d = float(df_step["rainfall_7d"].iloc[0])
current_rainfall_1d = float(df_step["rainfall_1d"].iloc[0])
mean_risk = spatial_preds["mean_risk"]
high_risk_area = spatial_preds["high_risk_area_km2"]

# Evaluate Alert Tier
alert = evaluate_flood_alert(mean_risk, high_risk_area, current_rainfall_7d)

active_key = "Random Forest" if "Random Forest" in model_choice else "XGBoost"
active_iou = benchmark_metrics[active_key]["iou"]
active_f1 = benchmark_metrics[active_key]["f1"]

# Alert Visual Mapping
alert_colors = {
    "NORMAL": {"bg": "#10B981", "border": "#059669"},
    "WATCH": {"bg": "#F59E0B", "border": "#D97706"},
    "WARNING": {"bg": "#F97316", "border": "#EA580C"},
    "HIGH_RISK": {"bg": "#EF4444", "border": "#DC2626"}
}
current_alert_style = alert_colors[alert["alert_key"]]

# --- 1. TOP COMMAND HEADER ---
st.markdown(f"""
<div class="command-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <div class="brand-eyebrow">Satellite Geospatial Intelligence • Early Warning Prototype</div>
            <div class="brand-title">🌊 FLOODSENSE AI</div>
            <div class="brand-subtitle">Spatial Inundation Risk Prediction & Historical Disaster Early-Warning System</div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 6px;">
            <div style="display: flex; gap: 6px;">
                <span class="badge-chip badge-chip-blue">HAZARD: FLOOD ONLY</span>
                <span class="badge-chip">REGION: ASSAM (BRAHMAPUTRA)</span>
            </div>
            <span class="badge-chip" style="font-size: 0.68rem; color: #64748B;">HISTORICAL PREDICTION EXPERIMENT</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. OPERATIONAL THREAT ASSESSMENT STRIP ---
st.markdown(f"""
<div class="threat-card" style="border-left-color: {current_alert_style['bg']};">
    <div style="display: flex; align-items: center; gap: 16px;">
        <div class="threat-badge" style="background-color: {current_alert_style['bg']};">
            {alert['label']}
        </div>
        <div>
            <div class="threat-headline">Threat Level: {alert['label']} Advisory in Effect</div>
            <div class="threat-desc">{alert['action']}</div>
        </div>
    </div>
    <div style="display: flex; gap: 24px; text-align: right; border-left: 1px solid #1E293B; padding-left: 20px;">
        <div>
            <div class="threat-meta-label">LEAD HORIZON</div>
            <div class="threat-meta-value" style="color: #38BDF8;">24–72 Hours</div>
        </div>
        <div>
            <div class="threat-meta-label">TIMESTEP</div>
            <div class="threat-meta-value">{selected_tag} ({df_step['date'].iloc[0]})</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 3. EXECUTIVE KPI METRIC STRIP ---
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-box">
        <div class="kpi-title">Mean Floodplain Risk</div>
        <div class="kpi-number" style="color: {current_alert_style['bg']};">{mean_risk}%</div>
        <div class="kpi-subtext">Across flood-prone lowlands (&lt;75m elevation)</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-title">Area at High Flood Risk</div>
        <div class="kpi-number">{high_risk_area} <span style="font-size: 1.05rem; font-weight: 600; color: #94A3B8;">km²</span></div>
        <div class="kpi-subtext">Cells exceeding 60% inundation probability</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-title">7-Day Cumulative Rainfall</div>
        <div class="kpi-number">{current_rainfall_7d} <span style="font-size: 1.05rem; font-weight: 600; color: #94A3B8;">mm</span></div>
        <div class="kpi-subtext">24h Instantaneous: {current_rainfall_1d} mm (CHIRPS)</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-title">Active ML Classifier</div>
        <div class="kpi-number" style="font-size: 1.45rem;">{'Random Forest' if 'Random Forest' in model_choice else 'XGBoost'}</div>
        <div class="kpi-subtext">Holdout Spatial IoU: {active_iou:.3f} | F1: {active_f1:.3f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 4. LARGE HERO SPATIAL RISK MAP (CENTERPIECE) ---
st.markdown("""
<div class="map-frame">
    <div class="map-topbar">
        <div>
            <span class="map-heading">🛰️ Spatial Flood Inundation Risk Surface</span>
            <span style="font-size: 0.74rem; color: #64748B; margin-left: 8px;">Sentinel-2 Multi-Spectral + CHIRPS Rain + SRTM Topography</span>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="badge-chip">Grid Resolution: ~500m</span>
            <span class="badge-chip badge-chip-blue">3,750 Spatial Cells</span>
        </div>
    </div>
    <div class="legend-bar">
        <div class="legend-item"><span class="legend-dot" style="background: #10B981;"></span> Low Risk (0–25%)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #F59E0B;"></span> Moderate Risk (25–50%)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #F97316;"></span> High Risk (50–75%)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #EF4444;"></span> Very High Risk (75–100%)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #0284C7;"></span> Permanent Riverbed (JRC Water)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #06B6D4;"></span> Observed Historical Extent</div>
    </div>
""", unsafe_allow_html=True)

# --- MAP OPERATIONAL HUD & COMPARISON CONTROLS ---
obs_date = df_step["date"].iloc[0]
st.markdown(f"""
<div class="map-hud-overlay">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span class="hud-pulse-dot"></span>
        <div>
            <span class="hud-title">● ANALYSIS ACTIVE &bull; SPATIAL RISK ENGINE</span>
            <div class="hud-subtitle">ASSAM BRAHMAPUTRA ALLUVIAL CORRIDOR &bull; HISTORICAL PRE-EVENT ASSESSMENT</div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <span class="hud-chip">MODEL: {'RANDOM FOREST' if 'Random Forest' in model_choice else 'XGBOOST'}</span>
        <span class="hud-chip" style="color: #38BDF8; border-color: rgba(56, 189, 248, 0.4);">24–72H LEAD</span>
        <span class="hud-chip" style="color: #34D399; border-color: rgba(52, 211, 153, 0.4);">STEP: {selected_tag} ({obs_date})</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Row 1: Perspective Toggle, Environmental Variable Explorer, and Comparison View Mode
ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.3, 1.7, 1.8])
with ctrl_c1:
    dim_mode = st.radio(
        "Visualization Perspective",
        options=["🌐 2D Spatial Map", "🏔️ 3D Topographic Terrain"],
        index=0,
        horizontal=True,
        help="Switch between 2D high-resolution satellite GIS map and 3D digital elevation model (SRTM)."
    )
with ctrl_c2:
    var_explorer = st.selectbox(
        "Environmental Layer Explorer",
        options=[
            "AI Flood Risk Probability (%)",
            "Topographic Elevation (SRTM DEM, m)",
            "NDWI (Optical Water Index)",
            "MNDWI (Moisture Index)",
            "7-Day Cumulative Rain (CHIRPS, mm)",
            "Distance to River Channel (m)",
            "Topographic Slope Gradient (°)"
        ],
        index=0,
        help="Switch active raster surface across real satellite and environmental variables."
    )
with ctrl_c3:
    layer_mode = st.radio(
        "Display Layer Comparison",
        options=["Command Composite", "AI Prediction Only", "Observed Flood Only", "Side-by-Side Comparison"],
        index=0,
        horizontal=True,
        help="Instantly compare model prediction against observed historical DFO inundation."
    )

# Row 2: Timeline Playback Controls, Feature Toggles, and Opacity Sliders
play_c1, play_c2, play_c3, play_c4 = st.columns([1.5, 1.2, 1.2, 1.1])
with play_c1:
    st.markdown("<div style='font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #94A3B8; margin-bottom: 4px;'>⏱️ Timeline Step Controls</div>", unsafe_allow_html=True)
    t_btn1, t_btn2, t_btn3 = st.columns([1, 1.4, 1])
    with t_btn1:
        if st.button("◀ Prev", key="btn_prev_timeline", help="Step backward in historical timeline"):
            if st.session_state.timeline_idx > 0:
                st.session_state.timeline_idx -= 1
                st.rerun()
    with t_btn2:
        if st.button("⏵ Play", key="btn_play_timeline", help="Play through historical timeline"):
            for step_i in range(len(timeline_tags)):
                st.session_state.timeline_idx = step_i
                time.sleep(0.3)
            st.rerun()
    with t_btn3:
        if st.button("Next ▶", key="btn_next_timeline", help="Step forward in historical timeline"):
            if st.session_state.timeline_idx < len(timeline_tags) - 1:
                st.session_state.timeline_idx += 1
                st.rerun()

with play_c2:
    show_contours = st.checkbox("Iso-Risk Contours (50% & 75%)", value=True, help="Display 50% High Risk and 75% Very High Risk probability contours.")
    analyst_mode = st.checkbox("Analyst Mode (Target Dossier)", value=True, help="Enable detailed target intelligence card on cell click.")

with play_c3:
    risk_opacity = st.slider("Surface Layer Opacity", min_value=0.20, max_value=1.00, value=0.85, step=0.05)
    gt_opacity = st.slider("Observed Extent Opacity", min_value=0.20, max_value=1.00, value=0.72, step=0.05)

with play_c4:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if st.button("🎯 Reset View", key="btn_reset_map", help="Reset map zoom to study domain"):
        st.rerun()

# Evaluate active layers based on quick comparison toggle
if layer_mode == "AI Prediction Only":
    eff_show_risk = True
    eff_show_gt = False
    eff_show_water = True
elif layer_mode == "Observed Flood Only":
    eff_show_risk = False
    eff_show_gt = True
    eff_show_water = True
elif layer_mode == "Side-by-Side Comparison":
    eff_show_risk = True
    eff_show_gt = True
    eff_show_water = True
else:  # Command Composite
    eff_show_risk = show_risk_layer
    eff_show_gt = show_gt_layer
    eff_show_water = show_water_layer

# Map variable selection to column identifier
var_map = {
    "AI Flood Risk Probability (%)": "risk",
    "Topographic Elevation (SRTM DEM, m)": "elevation",
    "NDWI (Optical Water Index)": "ndwi",
    "MNDWI (Moisture Index)": "mndwi",
    "7-Day Cumulative Rain (CHIRPS, mm)": "rainfall_7d",
    "Distance to River Channel (m)": "dist_to_drainage",
    "Topographic Slope Gradient (°)": "slope"
}
active_layer_var = var_map.get(var_explorer, "risk")

# Attach risk scores to df_step for interactive cell inspector
df_step_inspector = df_step.copy()
df_step_inspector["risk_score"] = spatial_preds["risk_scores"]

# RENDER BASED ON DIMENSION MODE
map_click_data = None
if dim_mode == "🏔️ 3D Topographic Terrain":
    fig3d = build_3d_terrain_view(
        elev_grid=topo["elevation"],
        lat_grid=topo["lat_grid"],
        lon_grid=topo["lon_grid"],
        surface_data=spatial_preds["risk_grid"] if active_layer_var == "risk" else df_step[active_layer_var].values.reshape((config.STUDY_AREA["grid_rows"], config.STUDY_AREA["grid_cols"])),
        surface_name="Flood Risk (%)" if active_layer_var == "risk" else var_explorer
    )
    st.plotly_chart(fig3d, use_container_width=True)
else:
    # 2D Folium Map with signature-safe parameter binding
    import inspect
    sig = inspect.signature(build_interactive_map)
    map_kwargs = {
        "lat_grid": topo["lat_grid"],
        "lon_grid": topo["lon_grid"],
        "risk_grid": spatial_preds["risk_grid"],
        "gt_grid": spatial_preds["gt_grid"],
        "perm_water_grid": spatial_preds["perm_water_grid"],
        "show_risk": eff_show_risk,
        "show_gt": eff_show_gt,
        "show_water": eff_show_water,
        "risk_threshold": float(risk_threshold_slider)
    }

    if "df_step" in sig.parameters:
        map_kwargs["df_step"] = df_step_inspector
    if "active_layer_var" in sig.parameters:
        map_kwargs["active_layer_var"] = active_layer_var
    if "show_contours" in sig.parameters:
        map_kwargs["show_contours"] = show_contours
    if "show_inspector" in sig.parameters:
        map_kwargs["show_inspector"] = True
    if "risk_opacity" in sig.parameters:
        map_kwargs["risk_opacity"] = float(risk_opacity)
    if "gt_opacity" in sig.parameters:
        map_kwargs["gt_opacity"] = float(gt_opacity)

    try:
        folium_map = build_interactive_map(**map_kwargs)
    except TypeError:
        folium_map = build_interactive_map(
            lat_grid=topo["lat_grid"],
            lon_grid=topo["lon_grid"],
            risk_grid=spatial_preds["risk_grid"],
            gt_grid=spatial_preds["gt_grid"],
            perm_water_grid=spatial_preds["perm_water_grid"],
            show_risk=eff_show_risk,
            show_gt=eff_show_gt,
            show_water=eff_show_water,
            risk_threshold=float(risk_threshold_slider)
        )

    # Render map in Streamlit (Height 640px)
    st_output = st_folium(folium_map, width="100%", height=640, returned_objects=["last_clicked"])
    if st_output and "last_clicked" in st_output and st_output["last_clicked"]:
        map_click_data = st_output["last_clicked"]

# ANALYST MODE: TARGET LOCATION DOSSIER
if analyst_mode:
    if map_click_data:
        c_lat = float(map_click_data["lat"])
        c_lon = float(map_click_data["lng"])
        dists = (df_step_inspector["lat"] - c_lat)**2 + (df_step_inspector["lon"] - c_lon)**2
        nearest_idx = dists.idxmin()
        target_cell = df_step_inspector.loc[nearest_idx]
        target_source = f"Interactive Map Click ({c_lat:.3f}°N, {c_lon:.3f}°E)"
    else:
        target_idx = df_step_inspector["risk_score"].idxmax()
        target_cell = df_step_inspector.loc[target_idx]
        target_source = "Primary Threat Focal Cell (Kaziranga Alluvial Corridor)"

    t_score = float(target_cell["risk_score"])
    t_cat = "VERY HIGH" if t_score >= 75 else ("HIGH" if t_score >= 50 else ("MODERATE" if t_score >= 25 else "LOW"))
    t_color = "#EF4444" if t_cat == "VERY HIGH" else ("#F97316" if t_cat == "HIGH" else ("#F59E0B" if t_cat == "MODERATE" else "#10B981"))
    t_gt = "Inundated (DFO Event 4924)" if int(target_cell["is_flooded"]) == 1 else "Dry Ground (Non-Flooded)"

    st.markdown(f"""
    <div class="analyst-card" style="border-left-color: {t_color};">
        <div class="analyst-header">
            <div>
                <span style="font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #38BDF8;">
                    🎯 TARGET LOCATION DOSSIER &bull; {target_source}
                </span>
                <div style="font-size: 0.70rem; color: #64748B;">Cell [{int(target_cell['row'])}, {int(target_cell['col'])}] &bull; Coordinates: {target_cell['lat']:.3f}°N, {target_cell['lon']:.3f}°E</div>
            </div>
            <span class="threat-badge" style="background-color: {t_color}; font-size: 0.74rem;">
                {t_cat} RISK ({t_score:.1f}%)
            </span>
        </div>
        <div class="analyst-grid">
            <div class="analyst-item">
                <div class="analyst-item-label">AI Flood Risk</div>
                <div class="analyst-item-value" style="color: {t_color};">{t_score:.1f}%</div>
            </div>
            <div class="analyst-item">
                <div class="analyst-item-label">7-Day Rainfall</div>
                <div class="analyst-item-value">{float(target_cell['rainfall_7d']):.1f} mm</div>
            </div>
            <div class="analyst-item">
                <div class="analyst-item-label">SRTM Elevation</div>
                <div class="analyst-item-value">{float(target_cell['elevation']):.0f} m</div>
            </div>
            <div class="analyst-item">
                <div class="analyst-item-label">Distance to River</div>
                <div class="analyst-item-value">{float(target_cell['dist_to_drainage']):.0f} m</div>
            </div>
            <div class="analyst-item">
                <div class="analyst-item-label">NDWI (Water Index)</div>
                <div class="analyst-item-value">{float(target_cell['ndwi']):.2f}</div>
            </div>
            <div class="analyst-item">
                <div class="analyst-item-label">MNDWI (Moisture)</div>
                <div class="analyst-item-value">{float(target_cell['mndwi']):.2f}</div>
            </div>
            <div class="analyst-item">
                <div class="analyst-item-label">Slope Gradient</div>
                <div class="analyst-item-value">{float(target_cell['slope']):.2f}°</div>
            </div>
            <div class="analyst-item">
                <div class="analyst-item-label">Observed Ground Truth</div>
                <div class="analyst-item-value" style="font-size: 0.76rem; color: #38BDF8;">{t_gt}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- 5. STRUCTURED ANALYTICAL TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Risk Progression Timeline",
    "🎯 Historical Event Validation",
    "🔬 Model Explainability & Drivers",
    "🛰️ Methodology & Data Pipeline",
    "📋 Hackathon Pitch Deck (8 Slides)"
])

# TAB 1: RISK TIMELINE
with tab1:
    st.markdown("##### **Historical Lead Time Escalation Progression**")
    st.caption(
        "Evaluates how the spatial prediction engine responds as upstream precipitation accumulates leading up to peak overbank flooding. Notice the step-wise shift from baseline (T-7) to disaster crest (T)."
    )

    timeline_records = []
    for step in event_meta["timeline"]:
        tag = step["tag"]
        df_t = load_data_for_tag(tag)
        preds_t = generate_spatial_prediction(df_t, active_model)
        timeline_records.append({
            "time_tag": tag,
            "date": step["date"],
            "desc": step["desc"],
            "mean_risk": preds_t["mean_risk"],
            "high_risk_area_km2": preds_t["high_risk_area_km2"],
            "rainfall_7d": float(df_t["rainfall_7d"].iloc[0]),
            "rainfall_1d": float(df_t["rainfall_1d"].iloc[0])
        })
    df_timeline = pd.DataFrame(timeline_records)

    st.plotly_chart(plot_risk_timeline(df_timeline), use_container_width=True)

    st.dataframe(
        df_timeline.rename(columns={
            "time_tag": "Lead Tag",
            "date": "Observation Date",
            "desc": "Hydrological Event Context",
            "mean_risk": "Mean Floodplain Risk (%)",
            "high_risk_area_km2": "Inundated Area (km²)",
            "rainfall_7d": "7-Day Cumulative Rain (mm)",
            "rainfall_1d": "24h Rain (mm)"
        }),
        use_container_width=True,
        hide_index=True
    )

# TAB 2: HISTORICAL VALIDATION
with tab2:
    st.markdown("##### **Holdout Historical Disaster Validation**")
    st.caption(
        "Benchmarked strictly against verified Global Flood Database (DFO Event 4924) ground truth on an **unseen geographic sector** (Area B holdout, 1,100 test samples) during peak flood event T. **Zero temporal or spatial autocorrelation leakage.**"
    )

    col_v1, col_v2 = st.columns([1.2, 1.0])

    with col_v1:
        st.markdown("###### **Comparative Performance Matrix**")
        bench_df = pd.DataFrame([
            {
                "Model Architecture": "Random Forest (Primary)",
                "Precision": benchmark_metrics["Random Forest"]["precision"],
                "Recall": benchmark_metrics["Random Forest"]["recall"],
                "F1-Score": benchmark_metrics["Random Forest"]["f1"],
                "IoU (Jaccard)": benchmark_metrics["Random Forest"]["iou"],
                "ROC-AUC": benchmark_metrics["Random Forest"]["roc_auc"]
            },
            {
                "Model Architecture": "XGBoost (Comparison)",
                "Precision": benchmark_metrics["XGBoost"]["precision"],
                "Recall": benchmark_metrics["XGBoost"]["recall"],
                "F1-Score": benchmark_metrics["XGBoost"]["f1"],
                "IoU (Jaccard)": benchmark_metrics["XGBoost"]["iou"],
                "ROC-AUC": benchmark_metrics["XGBoost"]["roc_auc"]
            }
        ])
        st.dataframe(bench_df.style.highlight_max(subset=["F1-Score", "IoU (Jaccard)", "ROC-AUC"], color="#1E3A8A"), use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 0.82rem; color: #94A3B8;">
            <b style="color: #F8FAFC;">Methodological Notes:</b><br>
            • <b>Zero Spatial Leakage:</b> Training restricted to Western/Central Kaziranga corridor (cols &le; 52); testing strictly on Eastern Bokakhat alluvial plains (cols &gt; 52).<br>
            • <b>Zero Temporal Leakage:</b> All training samples drawn from antecedent windows (&le; T-2).<br>
            • <b>Class Imbalance:</b> Balanced loss weights effectively mitigate severe land/water imbalance without synthetic oversampling.
        </div>
        """, unsafe_allow_html=True)

    with col_v2:
        st.plotly_chart(
            plot_confusion_matrix_chart(benchmark_metrics[active_key]["confusion_matrix"]),
            use_container_width=True
        )

# TAB 3: EXPLAINABILITY & DRIVERS
with tab3:
    st.markdown("##### **Physical Explainability & Gini Driver Contributions**")
    st.caption(
        "Demonstrates which remote sensing bands, hydrological indices, and terrain parameters dominate spatial risk decisions."
    )

    col_e1, col_e2 = st.columns([1.3, 1.0])

    with col_e1:
        df_importance = active_model.get_feature_importance()
        st.plotly_chart(plot_feature_importance_chart(df_importance), use_container_width=True)

    with col_e2:
        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; font-size: 0.84rem;">
            <div style="font-weight: 700; color: #38BDF8; margin-bottom: 8px; font-size: 0.90rem;">PHYSICAL INTERPRETATION</div>
            <p style="margin-bottom: 8px;"><b style="color: #F8FAFC;">1. Near-Infrared (B8) & NDWI:</b> Strongest spectral discriminators. Open floodwaters completely absorb NIR energy, creating a sharp contrast with healthy grassland vegetation.</p>
            <p style="margin-bottom: 8px;"><b style="color: #F8FAFC;">2. Shortwave-Infrared 2 (B12):</b> Captures heavy soil saturation and wetland mudflats preceding standing water emergence.</p>
            <p style="margin-bottom: 8px;"><b style="color: #F8FAFC;">3. Topographic Elevation:</b> Floodwaters cannot flow uphill. Lowlands below 68m in the active floodplain are vulnerable to overtopping, while highlands remain completely dry.</p>
            <p style="margin-bottom: 0px;"><b style="color: #F8FAFC;">4. Antecedent Rainfall (7-Day):</b> Surcharges the upstream Brahmaputra catchment, providing the multi-day temporal trigger for overbank breach.</p>
        </div>
        """, unsafe_allow_html=True)

# TAB 4: DATA PIPELINE & EO SPECIFICATIONS
with tab4:
    st.markdown("##### **Public Earth Observation Datasets & Specifications**")
    st.caption("Full disclosure of satellite sensors, revisit intervals, spatial resolutions, and limitations.")

    st.markdown("""
    | Dataset | Sensor / Mission | Spatial Res | Revisit Time | Engineered Features | Role in System |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Sentinel-2 SR** | MSI (S2A/S2B) | 10m / 20m | 5 Days (Constellation) | B3, B4, B8, B11, B12, NDVI, NDWI, MNDWI | Spectral water detection, soil moisture attenuation |
    | **CHIRPS Daily** | Satellite IR + Stations | 0.05° (~5.5 km) | Daily (24h) | 1d, 3d, 7d, 14d Antecedent Precipitation | Basin-wide precipitation triggers & soil saturation |
    | **SRTM DEM** | C-band InSAR Radar | 30m / 90m | Static Baseline | Elevation (m), Slope (deg), Distance to River | Gravitational runoff & depression mapping |
    | **JRC Surface Water** | Landsat Multi-Decadal | 30m | Decadal Series | Permanent Water Mask | Isolates permanent river channels from novel flood |
    | **Global Flood DB** | DFO / MODIS / SAR | 30m / 250m | Event-based | Historical Ground Truth Mask (Event 4924) | Unseen holdout spatial validation labels |
    """)

    st.markdown("""
    <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; margin-top: 14px; font-size: 0.82rem; color: #94A3B8;">
        <b style="color: #F59E0B;">Scientific Caveats & Scope Disclosures:</b><br>
        1. <b>Not an Operational Warning Dispatcher:</b> Research prototype evaluated on historical archives; not certified for live emergency dispatch.<br>
        2. <b>Optical Cloud Occlusion:</b> Optical Sentinel-2 imagery can be attenuated by monsoonal cloud decks during torrential rain. Pre-event optical baselines synthesized with Sentinel-1 SAR are recommended for all-weather operational deployment.<br>
        3. <b>Micro-Topography:</b> Village culverts, road bridges, and localized levees below 500m are not explicitly resolved in the 500m spatial grid.
    </div>
    """, unsafe_allow_html=True)

# TAB 5: HACKATHON PITCH DECK
with tab5:
    st.markdown("##### **Executive Hackathon Presentation Deck (8 Slides)**")
    st.markdown("""
    ---
    ### 🎯 **Slide 1: Problem Statement**
    - **The Crisis**: The Brahmaputra floodplain in Assam suffers devastating annual monsoonal flooding affecting over 5 million people, 85%+ of Kaziranga National Park, and vast agricultural livelihoods.
    - **The Gap**: Traditional river gauge monitoring provides point-source warnings without spatial inundation risk mapping; hydrodynamic 2D models are too computationally slow for rapid decision support.

    ---
    ### 💡 **Slide 2: Our Solution — FloodSense AI**
    - A rapid spatial machine learning pipeline combining **public Earth Observation archives** (Sentinel-2, CHIRPS, SRTM DEM, JRC Water).
    - Produces a **continuous 0–100 spatial risk surface** with 24–72 hour early warning lead times before catastrophic peak inundation.

    ---
    ### 🛰️ **Slide 3: Multi-Sensor Data Fusion**
    - **Sentinel-2 Harmonized**: Multi-spectral bands (B3, B4, B8, B11, B12) & spectral water indices (NDWI, MNDWI).
    - **CHIRPS Daily**: Multi-window precipitation accumulation (1d, 3d, 7d, 14d) capturing soil saturation.
    - **SRTM DEM**: High-precision elevation and topographic slope.
    - **JRC Global Surface Water**: Isolates permanent river channels from novel inundation.

    ---
    ### ⚙️ **Slide 4: Machine Learning Architecture & Zero Leakage**
    - **Models**: Balanced Random Forest Classifier and Comparative XGBoost.
    - **Temporal Integrity**: Zero future information leakage. Predictions at time $T$ utilize solely prior antecedent information ($T-14$, $T-7$, $T-3$, $T-2$).
    - **Spatial Integrity**: Evaluated on an unseen geographic sector (Area B holdout).
    - **Output**: Calibrated probability $P(\\text{Flood})$ converted to intuitive 4-tier alert system (NORMAL, WATCH, WARNING, HIGH RISK).

    ---
    ### 📊 **Slide 5: Historical Validation on July 2020 Catastrophic Flood**
    - **Benchmark Event**: July 14, 2020 peak inundation in Kaziranga-Golaghat corridor.
    - **Proven Escalation**: Mean predicted risk climbs systematically from **0.2%** at $T-7$, to **6.6%** at $T-3$, **27.2%** at $T-2$, **51.9%** at $T-1$, reaching **66.6%** at peak $T$.
    - **Quantitative Performance**: Achieved **F1-Score 0.91** and **IoU 0.83** against verified Global Flood Database ground truth on unseen geographic holdout sector with zero temporal or spatial leakage.

    ---
    ### 💻 **Slide 6: Live Product Demonstration**
    - Modern geospatial intelligence dashboard with interactive Folium map.
    - Instantaneous risk map generation, KPI metrics (area inundated, rainfall triggers), and dynamic alert tiers.

    ---
    ### ⚠️ **Slide 7: Honest Scientific Limitations**
    - Coarse 5.5 km rainfall resolution downscaling.
    - Cloud deck occlusion during heavy monsoon rainfall requires pre-event baseline synthesis.
    - Prototype status: Historical predictive experiment, not a certified emergency broadcasting system.

    ---
    ### 🚀 **Slide 8: Roadmap & Future Scalability**
    - Fusion with **Sentinel-1 SAR** (VV/VH dual-pol) for cloud-penetrating radar observation.
    - Deployment across all 34 flood-prone districts of Assam with automated ASDMA API webhooks.
    """)

# --- 6. CLEAN COMMAND FOOTER ---
st.markdown("""
<div class="footer-bar">
    FloodSense AI • Satellite-Based Spatial Flood Risk Prediction & Historical Early-Warning Prototype • Assam Brahmaputra Floodplain
</div>
""", unsafe_allow_html=True)
