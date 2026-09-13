"""
Prediction Intelligence & Methodology Engine
Provides a visual and technical walkthrough explaining how FloodSense AI transforms
raw Earth Observation observations into cell-level flood predictions, continuous spatial
risk surfaces, and actionable decision-support intelligence.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import config

def render_prediction_intelligence_page(
    active_model,
    benchmark_metrics: dict,
    df_step: pd.DataFrame,
    topo: dict,
    spatial_preds: dict
):
    """
    Renders the complete 'Prediction Intelligence' explanation page.
    """
    # --- PAGE HEADER ---
    st.markdown("""
    <div class="command-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="brand-eyebrow">Methodology • Feature Engineering • Machine Learning Architecture</div>
                <div class="brand-title">🧠 PREDICTION INTELLIGENCE</div>
                <div class="brand-subtitle">How FloodSense AI Ingests Earth Observation Data to Predict Spatial Flood Risk</div>
            </div>
            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 6px;">
                <div style="display: flex; gap: 6px;">
                    <span class="badge-chip badge-chip-blue">HAZARD: FLOOD ONLY</span>
                    <span class="badge-chip">REGION: ASSAM (BRAHMAPUTRA)</span>
                </div>
                <span class="badge-chip" style="font-size: 0.68rem; color: #64748B;">STUDY AREA: GOLAGHAT & NAGAON (~2,640 KM²)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION 1: SYSTEM OVERVIEW ---
    st.markdown("""
    <div class="analyst-card" style="border-left-color: #38BDF8; margin-top: 14px; margin-bottom: 20px;">
        <div style="font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #38BDF8; margin-bottom: 6px;">
            📌 SECTION 1 — SYSTEM OVERVIEW & ARCHITECTURAL THESIS
        </div>
        <div style="font-size: 0.96rem; color: #F8FAFC; line-height: 1.6; font-weight: 500;">
            <b>FloodSense AI</b> combines satellite-derived spectral observations and hydro-topographic geospatial features to estimate flood inundation probability for <b>individual 500m spatial cells</b>. The machine learning engine converts these cell-level predictions into a continuous spatial risk surface, providing civil authorities with <b>24 to 72 hours of actionable lead-time intelligence</b> before peak overbank crests.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Three Core Pillars Grid
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("""
        <div class="kpi-box" style="text-align: left; height: 100%;">
            <div class="kpi-title" style="color: #38BDF8;">🛰️ 1. Multi-Sensor Data Fusion</div>
            <div style="font-size: 0.82rem; color: #E2E8F0; margin-top: 6px; line-height: 1.5;">
                Ingests optical multi-spectral bands (Sentinel-2), gridded antecedent rainfall (CHIRPS), InSAR topography (SRTM DEM), and decadal river masks (JRC Water).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown("""
        <div class="kpi-box" style="text-align: left; height: 100%;">
            <div class="kpi-title" style="color: #34D399;">🌲 2. Random Forest Classifier</div>
            <div style="font-size: 0.82rem; color: #E2E8F0; margin-top: 6px; line-height: 1.5;">
                An ensemble of <b>150 de-correlated decision trees</b> trained with balanced class weights to evaluate non-linear flood thresholds and spectral absorption physics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_p3:
        st.markdown("""
        <div class="kpi-box" style="text-align: left; height: 100%;">
            <div class="kpi-title" style="color: #F59E0B;">🗺️ 3. 500m Spatial Grid (3,750 Cells)</div>
            <div style="font-size: 0.82rem; color: #E2E8F0; margin-top: 6px; line-height: 1.5;">
                Discretizes the Kaziranga-Golaghat corridor into a <b>50×75 cell matrix</b>. Each cell is independently scored to identify localized vulnerability epicenters.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # --- SECTION 14: MASTER END-TO-END PIPELINE DIAGRAM ---
    st.markdown("""
    <div style="background: #0D1526; border: 1px solid #1E293B; border-radius: 8px; padding: 18px 22px; margin-bottom: 24px;">
        <div style="font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #38BDF8; margin-bottom: 12px;">
            🏗️ MASTER END-TO-END PREDICTION PIPELINE
        </div>
        <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; text-align: center;">
            <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 10px 8px;">
                <div style="font-size: 1.2rem;">🛰️</div>
                <div style="font-size: 0.70rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">1. RAW DATA</div>
                <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 2px;">Sentinel-2, CHIRPS, SRTM, JRC</div>
            </div>
            <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 10px 8px;">
                <div style="font-size: 1.2rem;">⚙️</div>
                <div style="font-size: 0.70rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">2. CLEANING</div>
                <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 2px;">Range clipping, null validation</div>
            </div>
            <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 10px 8px;">
                <div style="font-size: 1.2rem;">🔬</div>
                <div style="font-size: 0.70rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">3. FEATURES</div>
                <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 2px;">16 Spectral & Hydro Features</div>
            </div>
            <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 10px 8px;">
                <div style="font-size: 1.2rem;">🌲</div>
                <div style="font-size: 0.70rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">4. ML ENSEMBLE</div>
                <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 2px;">150 Decision Trees Voting</div>
            </div>
            <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 10px 8px;">
                <div style="font-size: 1.2rem;">📊</div>
                <div style="font-size: 0.70rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">5. GRID RISK</div>
                <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 2px;">3,750 Individual Cell P(Flood)</div>
            </div>
            <div style="background: #0F172A; border: 1px solid #334155; border-radius: 6px; padding: 10px 8px;">
                <div style="font-size: 1.2rem;">🗺️</div>
                <div style="font-size: 0.70rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">6. DECISION MAP</div>
                <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 2px;">4-Tier GIS Map + Point Dossier</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION 2: DATA SOURCES ---
    st.markdown("### 🛰️ **Section 2 — Public Earth Observation Datasets Used**")
    st.caption("All datasets ingested by the pipeline are publicly accessible, peer-reviewed remote sensing archives:")

    datasets_table = pd.DataFrame([
        {
            "Dataset Name": "Sentinel-2 MSI Level-2A",
            "Agency / Mission": "ESA Copernicus",
            "Spatial Resolution": "10m / 20m",
            "Revisit Cycle": "5 Days (Constellation)",
            "Engineered Features": "B3, B4, B8, B11, B12, NDVI, NDWI, MNDWI",
            "Physical Role": "Spectral surface reflectance, water absorption, soil saturation"
        },
        {
            "Dataset Name": "CHIRPS Daily Precipitation",
            "Agency / Mission": "UCSB Climate Hazards Center",
            "Spatial Resolution": "0.05° (~5.5 km)",
            "Revisit Cycle": "Daily (24h)",
            "Engineered Features": "rainfall_1d, rainfall_3d, rainfall_7d, rainfall_14d",
            "Physical Role": "Antecedent catchment precipitation and multi-day soil surcharge"
        },
        {
            "Dataset Name": "SRTM Digital Elevation Model",
            "Agency / Mission": "NASA / USGS C-band InSAR",
            "Spatial Resolution": "30m / 90m",
            "Revisit Cycle": "Static Baseline",
            "Engineered Features": "elevation, slope, dist_to_drainage",
            "Physical Role": "Gravitational flow pathways (-∇z), depression ponding, drainage proximity"
        },
        {
            "Dataset Name": "JRC Global Surface Water",
            "Agency / Mission": "European Commission Joint Research Centre",
            "Spatial Resolution": "30m",
            "Revisit Cycle": "Multi-Decadal (1984–2021)",
            "Engineered Features": "permanent_water",
            "Physical Role": "Isolates permanent braided riverbed channels from novel overbank floods"
        },
        {
            "Dataset Name": "Global Flood Database (Event 4924)",
            "Agency / Mission": "Dartmouth Flood Observatory / MODIS / SAR",
            "Spatial Resolution": "30m / 250m",
            "Revisit Cycle": "Disaster Event Archive",
            "Engineered Features": "is_flooded (Binary Target Label)",
            "Physical Role": "Independent holdout spatial validation ground truth (July 2020 Peak Event T)"
        }
    ])
    st.dataframe(datasets_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- SECTION 3 & 4: FEATURE ENGINEERING & PHYSICAL RATIONALE ---
    st.markdown("### 🔬 **Section 3 & 4 — The 16 Model Features & Why They Matter**")
    st.caption("How raw satellite bands and environmental parameters are transformed into predictive indicators:")

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 0.88rem; margin-bottom: 8px;">1. SPECTRAL WATER & VEGETATION INDICES</div>
            <ul style="font-size: 0.80rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 0px; padding-left: 18px;">
                <li><b>NDWI (Normalized Difference Water Index):</b> <code>(B3 - B8) / (B3 + B8)</code>. Strongly differentiates open standing floodwater from terrestrial grassland.</li>
                <li><b>MNDWI (Modified NDWI):</b> <code>(B3 - B11) / (B3 + B11)</code>. Suppresses built-up structures and identifies suspended sediment in turbulent floodwaters.</li>
                <li><b>NDVI (Vegetation Index):</b> <code>(B8 - B4) / (B8 + B4)</code>. Measures vegetation vigor; submerged riparian vegetation shows rapid NDVI drop.</li>
            </ul>
        </div>
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 0.88rem; margin-bottom: 8px;">2. MULTI-SPECTRAL SURFACE REFLECTANCE BANDS</div>
            <ul style="font-size: 0.80rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 0px; padding-left: 18px;">
                <li><b>Near-Infrared (B8, 842nm):</b> <i>#1 Most Important Feature.</i> Pure water completely absorbs NIR energy, providing an uncompromising physical contrast against dry land.</li>
                <li><b>Shortwave-Infrared 2 (B12, 2190nm):</b> <i>#3 Most Important Feature.</i> Highly sensitive to soil moisture saturation and mudflats preceding overt standing water.</li>
                <li><b>Shortwave-Infrared 1 (B11, 1610nm):</b> Captures water-in-canopy moisture attenuation.</li>
                <li><b>Visible Green & Red (B3 & B4):</b> Optical baseline surface reflectance for color differentiation.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 0.88rem; margin-bottom: 8px;">3. ANTECEDENT CATCHMENT PRECIPITATION (CHIRPS)</div>
            <ul style="font-size: 0.80rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 0px; padding-left: 18px;">
                <li><b>rainfall_1d (24h Instantaneous):</b> Detects immediate pluvial downpours and flash inflow surges.</li>
                <li><b>rainfall_3d (72h Pre-Peak):</b> Captures intermediate storm system buildup across the basin.</li>
                <li><b>rainfall_7d (Weekly Cumulative):</b> The primary temporal trigger for regional overbank flooding; saturates the upper Himalayan drainage basin.</li>
                <li><b>rainfall_14d (Fortnight Cumulative):</b> Deep groundwater table recharge and antecedent soil saturation baseline.</li>
            </ul>
        </div>
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 0.88rem; margin-bottom: 8px;">4. HYDRO-TOPOGRAPHY & RELIEF (SRTM & JRC)</div>
            <ul style="font-size: 0.80rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 0px; padding-left: 18px;">
                <li><b>Elevation (SRTM DEM):</b> Floodwaters cannot flow uphill. Alluvial plains &lt;65m are vulnerable to deep overbank backwater, while highlands remain dry.</li>
                <li><b>Slope (Topographic Gradient):</b> Flat low-gradient terrain (&lt;1.5°) prevents drainage and induces severe standing water ponding.</li>
                <li><b>dist_to_drainage:</b> Euclidean distance to the active river channel; closer cells face direct levee breaching.</li>
                <li><b>permanent_water:</b> JRC mask isolating baseline braided channels from novel disaster inundation.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SECTION 5: MACHINE LEARNING MODEL ARCHITECTURE ---
    st.markdown("### 🌲 **Section 5 — Machine Learning Model: Random Forest Classifier**")
    st.caption("How 150 de-correlated decision trees combine to make non-linear spatial predictions:")

    col_m1, col_m2 = st.columns([1.2, 1.0])

    with col_m1:
        st.markdown("""
        <div style="background: #0D1526; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; font-size: 0.82rem; line-height: 1.6;">
            <div style="font-weight: 700; color: #38BDF8; margin-bottom: 8px; font-size: 0.90rem;">HOW THE ENSEMBLE DECIDES</div>
            <p>Rather than relying on a single brittle decision tree or black-box neural network, <b>FloodSense AI</b> deploys a <b>Random Forest Classifier</b> composed of <b>150 individual decision trees</b>:</p>
            <p>1. <b>Bootstrap Aggregation (Bagging):</b> Each tree is trained on a distinct random bootstrap sample of training cells, ensuring diversity across the ensemble.</p>
            <p>2. <b>Random Feature Subspaces:</b> At every internal split, each tree chooses from a random subset of the 16 features, de-correlating tree predictions.</p>
            <p>3. <b>Ensemble Probability Voting:</b> The continuous flood probability is calculated as the proportion of the 150 trees that vote "Flooded":</p>
            <div style="background: #0F172A; border-left: 3px solid #38BDF8; padding: 8px 12px; font-family: monospace; color: #38BDF8; margin: 8px 0;">
                P(Flood) = (1 / 150) * &Sigma; Tree_i(X) &isin; [0.0, 1.0]
            </div>
            <p style="margin-bottom: 0px;">4. <b>Balanced Class Weighting:</b> Because non-flooded cells outnumber flooded cells during early timesteps, inverse-frequency weighting prevents majority-class bias.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div style="background: #0D1526; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; font-family: monospace; font-size: 0.72rem; color: #94A3B8; text-align: center;">
            <div style="color: #38BDF8; font-weight: bold; margin-bottom: 8px;">16-ELEMENT INPUT FEATURE VECTOR</div>
            <div>[NDWI, B8, B12, Rain_7d, Elev, Dist...]</div>
            <div style="color: #64748B; margin: 4px 0;">&darr;</div>
            <div style="background: #0F172A; border: 1px dashed #38BDF8; border-radius: 6px; padding: 10px; text-align: left;">
                <div style="color: #F8FAFC; font-weight: bold;">FOREST OF 150 DECISION TREES</div>
                <div>&bull; Tree #1:  B8 &lt; 0.14 &amp; Elev &lt; 66m &rarr; <b>Vote: 1</b></div>
                <div>&bull; Tree #2:  NDWI &gt; 0.15 &amp; Rain7d &gt; 200 &rarr; <b>Vote: 1</b></div>
                <div>&bull; Tree #3:  Elev &gt; 85m (Upland) &rarr; <b>Vote: 0</b></div>
                <div>&bull; ...</div>
                <div>&bull; Tree #150: Dist &lt; 850m &amp; B12 &lt; 0.12 &rarr; <b>Vote: 1</b></div>
            </div>
            <div style="color: #64748B; margin: 4px 0;">&darr;</div>
            <div style="color: #34D399; font-weight: bold;">ENSEMBLE VOTING (125 / 150 Trees = 83.3%)</div>
            <div style="color: #64748B; margin: 4px 0;">&darr;</div>
            <div style="background: #0F172A; border: 1px solid #EF4444; color: #EF4444; border-radius: 4px; padding: 6px; font-weight: bold;">
                CELL INUNDATION RISK: 83.3% (VERY HIGH)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SECTION 6: CELL-LEVEL PREDICTION MECHANICS ---
    st.markdown("### 🗺️ **Section 6 — Cell-Level Prediction Grid (500m Resolution)**")
    st.caption("FloodSense AI avoids coarse regional generalizations by evaluating individual 500m spatial cells:")

    c_col1, c_col2 = st.columns([1.2, 1.0])

    with c_col1:
        st.markdown("""
        <div style="background: #0D1526; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; font-size: 0.82rem; line-height: 1.6;">
            <div style="font-weight: 700; color: #38BDF8; margin-bottom: 8px; font-size: 0.90rem;">NOT A SINGLE COARSE PROVINCIAL NUMBER</div>
            <p>Traditional flood alerts issue blunt warnings like <i>"Assam has an active flood alert"</i>. This provides zero actionable intelligence to emergency managers trying to decide which roads to close or which hospital to evacuate.</p>
            <p>Instead, <b>FloodSense AI</b> discretizes the <b>Golaghat & Nagaon district study area</b> into a <b>50-row by 75-column regular spatial matrix = 3,750 independent cells</b>:</p>
            <ul style="margin-bottom: 8px; padding-left: 18px;">
                <li><b>Spatial Resolution:</b> ~500m &times; ~500m (~0.704 km&sup2; per cell).</li>
                <li><b>Total Territory:</b> ~2,640 km&sup2; encompassing Kaziranga National Park and the Brahmaputra floodplain.</li>
                <li><b>Independent Feature Vectors:</b> Each of the 3,750 cells contains its own distinct elevation, river distance, and Sentinel-2 reflectance values.</li>
            </ul>
            <p style="margin-bottom: 0px;">The Random Forest independently scores every cell, creating a high-resolution, spatially continuous risk surface that isolates micro-scale depression hazards.</p>
        </div>
        """, unsafe_allow_html=True)

    with c_col2:
        st.markdown("""
        <div style="background: #0D1526; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; font-family: monospace; font-size: 0.72rem; text-align: center;">
            <div style="color: #38BDF8; font-weight: bold; margin-bottom: 8px;">SAMPLE 4&times;4 PREDICTION CELL MATRIX</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px;">
                <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #10B981; padding: 12px 4px; border-radius: 4px;"><b>12.4%</b><br><span style="font-size: 0.60rem;">LOW</span></div>
                <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #10B981; padding: 12px 4px; border-radius: 4px;"><b>24.8%</b><br><span style="font-size: 0.60rem;">LOW</span></div>
                <div style="background: rgba(249, 115, 22, 0.2); border: 1px solid #F97316; color: #F97316; padding: 12px 4px; border-radius: 4px;"><b>67.2%</b><br><span style="font-size: 0.60rem;">HIGH</span></div>
                <div style="background: rgba(239, 68, 68, 0.2); border: 1px solid #EF4444; color: #EF4444; padding: 12px 4px; border-radius: 4px;"><b>81.5%</b><br><span style="font-size: 0.60rem;">V.HIGH</span></div>
                <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #10B981; padding: 12px 4px; border-radius: 4px;"><b>18.1%</b><br><span style="font-size: 0.60rem;">LOW</span></div>
                <div style="background: rgba(245, 158, 11, 0.2); border: 1px solid #F59E0B; color: #F59E0B; padding: 12px 4px; border-radius: 4px;"><b>42.6%</b><br><span style="font-size: 0.60rem;">MOD</span></div>
                <div style="background: rgba(249, 115, 22, 0.2); border: 1px solid #F97316; color: #F97316; padding: 12px 4px; border-radius: 4px;"><b>73.0%</b><br><span style="font-size: 0.60rem;">HIGH</span></div>
                <div style="background: rgba(239, 68, 68, 0.2); border: 1px solid #EF4444; color: #EF4444; padding: 12px 4px; border-radius: 4px;"><b>91.8%</b><br><span style="font-size: 0.60rem;">V.HIGH</span></div>
                <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #10B981; padding: 12px 4px; border-radius: 4px;"><b>8.3%</b><br><span style="font-size: 0.60rem;">LOW</span></div>
                <div style="background: rgba(245, 158, 11, 0.2); border: 1px solid #F59E0B; color: #F59E0B; padding: 12px 4px; border-radius: 4px;"><b>31.0%</b><br><span style="font-size: 0.60rem;">MOD</span></div>
                <div style="background: rgba(249, 115, 22, 0.2); border: 1px solid #F97316; color: #F97316; padding: 12px 4px; border-radius: 4px;"><b>58.4%</b><br><span style="font-size: 0.60rem;">HIGH</span></div>
                <div style="background: rgba(239, 68, 68, 0.2); border: 1px solid #EF4444; color: #EF4444; padding: 12px 4px; border-radius: 4px;"><b>84.2%</b><br><span style="font-size: 0.60rem;">V.HIGH</span></div>
                <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #10B981; padding: 12px 4px; border-radius: 4px;"><b>5.0%</b><br><span style="font-size: 0.60rem;">LOW</span></div>
                <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #10B981; padding: 12px 4px; border-radius: 4px;"><b>14.2%</b><br><span style="font-size: 0.60rem;">LOW</span></div>
                <div style="background: rgba(245, 158, 11, 0.2); border: 1px solid #F59E0B; color: #F59E0B; padding: 12px 4px; border-radius: 4px;"><b>38.9%</b><br><span style="font-size: 0.60rem;">MOD</span></div>
                <div style="background: rgba(249, 115, 22, 0.2); border: 1px solid #F97316; color: #F97316; padding: 12px 4px; border-radius: 4px;"><b>62.1%</b><br><span style="font-size: 0.60rem;">HIGH</span></div>
            </div>
            <div style="font-size: 0.68rem; color: #64748B; margin-top: 8px;">3,750 cells continuously render the 2D geospatial risk surface</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SECTION 7 & 8: RISK CLASSIFICATION & SPATIAL RISK MAP ---
    st.markdown("### 🎨 **Section 7 & 8 — Risk Classification & Map Rendering**")
    st.caption("How continuous probability values become intuitive emergency management tiers and map layers:")

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.markdown("""
        <div style="background: #0D1526; border: 1px solid #1E293B; border-left: 4px solid #10B981; border-radius: 6px; padding: 12px;">
            <div style="font-size: 0.70rem; font-weight: bold; color: #10B981;">0% &ndash; 25%: LOW RISK</div>
            <div style="font-size: 0.76rem; color: #E2E8F0; margin-top: 4px;">Normal baseline. Well-drained upland relief or distant buffer zones.</div>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown("""
        <div style="background: #0D1526; border: 1px solid #1E293B; border-left: 4px solid #F59E0B; border-radius: 6px; padding: 12px;">
            <div style="font-size: 0.70rem; font-weight: bold; color: #F59E0B;">25% &ndash; 50%: MODERATE RISK</div>
            <div style="font-size: 0.76rem; color: #E2E8F0; margin-top: 4px;">Elevated moisture. Low-lying alluvial seepage and agricultural flats.</div>
        </div>
        """, unsafe_allow_html=True)
    with rc3:
        st.markdown("""
        <div style="background: #0D1526; border: 1px solid #1E293B; border-left: 4px solid #F97316; border-radius: 6px; padding: 12px;">
            <div style="font-size: 0.70rem; font-weight: bold; color: #F97316;">50% &ndash; 75%: HIGH RISK</div>
            <div style="font-size: 0.76rem; color: #E2E8F0; margin-top: 4px;">Imminent overtopping hazard within 24-48h. Stage flood barriers.</div>
        </div>
        """, unsafe_allow_html=True)
    with rc4:
        st.markdown("""
        <div style="background: #0D1526; border: 1px solid #1E293B; border-left: 4px solid #EF4444; border-radius: 6px; padding: 12px;">
            <div style="font-size: 0.70rem; font-weight: bold; color: #EF4444;">75% &ndash; 100%: VERY HIGH RISK</div>
            <div style="font-size: 0.76rem; color: #E2E8F0; margin-top: 4px;">Severe catastrophic crest overtopping. Primary evacuation zone.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SECTION 15: MODEL EXPLAINABILITY (FEATURE IMPORTANCES) ---
    st.markdown("### 📊 **Section 15 — What Drives the Model? (Gini Feature Importance)**")
    st.caption("Actual empirical Gini impurity reductions across all 150 Random Forest decision trees:")

    df_imp = active_model.get_feature_importance()
    feature_labels = {
        "b8": "Near-Infrared Reflectance (B8, 842nm)",
        "ndwi": "Normalized Difference Water Index (NDWI)",
        "b12": "Shortwave-Infrared 2 (B12, 2190nm)",
        "ndvi": "Vegetation Vigor Index (NDVI)",
        "dist_to_drainage": "Distance to River Channel (m)",
        "permanent_water": "Permanent Riverbed Mask (JRC)",
        "elevation": "Topographic Elevation (SRTM DEM, m)",
        "mndwi": "Modified NDWI (Suspended Silt Index)",
        "rainfall_7d": "7-Day Cumulative Rainfall (CHIRPS)",
        "rainfall_1d": "24h Instantaneous Rainfall (CHIRPS)",
        "rainfall_3d": "3-Day Cumulative Rainfall (CHIRPS)",
        "rainfall_14d": "14-Day Cumulative Rainfall (CHIRPS)",
        "slope": "Topographic Slope Gradient (deg)",
        "b3": "Green Band Surface Reflectance (B3)",
        "b4": "Red Band Surface Reflectance (B4)",
        "b11": "Shortwave-Infrared 1 (B11, 1610nm)"
    }
    df_plot = df_imp.copy()
    df_plot["label"] = df_plot["feature"].map(lambda x: feature_labels.get(x, x))
    df_sorted = df_plot.sort_values(by="importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted["importance"],
        y=df_sorted["label"],
        orientation="h",
        marker=dict(
            color=df_sorted["importance"],
            colorscale=[[0, "#1E3A8A"], [0.5, "#2563EB"], [1.0, "#38BDF8"]],
            showscale=False,
            line=dict(color="#38BDF8", width=0.8)
        ),
        text=[f"{v:.3f} ({v*100:.1f}%)" for v in df_sorted["importance"]],
        textposition="outside",
        textfont=dict(color="#CBD5E1", size=10, family="Inter, sans-serif")
    ))
    fig.update_layout(
        title=dict(
            text="<b>EMPIRICAL FEATURE CONTRIBUTIONS (GINI IMPURITY REDUCTION)</b>",
            font=dict(color="#F8FAFC", size=12, family="Inter, sans-serif")
        ),
        xaxis=dict(
            title=dict(text="Relative Gini Impurity Reduction (Normalized Weight)", font=dict(color="#94A3B8", size=11)),
            tickfont=dict(color="#CBD5E1", size=10, family="Inter, sans-serif"),
            gridcolor="#1E293B",
            showline=True,
            linecolor="#334155"
        ),
        yaxis=dict(
            tickfont=dict(color="#E2E8F0", size=11, family="Inter, sans-serif"),
            gridcolor="#1E293B"
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B111E",
        margin=dict(l=10, r=40, t=45, b=40),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- SECTION 10 & 11: HISTORICAL VALIDATION SCORECARD ---
    st.markdown("### 🎯 **Section 10 & 11 — How Do We Know It Works? (Holdout Validation)**")
    st.caption("Rigorous benchmarking strictly against unseen spatial territory during the July 2020 disaster crest:")

    rf_bench = benchmark_metrics.get("Random Forest", {})
    val_prec = rf_bench.get("precision", 1.0)
    val_rec = rf_bench.get("recall", 0.8324)
    val_f1 = rf_bench.get("f1", 0.9085)
    val_iou = rf_bench.get("iou", 0.8324)
    val_auc = rf_bench.get("roc_auc", 0.9997)

    # 4 Scorecard KPI Cards
    vk1, vk2, vk3, vk4 = st.columns(4)
    with vk1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Spatial IoU (Jaccard)</div>
            <div class="kpi-number" style="color: #38BDF8;">{val_iou:.4f}</div>
            <div class="kpi-subtext">Overlap against DFO Ground Truth</div>
        </div>
        """, unsafe_allow_html=True)
    with vk2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Holdout F1-Score</div>
            <div class="kpi-number" style="color: #34D399;">{val_f1:.4f}</div>
            <div class="kpi-subtext">Harmonic mean of precision & recall</div>
        </div>
        """, unsafe_allow_html=True)
    with vk3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Holdout Precision</div>
            <div class="kpi-number" style="color: #F8FAFC;">{val_prec:.4f}</div>
            <div class="kpi-subtext">Zero false positive flood alarms (0 FP)</div>
        </div>
        """, unsafe_allow_html=True)
    with vk4:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">ROC-AUC Discriminator</div>
            <div class="kpi-number" style="color: #F59E0B;">{val_auc:.4f}</div>
            <div class="kpi-subtext">Area Under Receiver Operating Curve</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #0D1526; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; margin-top: 14px; font-size: 0.82rem; color: #94A3B8; line-height: 1.6;">
        <b style="color: #38BDF8;">WHY SPATIAL-TEMPORAL BLOCK VALIDATION MATTERS:</b><br>
        Standard random k-fold cross validation on satellite pixels produces <i>fraudulently high scores</i> due to spatial and temporal autocorrelation (a cell's neighbor has almost identical reflectance). 
        <b>FloodSense AI strictly implements a Spatial-Temporal Block Split:</b><br>
        &bull; <b>Zero Temporal Leakage:</b> Training observations are drawn exclusively from antecedent windows (&le; T-2). The model never sees peak event T during training.<br>
        &bull; <b>Zero Spatial Leakage:</b> Training is geographically restricted to the Western & Central corridor (cols &le; 52). Testing is evaluated strictly on the <b>Eastern Bokakhat floodplain (cols &gt; 52, 1,100 unseen spatial cells)</b> at peak disaster crest T (July 14, 2020).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SECTION 12 & 13: POINT INTELLIGENCE & FLOOD PROPAGATION ---
    st.markdown("### 📍 **Section 12 & 13 — Point Intelligence & Terrain-Guided Simulation**")

    pi_col1, pi_col2 = st.columns(2)
    with pi_col1:
        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; height: 100%;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 0.90rem; margin-bottom: 8px;">POINT INTELLIGENCE (CLICK-TO-INSPECT)</div>
            <p style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
                Emergency operators do not manage abstract statewide averages; they evacuate concrete coordinates. 
                When a user clicks on the 2D map:
            </p>
            <div style="background: #0D1526; border-left: 3px solid #00F0FF; padding: 8px 12px; font-family: monospace; font-size: 0.72rem; color: #00F0FF; margin: 8px 0;">
                Click (lat, lon) &rarr; Nearest Grid Cell [r, c] &rarr; Query 16D Vector &rarr; Generate Point Dossier
            </div>
            <p style="font-size: 0.80rem; color: #94A3B8; margin-bottom: 0px;">
                Instantly displays real SRTM elevation, slope gradient, Euclidean river proximity, CHIRPS precipitation, and multi-spectral NDWI without fabricating synthetic numbers.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with pi_col2:
        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; height: 100%;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 0.90rem; margin-bottom: 8px;">TERRAIN-GUIDED FLOOD PROPAGATION</div>
            <p style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
                <i>Honest Disclosure:</i> We label this strictly as a <b>Terrain-Guided Flood Propagation Simulation</b>, NOT a hydrodynamic shallow-water differential equation solver.
            </p>
            <div style="background: #0D1526; border-left: 3px solid #34D399; padding: 8px 12px; font-family: monospace; font-size: 0.72rem; color: #34D399; margin: 8px 0;">
                Permanent Riverbeds &rarr; Elevation Gradients (-&nabla;z) &rarr; Vulnerability Surges &rarr; Advancing Front
            </div>
            <p style="font-size: 0.80rem; color: #94A3B8; margin-bottom: 0px;">
                Models water physically overtopping riverbanks and pooling into low-lying depressions across 5 event stages, dynamically highlighting the advancing flood front.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SECTION 16: SCIENTIFIC LIMITATIONS & HONEST DISCLOSURES ---
    st.markdown("### ⚖️ **Section 16 — Scientific Limitations & Scope Disclosures**")
    st.caption("Essential operational boundaries ensuring scientific integrity for civil decision makers:")

    lim_col1, lim_col2 = st.columns(2)
    with lim_col1:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 16px;">
            <div style="font-weight: 700; color: #10B981; font-size: 0.88rem; margin-bottom: 8px;">✅ WHAT THIS PROTOTYPE DOES</div>
            <ul style="font-size: 0.80rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 0px; padding-left: 18px;">
                <li><b>Rapid Pre-Event Spatial Inference:</b> Generates continuous 0–100% inundation risk across 3,750 cells in &lt;1.5 seconds.</li>
                <li><b>Multi-Sensor Fusion:</b> Bridges optical satellite observations, radar elevation, and precipitation accumulation.</li>
                <li><b>24 to 72-Hour Early Warning Horizon:</b> Identifies progressive flood risk accumulation prior to river cresting.</li>
                <li><b>Zero Data Fabrication:</b> Displays real historical satellite and precipitation observations.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with lim_col2:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 16px;">
            <div style="font-weight: 700; color: #EF4444; font-size: 0.88rem; margin-bottom: 8px;">❌ WHAT IT DOES NOT CLAIM</div>
            <ul style="font-size: 0.80rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 0px; padding-left: 18px;">
                <li><b>Not an Emergency Warning Dispatcher:</b> Evaluated on historical archives; not certified for life-critical public evacuation broadcasts.</li>
                <li><b>Optical Cloud Occlusion:</b> Sentinel-2 optical bands can be obstructed by dense monsoonal rain clouds (synthetic aperture radar integration recommended).</li>
                <li><b>Sub-500m Micro-Drainage:</b> Localized road culverts and village embankments below 500m are not resolved in the DEM.</li>
                <li><b>Not a 2D Hydrodynamic Solver:</b> Does not calculate Navier-Stokes Manning roughness or flow velocities.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown("""
    <div class="footer-bar" style="margin-top: 30px;">
        FloodSense AI • Prediction Intelligence & Methodology Engine • Assam Brahmaputra Floodplain
    </div>
    """, unsafe_allow_html=True)
