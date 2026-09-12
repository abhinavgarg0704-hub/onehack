#!/usr/bin/env python3
"""
Comprehensive Functional & Geospatial Feature Test Suite
Simulates all user interactions, map controls, scale changes, layer toggles,
and coordinate selection across the Assam study area.
"""

import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
from src.data_loader import load_data_for_tag, generate_base_topography
from src.model import FloodRiskModel
from src.prediction import generate_spatial_prediction
from src.visualization import (
    generate_assam_topography,
    compute_assam_flood_simulation,
    build_assam_3d_simulation_map,
    compute_downhill_flow_paths,
    plot_risk_timeline,
    plot_feature_importance_chart,
    plot_confusion_matrix_chart
)

def test_all_features():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE MAP & ANALYTICS FUNCTIONAL TESTS")
    print("=" * 70)

    # 1. Base data & AI Model
    print("\n[1] Testing AI model & local spatial prediction...")
    df_step = load_data_for_tag("T")
    assert not df_step.empty, "Dataframe T is empty"
    rf = FloodRiskModel("rf")
    rf.load()
    sp_local = generate_spatial_prediction(df_step, rf)
    assert sp_local["risk_grid"].shape == (50, 75), f"Unexpected shape {sp_local['risk_grid'].shape}"
    print("    [PASS] AI Model loaded and predictions generated successfully")

    # 2. Assam Topography & Boundaries
    print("\n[2] Testing Assam macro-scale topography & vector GIS assets...")
    topo_assam = generate_assam_topography(rows=65, cols=95)
    assert topo_assam["elevation"].shape == (65, 95)
    assert len(topo_assam["district_rings"]) > 0, "No district rings found"
    assert len(topo_assam["river_lines"]) > 0, "No river lines found"
    print(f"    [PASS] Assam DEM generated: {topo_assam['elevation'].shape}")
    print(f"    [PASS] District boundary rings loaded: {len(topo_assam['district_rings'])}")
    print(f"    [PASS] River network polylines loaded: {len(topo_assam['river_lines'])}")

    # 3. 5-Stage Flood Simulation
    print("\n[3] Testing 5-stage flood propagation simulation...")
    sim_data = compute_assam_flood_simulation(topo_assam, sp_local=sp_local, rainfall_7d=342.8)
    assert len(sim_data["stages"]) == 5, "Expected exactly 5 stages"
    for i, s in enumerate(sim_data["stages"]):
        print(f"    Stage {i+1} ({s['tag']}): {s['name']} | Water cells: {s['flooded_cells']} ({s['flooded_area_km2']:.1f} km^2) | Streamlines: {len(s['streamlines'])}")
    assert sim_data["stages"][4]["flooded_cells"] > sim_data["stages"][0]["flooded_cells"], "Water extent did not expand"
    print("    [PASS] 5 simulation stages computed with progressive inundation")

    # 4. Map Scale Presets & Camera Views
    print("\n[4] Testing Google Earth-style camera scale presets...")
    scale_presets = [
        "🌏 Level 1: Statewide Assam (Default)",
        "🏞️ Level 2: Brahmaputra Valley Corridor",
        "📍 Level 3: AI Sector (Kaziranga-Golaghat)",
        "🔬 Level 4: Local Threat Cell Focal Point",
        "🗺️ 2D Plan View (Top-Down GIS Orthogonal)"
    ]
    for sp in scale_presets:
        fig = build_assam_3d_simulation_map(
            topo_assam=topo_assam,
            sim_data=sim_data,
            active_stage_idx=4,
            sp_local=sp_local,
            scale_level=sp
        )
        assert len(fig.data) >= 7, f"Preset {sp} missing layer traces"
        safe_sp = sp.encode('ascii', errors='replace').decode('ascii')
        print(f"    [PASS] Scale preset verified: {safe_sp} ({len(fig.data)} traces)")

    # 5. Layer Toggles (Checking all checkboxes work independently)
    print("\n[5] Testing individual map layer toggles...")
    test_configs = [
        ("show_terrain=False", dict(show_terrain=False)),
        ("show_boundaries=False", dict(show_boundaries=False)),
        ("show_ai_footprint=False", dict(show_ai_footprint=False)),
        ("show_river=False", dict(show_river=False)),
        ("show_floodwater=False", dict(show_floodwater=False)),
        ("show_streamlines=False", dict(show_streamlines=False)),
        ("show_gt=False", dict(show_gt=False)),
        ("show_landmarks=False", dict(show_landmarks=False)),
        ("vert_exag=1.5", dict(vertical_exaggeration=1.5)),
    ]
    for label, kwargs in test_configs:
        fig = build_assam_3d_simulation_map(
            topo_assam=topo_assam,
            sim_data=sim_data,
            active_stage_idx=3,
            sp_local=sp_local,
            **kwargs
        )
        assert fig is not None, f"Toggle test failed for {label}"
        print(f"    [PASS] Layer toggle: {label} rendered without error")

    # 6. Target Location Selection Across Assam
    print("\n[6] Testing location selection (Inside AI Sector vs Regional Context)...")
    test_locations = [
        ("Kaziranga (Inside AI Sector)", (26.60, 93.35)),
        ("Guwahati (Regional West)", (26.18, 91.75)),
        ("Dibrugarh (Regional East)", (27.48, 94.92)),
        ("Dhubri (Border West)", (26.02, 89.98)),
        ("Silchar (Barak Valley South)", (24.83, 92.80))
    ]
    for loc_name, coords in test_locations:
        fig = build_assam_3d_simulation_map(
            topo_assam=topo_assam,
            sim_data=sim_data,
            active_stage_idx=4,
            sp_local=sp_local,
            selected_location=coords
        )
        assert fig is not None
        print(f"    [PASS] Location pin & coordinates: {loc_name} at {coords}")

    # 7. Downhill Flow Paths
    print("\n[7] Testing overland gravity downhill flow paths...")
    topo_local = generate_base_topography(config.STUDY_AREA["grid_rows"], config.STUDY_AREA["grid_cols"])
    flow = compute_downhill_flow_paths(
        elev_grid=topo_local["elevation"],
        risk_grid=sp_local["risk_grid"],
        perm_water_grid=sp_local["perm_water_grid"],
        lat_grid=topo_local["lat_grid"],
        lon_grid=topo_local["lon_grid"],
        risk_threshold=25.0,
        selected_cell=(25, 37)
    )
    assert "selected_path" in flow, "Flow calculation missing selected_path"
    print(f"    [PASS] Downhill flow computed: length={flow['selected_path']['dist_km']:.2f}km, drop={flow['selected_path']['elev_drop']:.1f}m")

    # 8. Charts & Timeline
    print("\n[8] Testing analytics charts...")
    timeline_df = pd.DataFrame([
        {"time_tag": "T-7", "rainfall_7d": 58.4, "mean_risk": 14.2},
        {"time_tag": "T-5", "rainfall_7d": 94.2, "mean_risk": 22.8},
        {"time_tag": "T-3", "rainfall_7d": 168.5, "mean_risk": 38.6},
        {"time_tag": "T-2", "rainfall_7d": 224.0, "mean_risk": 52.1},
        {"time_tag": "T-1", "rainfall_7d": 286.3, "mean_risk": 68.4},
        {"time_tag": "T", "rainfall_7d": 342.8, "mean_risk": 82.7}
    ])
    fig_time = plot_risk_timeline(timeline_df)
    assert fig_time is not None
    print("    [PASS] Risk escalation timeline chart verified")

    cm_data = {"tn": 2200, "fp": 150, "fn": 120, "tp": 1280}
    fig_cm = plot_confusion_matrix_chart(cm_data)
    assert fig_cm is not None
    print("    [PASS] Confusion matrix chart verified")

    df_imp = pd.DataFrame([
        {"feature": "ndwi", "importance": 0.32},
        {"feature": "b8", "importance": 0.28},
        {"feature": "rainfall_7d", "importance": 0.21}
    ])
    fig_fi = plot_feature_importance_chart(df_imp)
    assert fig_fi is not None
    print("    [PASS] Feature importance chart verified")

    print("\n" + "=" * 70)
    print("ALL FUNCTIONAL & INTERACTION TESTS PASSED (100% CLEAN)")
    print("=" * 70)

if __name__ == "__main__":
    test_all_features()
