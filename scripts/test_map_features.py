#!/usr/bin/env python3
"""
Comprehensive Functional & Geospatial Feature Test Suite
Simulates all user interactions, map controls, 2D layer toggles, simulation stages,
and coordinate selection across the Golaghat & Nagaon study area.
"""

import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
from src.data_loader import load_data_for_tag, generate_base_topography, load_assam_boundary_and_rivers
from src.model import FloodRiskModel
from src.prediction import generate_spatial_prediction
from src.visualization import (
    compute_district_flood_simulation,
    compute_downhill_flow_paths,
    build_interactive_map,
    plot_risk_timeline,
    plot_feature_importance_chart,
    plot_confusion_matrix_chart
)

def test_all_features():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE 2D GIS MAP & SIMULATION FUNCTIONAL TESTS")
    print("=" * 70)

    # 1. Base data & AI Model
    print("\n[1] Testing AI model & local spatial prediction...")
    df_step = load_data_for_tag("T")
    assert not df_step.empty, "Dataframe T is empty"
    rf = FloodRiskModel("rf")
    rf.load()
    sp_local = generate_spatial_prediction(df_step, rf)
    assert sp_local["risk_grid"].shape == (50, 75), f"Unexpected shape {sp_local['risk_grid'].shape}"
    print("    [PASS] AI Model loaded and predictions generated successfully (3,750 cells)")

    # 2. Base Topography & Vector GIS Assets
    print("\n[2] Testing district study area topography & vector GIS assets...")
    topo_local = generate_base_topography(config.STUDY_AREA["grid_rows"], config.STUDY_AREA["grid_cols"])
    assert topo_local["elevation"].shape == (50, 75)
    assert "permanent_water" in topo_local
    geo_assets = load_assam_boundary_and_rivers()
    assert len(geo_assets["district_rings"]) > 0, "No district rings found"
    assert len(geo_assets["river_lines"]) > 0, "No river lines found"
    print(f"    [PASS] District DEM verified: {topo_local['elevation'].shape}")
    print(f"    [PASS] Vector district boundary rings loaded: {len(geo_assets['district_rings'])}")
    print(f"    [PASS] Vector river lines loaded: {len(geo_assets['river_lines'])}")

    # 3. 5-Stage District-Scale Flood Simulation with Advancing Front
    print("\n[3] Testing District-Scale (Golaghat & Nagaon) 5-stage flood simulation with Advancing Front...")
    dist_sim = compute_district_flood_simulation(topo_local, spatial_preds=sp_local, rainfall_7d=342.8)
    assert len(dist_sim["stages"]) == 5, "Expected exactly 5 district stages"
    for i, s in enumerate(dist_sim["stages"]):
        print(f"    Stage {i} ({s['tag']}): {s['name']} | Flooded: {s['flooded_cells']} cells ({s['flooded_area_km2']:.1f} km²) | New Front: +{s['newly_flooded_cells']} cells")
    assert dist_sim["stages"][4]["flooded_cells"] > dist_sim["stages"][0]["flooded_cells"], "District water extent did not expand"
    assert dist_sim["stages"][1]["newly_flooded_cells"] > 0, "Stage 1 advancing front must have newly flooded cells"
    print("    [PASS] District simulation computed with active advancing flood front (+newly inundated cells)")

    # 4. 2D Interactive GIS Map Construction across Simulation Stages
    print("\n[4] Testing 2D Folium Disaster Map across all 5 simulation stages...")
    for stage_idx in range(5):
        m = build_interactive_map(
            lat_grid=topo_local["lat_grid"],
            lon_grid=topo_local["lon_grid"],
            risk_grid=sp_local["risk_grid"],
            gt_grid=sp_local["gt_grid"],
            perm_water_grid=sp_local["perm_water_grid"],
            df_step=df_step,
            sim_data=dist_sim,
            active_stage_idx=stage_idx,
            show_risk=True,
            show_sim_floodwater=True,
            show_flood_front=(stage_idx > 0),
            show_water=True,
            show_gt=True,
            show_boundaries=True,
            show_contours=True,
            selected_location=(26.60, 93.35)
        )
        assert m is not None
        html = m.get_root().render()
        assert len(html) > 5000
        print(f"    [PASS] Stage {stage_idx} Map rendered ({len(html)} bytes HTML)")

    # 5. Layer Toggle Scenarios
    print("\n[5] Testing layer toggle combinations...")
    # Test only simulated water
    m_water_only = build_interactive_map(
        lat_grid=topo_local["lat_grid"],
        lon_grid=topo_local["lon_grid"],
        risk_grid=sp_local["risk_grid"],
        gt_grid=sp_local["gt_grid"],
        perm_water_grid=sp_local["perm_water_grid"],
        df_step=df_step,
        sim_data=dist_sim,
        active_stage_idx=4,
        show_risk=False,
        show_sim_floodwater=True,
        show_flood_front=True,
        show_water=False,
        show_gt=False,
        show_boundaries=False
    )
    assert m_water_only is not None
    print("    [PASS] Water-only layer combination rendered successfully")

    # Test environmental layer (e.g. elevation surface)
    m_env = build_interactive_map(
        lat_grid=topo_local["lat_grid"],
        lon_grid=topo_local["lon_grid"],
        risk_grid=sp_local["risk_grid"],
        gt_grid=sp_local["gt_grid"],
        perm_water_grid=sp_local["perm_water_grid"],
        df_step=df_step,
        active_layer_var="elevation",
        show_risk=True
    )
    assert m_env is not None
    print("    [PASS] Environmental elevation surface layer rendered successfully")

    # 6. Interactive Click Coordinate Resolution
    print("\n[6] Testing interactive click coordinate resolution...")
    click_lat, click_lon = 26.65, 93.35
    df_step_inspector = df_step.copy()
    df_step_inspector["risk_score"] = sp_local["risk_scores"]
    dists = (df_step_inspector["lat"] - click_lat)**2 + (df_step_inspector["lon"] - click_lon)**2
    closest_idx = dists.idxmin()
    matched_cell = df_step_inspector.loc[closest_idx]
    assert abs(matched_cell["lat"] - click_lat) < 0.05
    assert abs(matched_cell["lon"] - click_lon) < 0.05
    print(f"    [PASS] Click ({click_lat}, {click_lon}) resolved to Cell [{int(matched_cell['row'])}, {int(matched_cell['col'])}] ({matched_cell['lat']:.3f}°N, {matched_cell['lon']:.3f}°E)")

    # 7. Downhill Hydraulic Flow Path
    print("\n[7] Testing downhill hydraulic flow path tracing...")
    flow = compute_downhill_flow_paths(
        elev_grid=topo_local["elevation"],
        risk_grid=sp_local["risk_grid"],
        perm_water_grid=sp_local["perm_water_grid"],
        lat_grid=topo_local["lat_grid"],
        lon_grid=topo_local["lon_grid"],
        risk_threshold=20.0,
        selected_cell=(int(matched_cell["row"]), int(matched_cell["col"]))
    )
    assert "selected_path" in flow
    sp_path = flow["selected_path"]
    print(f"    [PASS] Hydraulic flow path traced: {sp_path['dist_km']:.2f} km, drop {sp_path['elev_drop']:.1f} m, destination: {sp_path['destination']}")

    # 8. Analytical Plotly Charts
    print("\n[8] Testing analytical charts...")
    timeline_records = []
    for step in list(config.HISTORICAL_EVENTS.values())[0]["timeline"]:
        t_df = load_data_for_tag(step["tag"])
        timeline_records.append({
            "time_tag": step["tag"],
            "date": step["date"],
            "mean_risk": 50.0,
            "high_risk_area_km2": 850.0,
            "rainfall_7d": float(t_df["rainfall_7d"].iloc[0])
        })
    df_tl = pd.DataFrame(timeline_records)
    fig_tl = plot_risk_timeline(df_tl)
    assert len(fig_tl.data) == 2, "Expected 2 traces (bars + line)"
    print("    [PASS] plot_risk_timeline rendered successfully")

    fig_fi = plot_feature_importance_chart(rf.get_feature_importance())
    assert len(fig_fi.data) == 1
    print("    [PASS] plot_feature_importance_chart rendered successfully")

    cm_dict = {"tp": 850, "fp": 60, "fn": 45, "tn": 2795}
    fig_cm = plot_confusion_matrix_chart(cm_dict)
    assert len(fig_cm.data) == 1
    print("    [PASS] plot_confusion_matrix_chart rendered successfully")

    print("\n" + "=" * 70)
    print("ALL 8 GEOSPATIAL & ANALYTICAL FEATURE SUITES PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_all_features()
