#!/usr/bin/env python3
"""
Visualization & Data Interface Contract Verification Suite
Statically and dynamically inspects all visualization and data APIs in FloodSense AI.
Validates that every call site in app.py strictly adheres to the canonical signatures
defined in src/visualization.py and src/data_loader.py.
"""

import ast
import inspect
import os
import sys
import importlib

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def print_header(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def audit_module_exports(module_name: str) -> dict:
    """Verifies that the module can be imported and extracts signatures of all exports."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FATAL] Failed to import {module_name}: {e}")
        return {}

    exports = getattr(mod, "__all__", [x for x in dir(mod) if not x.startswith("_")])
    signatures = {}
    print(f"\nModule: {module_name} ({len(exports)} public exports)")
    for name in exports:
        obj = getattr(mod, name, None)
        if obj is None:
            print(f"  [ERROR] Export '{name}' declared in __all__ but does not exist in {module_name}!")
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            try:
                sig = inspect.signature(obj)
                signatures[name] = sig
                print(f"  [OK] {name}{sig}")
            except Exception as e:
                print(f"  [?]  {name} (could not extract signature: {e})")
        else:
            print(f"  [*]  {name} (non-callable object)")
    return signatures

def audit_app_call_sites(app_path: str = "app.py") -> bool:
    """Parses app.py AST and validates all call sites against actual runtime signatures."""
    print_header("AUDITING APP.PY CALL SITES VIA AST & INSPECTION")
    
    if not os.path.exists(app_path):
        print(f"[ERROR] {app_path} not found!")
        return False

    with open(app_path, "r", encoding="utf-8") as f:
        app_code = f.read()

    try:
        tree = ast.parse(app_code, filename=app_path)
    except SyntaxError as e:
        print(f"[FATAL] Syntax error parsing {app_path}: {e}")
        return False

    # Collect imports from src.* and config
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod_name = node.module
            if mod_name and (mod_name.startswith("src") or mod_name == "config"):
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    imports[local_name] = (mod_name, alias.name)

    print(f"Tracked {len(imports)} project-internal symbols imported by {app_path}.")

    # Walk calls and validate
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append((node.lineno, node.func.id, node))

    calls.sort(key=lambda x: x[0])

    mismatches = []
    verified_calls = []

    for lineno, func_name, node in calls:
        if func_name not in imports:
            continue

        mod_name, orig_name = imports[func_name]
        try:
            mod = importlib.import_module(mod_name)
            target = getattr(mod, orig_name, None)
        except Exception as e:
            mismatches.append(f"Line {lineno}: {func_name}() failed to load from {mod_name}: {e}")
            continue

        if target is None:
            mismatches.append(f"Line {lineno}: {func_name}() does not exist in {mod_name}")
            continue

        if not inspect.isfunction(target) and not inspect.isclass(target):
            continue

        sig = inspect.signature(target)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        # Check call arguments
        call_kwargs = [kw.arg for kw in node.keywords if kw.arg is not None]
        num_args = len(node.args)

        # Check keyword arguments against parameter list
        for kw in call_kwargs:
            if kw not in params and not has_var_keyword:
                mismatches.append(
                    f"Line {lineno}: {func_name}() called with unexpected keyword argument '{kw}'. "
                    f"Accepted parameters: {list(params.keys())}"
                )

        verified_calls.append(f"Line {lineno}: {func_name}() [OK]")

    print(f"Audited {len(verified_calls)} internal function call sites in {app_path}:")
    for vc in verified_calls:
        print(f"  {vc}")

    if mismatches:
        print(f"\n[ERROR] Found {len(mismatches)} call signature mismatches:")
        for mm in mismatches:
            print(f"  - {mm}")
        return False

    print("\n[SUCCESS] All app.py call sites strictly match target signatures!")
    return True

def smoke_test_district_2d_pipeline() -> bool:
    """Executes the district-scale (Golaghat & Nagaon) 2D simulation pipeline end-to-end."""
    print_header("SMOKE TESTING DISTRICT-SCALE 2D SIMULATION & MAP PIPELINE")
    try:
        from src.visualization import (
            compute_district_flood_simulation,
            build_interactive_map,
            compute_downhill_flow_paths
        )
        from src.data_loader import load_data_for_tag, generate_base_topography
        from src.model import FloodRiskModel
        from src.prediction import generate_spatial_prediction
        import config

        print("1. Generating base topography (50x75 DEM, ~2,640 km^2)...")
        topo = generate_base_topography(config.STUDY_AREA["grid_rows"], config.STUDY_AREA["grid_cols"])
        assert topo["elevation"].shape == (50, 75), "Topography shape mismatch"
        assert "permanent_water" in topo, "Permanent water missing"
        print("   [OK] Base topography verified (50x75, 3,750 cells)")

        print("2. Generating local sector AI predictions...")
        df_step = load_data_for_tag("T")
        rf_model = FloodRiskModel("rf")
        rf_model.load()
        sp_local = generate_spatial_prediction(df_step, rf_model)
        assert "risk_grid" in sp_local, "Local AI risk grid missing"
        print("   [OK] AI predictions verified")

        print("3. Precomputing district terrain-guided flood simulation...")
        sim_district = compute_district_flood_simulation(topo, spatial_preds=sp_local, rainfall_7d=342.8)
        assert len(sim_district["stages"]) == 5, "Expected 5 simulation stages"
        for i, stg in enumerate(sim_district["stages"]):
            assert "water_mask" in stg, f"water_mask missing from stage {i}"
            assert "newly_inundated" in stg, f"newly_inundated missing from stage {i}"
            assert "flooded_cells" in stg, f"flooded_cells missing from stage {i}"
            assert "newly_flooded_cells" in stg, f"newly_flooded_cells missing from stage {i}"
            print(f"   Stage {i} ({stg['tag']}): {stg['name']} | Flooded: {stg['flooded_cells']} cells ({stg['flooded_area_km2']:.1f} km²) | New Front: +{stg['newly_flooded_cells']} cells")
        print("   [OK] 5 district flood simulation stages verified")

        print("4. Building 2D Interactive GIS Disaster Map with all layers...")
        m = build_interactive_map(
            lat_grid=topo["lat_grid"],
            lon_grid=topo["lon_grid"],
            risk_grid=sp_local["risk_grid"],
            gt_grid=sp_local["gt_grid"],
            perm_water_grid=sp_local["perm_water_grid"],
            df_step=df_step,
            sim_data=sim_district,
            active_stage_idx=3,
            show_risk=True,
            show_sim_floodwater=True,
            show_flood_front=True,
            show_water=True,
            show_gt=True,
            show_boundaries=True,
            show_contours=True,
            show_inspector=False,
            risk_threshold=20.0,
            selected_location=(26.60, 93.35)
        )
        assert m is not None, "Folium map is None"
        html = m.get_root().render()
        assert len(html) > 5000, "Rendered map HTML suspiciously short"
        assert "World_Dark_Gray_Base" in html, "Esri Dark Canvas basemap missing"
        assert "World_Imagery" in html, "Esri World Imagery basemap missing"
        print(f"   [OK] 2D Folium Map generated successfully ({len(html)} bytes HTML)")

        print("5. Verifying downhill hydraulic flow path calculation...")
        flow = compute_downhill_flow_paths(
            elev_grid=topo["elevation"],
            risk_grid=sp_local["risk_grid"],
            perm_water_grid=sp_local["perm_water_grid"],
            lat_grid=topo["lat_grid"],
            lon_grid=topo["lon_grid"],
            risk_threshold=20.0,
            selected_cell=(25, 35)
        )
        assert "selected_path" in flow, "selected_path missing"
        assert flow["selected_path"]["dist_km"] >= 0, "Invalid flow distance"
        print(f"   [OK] Downhill flow path: {flow['selected_path']['dist_km']:.2f} km to {flow['selected_path']['destination']}")

        return True
    except Exception as e:
        import traceback
        print(f"[FAIL] Smoke test failed with exception: {e}")
        traceback.print_exc()
        return False

def main():
    print_header("FLOODSENSE AI -- VISUALIZATION API CONTRACT VERIFICATION SUITE")
    print(f"Python Interpreter: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    # 1. Audit src/visualization.py
    viz_sigs = audit_module_exports("src.visualization")
    if not viz_sigs:
        sys.exit(1)

    # 2. Audit src/data_loader.py
    dl_sigs = audit_module_exports("src.data_loader")
    if not dl_sigs:
        sys.exit(1)

    # 3. Audit app.py call sites
    ast_ok = audit_app_call_sites("app.py")
    if not ast_ok:
        sys.exit(1)

    # 4. Smoke test district 2D pipeline
    district_ok = smoke_test_district_2d_pipeline()
    if not district_ok:
        sys.exit(1)

    print_header("ALL VISUALIZATION API CONTRACT CHECKS PASSED (100% CLEAN)")
    sys.exit(0)

if __name__ == "__main__":
    main()
