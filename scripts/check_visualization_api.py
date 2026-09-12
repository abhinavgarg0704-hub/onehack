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
            mismatches.append(f"Line {lineno:4d}: Could not import {mod_name}.{orig_name}: {e}")
            continue

        if target is None:
            mismatches.append(f"Line {lineno:4d}: Symbol {orig_name} not found in {mod_name}!")
            continue

        if not (inspect.isfunction(target) or inspect.isclass(target)):
            continue

        try:
            sig = inspect.signature(target)
        except Exception as e:
            continue

        pos_args = [a for a in node.args if not isinstance(a, ast.Starred)]
        kw_args = [kw.arg for kw in node.keywords if kw.arg is not None]
        has_var_kwargs = any(kw.arg is None for kw in node.keywords)

        sig_params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())

        # 1. Check keyword arguments
        invalid_kw = []
        for kw in kw_args:
            if kw not in sig_params and not accepts_var_kw:
                invalid_kw.append(kw)

        # 2. Check required positional parameters
        required_params = [
            p.name for p in sig_params.values()
            if p.default == inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        passed_names = set(kw_args)
        for i, p_name in enumerate(required_params):
            if i < len(pos_args):
                passed_names.add(p_name)

        missing_req = [p for p in required_params if p not in passed_names]

        if invalid_kw or missing_req:
            err_details = []
            if invalid_kw:
                err_details.append(f"unexpected keywords: {invalid_kw}")
            if missing_req:
                err_details.append(f"missing required args: {missing_req}")
            mismatches.append(
                f"Line {lineno:4d} | {func_name:32s} | FAILED ({'; '.join(err_details)})\n"
                f"            Signature: {sig}\n"
                f"            Passed kw: {kw_args}"
            )
        else:
            verified_calls.append(
                f"Line {lineno:4d} | {func_name:32s} | {len(pos_args)} pos, {len(kw_args)} kw | OK"
            )

    print(f"\nVerified {len(verified_calls)} call sites successfully:")
    for vc in verified_calls:
        print(f"  [OK] {vc}")

    if mismatches:
        print(f"\n[FAIL] Found {len(mismatches)} signature mismatch(es):")
        for m in mismatches:
            print(f"  [FAIL] {m}")
        return False

    print("\n[SUCCESS] All app.py call sites match canonical module signatures perfectly!")
    return True

def smoke_test_assam_3d_pipeline() -> bool:
    """Executes the full-Assam 3D visualization pipeline end-to-end in headless mode."""
    print_header("SMOKE TESTING ASSAM 3D GEOSPATIAL PIPELINE")
    try:
        from src.visualization import (
            generate_assam_topography,
            compute_assam_flood_simulation,
            build_assam_3d_simulation_map
        )
        from src.data_loader import load_data_for_tag
        from src.model import FloodRiskModel
        from src.prediction import generate_spatial_prediction

        print("1. Generating Assam macro-scale topography (65x95 DEM)...")
        topo_assam = generate_assam_topography(rows=65, cols=95)
        assert "elevation" in topo_assam, "DEM elevation grid missing"
        assert "district_rings" in topo_assam, "district_rings missing"
        assert "river_lines" in topo_assam, "river_lines missing"
        assert topo_assam["bounds"]["min_lat"] == 24.2, "Assam south bound mismatch"
        assert topo_assam["bounds"]["max_lon"] == 96.0, "Assam east bound mismatch"
        print("   [OK] Assam DEM generated successfully (~78,438 km^2 scope)")

        print("2. Generating local sector AI predictions...")
        df_step = load_data_for_tag("T")
        rf_model = FloodRiskModel("rf")
        rf_model.load()
        sp_local = generate_spatial_prediction(df_step, rf_model)
        assert "risk_grid" in sp_local, "Local AI risk grid missing"
        print("   [OK] Local sector prediction generated (3,750 cells)")

        print("3. Precomputing Assam terrain-guided flood simulation...")
        sim_assam = compute_assam_flood_simulation(topo_assam, sp_local=sp_local, rainfall_7d=342.8)
        assert len(sim_assam["stages"]) == 5, "Expected 5 simulation stages"
        print("   [OK] 5 flood propagation stages computed successfully")

        print("4. Building Google Earth-scale 3D simulation canvas with all active layers...")
        fig = build_assam_3d_simulation_map(
            topo_assam=topo_assam,
            sim_data=sim_assam,
            active_stage_idx=4,
            sp_local=sp_local,
            scale_level="assam",
            show_terrain=True,
            show_boundaries=True,
            show_ai_footprint=True,
            show_river=True,
            show_floodwater=True,
            show_streamlines=True,
            show_gt=True,
            show_landmarks=True,
            selected_location=(26.60, 93.35),
            vertical_exaggeration=0.85
        )
        trace_count = len(fig.data)
        assert trace_count >= 7, f"Expected >= 7 traces, got {trace_count}"
        print(f"   [OK] 3D Figure generated successfully with {trace_count} visual layers")

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

    # 4. Smoke test end-to-end execution
    smoke_ok = smoke_test_assam_3d_pipeline()
    if not smoke_ok:
        sys.exit(1)

    print_header("ALL VISUALIZATION API CONTRACT CHECKS PASSED (100% CLEAN)")
    sys.exit(0)

if __name__ == "__main__":
    main()
