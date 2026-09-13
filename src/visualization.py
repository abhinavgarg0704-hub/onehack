"""
Geospatial & Analytical Visualization Engine
Renders interactive 2D Folium disaster-response maps, terrain-guided flood propagation
simulation layers, advancing flood front overlays, Plotly temporal escalation timelines,
and feature importance charts.
"""

import json
import os
from pathlib import Path
import folium
from folium import plugins
from folium.raster_layers import ImageOverlay
import branca.colormap as cm
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import config
from src.data_loader import load_assam_boundary_and_rivers

__all__ = [
    "load_assam_boundary_and_rivers",
    "compute_district_flood_simulation",
    "compute_downhill_flow_paths",
    "build_interactive_map",
    "plot_risk_timeline",
    "plot_feature_importance_chart",
    "plot_confusion_matrix_chart"
]

def compute_district_flood_simulation(
    topo: dict,
    spatial_preds: dict = None,
    rainfall_7d: float = 342.8,
    **kwargs
) -> dict:
    """
    Precomputes 5 realistic terrain-guided flood propagation stages across the
    Golaghat & Nagaon district study area (~2,640 km² at ~500m resolution, 3,750 cells).
    Simulates water progressively spilling from the braided Brahmaputra channel and
    inundating low-lying alluvial floodplains based on negative elevation gradients (-∇z),
    proximity to drainage channels, and AI model flood vulnerability scores.
    """
    elev = topo["elevation"]
    river = np.asarray(topo["permanent_water"], dtype=bool)
    dist_to_drainage = topo.get("dist_to_drainage", np.zeros_like(elev))
    lat_grid = topo["lat_grid"]
    lon_grid = topo["lon_grid"]
    rows, cols = elev.shape

    risk = spatial_preds["risk_grid"] if spatial_preds is not None and "risk_grid" in spatial_preds else np.full_like(elev, 50.0)
    gt = spatial_preds["gt_grid"] if spatial_preds is not None and "gt_grid" in spatial_preds else np.zeros_like(elev)

    gy, gx = np.gradient(elev)
    vy, vx = -gy, -gx

    # 5 Real Physical Event Progression Stages (Based on July 2020 Disaster)
    stages_meta = [
        {"stage": 0, "tag": "T-7", "date": "2020-07-07", "rain": 58.4, "alert": "NORMAL", "name": "Baseline Riverbed"},
        {"stage": 1, "tag": "T-5", "date": "2020-07-09", "rain": 94.2, "alert": "NORMAL", "name": "Early Rise & Lowland Seepage"},
        {"stage": 2, "tag": "T-2", "date": "2020-07-12", "rain": 224.0, "alert": "WATCH", "name": "Floodplain Expansion & Breach"},
        {"stage": 3, "tag": "T-1", "date": "2020-07-13", "rain": 286.3, "alert": "WARNING", "name": "Rapid Inundation Surge"},
        {"stage": 4, "tag": "T", "date": "2020-07-14", "rain": 342.8, "alert": "HIGH_RISK", "name": "Peak Catastrophic Crest"}
    ]

    # Compute overland flow streamlines (-∇z)
    all_streamlines = []
    for r in range(2, rows - 2, 4):
        for c in range(3, cols - 3, 5):
            if not river[r, c] and elev[r, c] < 120.0 and risk[r, c] >= 25.0:
                curr_r, curr_c = float(r), float(c)
                sx, sy, sz = [], [], []
                for _ in range(30):
                    ir, ic = int(round(curr_r)), int(round(curr_c))
                    if ir < 0 or ir >= rows or ic < 0 or ic >= cols:
                        break
                    sx.append(float(lon_grid[ir, ic]))
                    sy.append(float(lat_grid[ir, ic]))
                    sz.append(float(elev[ir, ic]))
                    if river[ir, ic]:
                        break
                    dr, dc = vy[ir, ic], vx[ir, ic]
                    mag = np.hypot(dr, dc)
                    if mag < 1e-4:
                        break
                    curr_r += (dr / mag) * 0.90
                    curr_c += (dc / mag) * 0.90
                if len(sx) >= 3:
                    all_streamlines.append({"lons": sx, "lats": sy, "zs": sz})

    stages = []
    m0 = river.copy()
    m1 = m0 | ((dist_to_drainage < 850.0) & (elev < 60.0) & (risk >= 25.0))
    m2 = m1 | ((dist_to_drainage < 2400.0) & (elev < 66.0) & (risk >= 40.0))
    m3 = m2 | ((elev < 71.0) & ((dist_to_drainage < 4500.0) | (risk >= 52.0)))
    m4 = m3 | (gt == 1) | ((risk >= 65.0) & (elev < 76.0))
    stage_masks = [m0.astype(bool), m1.astype(bool), m2.astype(bool), m3.astype(bool), m4.astype(bool)]

    cell_area_km2 = 2640.0 / (rows * cols)  # ~0.704 km2 per cell

    for s_idx, sm in enumerate(stages_meta):
        water_mask = stage_masks[s_idx].astype(bool)
        if s_idx == 0:
            newly_inundated = np.zeros_like(water_mask, dtype=bool)
        else:
            newly_inundated = (water_mask & ~stage_masks[s_idx - 1]).astype(bool)

        flooded_cells = int(np.sum(water_mask))
        flooded_area_km2 = float(flooded_cells * cell_area_km2)
        newly_flooded_cells = int(np.sum(newly_inundated))
        newly_flooded_km2 = float(newly_flooded_cells * cell_area_km2)
        high_risk_cells = int(np.sum(risk >= 70.0))
        max_risk = float(np.max(risk)) if len(risk) > 0 else 0.0

        active_streamlines = all_streamlines[:min(len(all_streamlines), (s_idx + 1) * 12)]

        stages.append({
            "stage_num": sm["stage"],
            "stage_idx": sm["stage"],
            "tag": sm["tag"],
            "date": sm["date"],
            "name": sm["name"],
            "alert": sm["alert"],
            "rainfall_7d": sm["rain"],
            "rain": sm["rain"],
            "water_mask": water_mask,
            "newly_inundated": newly_inundated,
            "flooded_cells": flooded_cells,
            "flooded_area_km2": flooded_area_km2,
            "newly_flooded_cells": newly_flooded_cells,
            "newly_flooded_km2": newly_flooded_km2,
            "high_risk_cells": high_risk_cells,
            "max_risk": max_risk,
            "streamlines": active_streamlines
        })

    return {
        "stages": stages,
        "total_streamlines": len(all_streamlines),
        "study_area": "Golaghat & Nagaon Districts (Kaziranga Floodplain Corridor), Assam",
        "bounds": {
            "min_lat": float(np.min(lat_grid)),
            "max_lat": float(np.max(lat_grid)),
            "min_lon": float(np.min(lon_grid)),
            "max_lon": float(np.max(lon_grid))
        }
    }

# Backward compatibility alias
def compute_assam_flood_simulation(topo, spatial_preds=None, rainfall_7d=342.8, **kwargs) -> dict:
    """Redirects to compute_district_flood_simulation."""
    return compute_district_flood_simulation(topo, spatial_preds=spatial_preds, rainfall_7d=rainfall_7d, **kwargs)

def generate_assam_topography(rows: int = 65, cols: int = 95, seed: int = 42, **kwargs) -> dict:
    """Redirects to base topography generator for district study area."""
    from src.data_loader import generate_base_topography
    return generate_base_topography(config.STUDY_AREA["grid_rows"], config.STUDY_AREA["grid_cols"], seed=seed)

def compute_downhill_flow_paths(
    elev_grid: np.ndarray,
    risk_grid: np.ndarray,
    perm_water_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    risk_threshold: float = 25.0,
    selected_cell: tuple = None,
    step_len: float = 0.85,
    max_steps: int = 35,
    **kwargs
) -> dict:
    """
    Computes terrain-guided overland flood propagation streamlines and single-cell downstream flow paths
    using Digital Elevation Model (SRTM) negative elevation gradients (-∇z).
    Water flows along steepest downhill trajectories from upland high-risk areas toward
    the Brahmaputra braided river channel or local alluvial depression sinks.
    """
    rows, cols = elev_grid.shape
    gy, gx = np.gradient(elev_grid)
    vy, vx = -gy, -gx

    streamlines = []
    for r in range(3, rows - 3, 3):
        for c in range(3, cols - 3, 4):
            if risk_grid[r, c] >= risk_threshold and not perm_water_grid[r, c]:
                curr_r, curr_c = float(r), float(c)
                sx, sy, sz = [], [], []
                for step in range(max_steps):
                    ir = int(round(curr_r))
                    ic = int(round(curr_c))
                    if ir < 0 or ir >= rows or ic < 0 or ic >= cols:
                        break
                    sx.append(float(lon_grid[ir, ic]))
                    sy.append(float(lat_grid[ir, ic]))
                    sz.append(float(elev_grid[ir, ic]))

                    if perm_water_grid[ir, ic] and step > 0:
                        break
                    
                    dr, dc = vy[ir, ic], vx[ir, ic]
                    mag = np.hypot(dr, dc)
                    if mag < 1e-4:
                        break
                    curr_r += (dr / mag) * step_len
                    curr_c += (dc / mag) * step_len

                if len(sx) >= 2:
                    streamlines.append({
                        "lons": sx,
                        "lats": sy,
                        "zs": sz,
                        "start_risk": float(risk_grid[r, c]),
                        "start_elev": float(elev_grid[r, c])
                    })

    selected_path = None
    if selected_cell is not None:
        sr, sc = int(selected_cell[0]), int(selected_cell[1])
        sr = max(0, min(sr, rows - 1))
        sc = max(0, min(sc, cols - 1))
        
        curr_r, curr_c = float(sr), float(sc)
        target_lons, target_lats, target_zs = [], [], []
        dist_km = 0.0
        z_start = float(elev_grid[sr, sc])
        z_end = z_start
        reached_river = False

        for step in range(max_steps * 2):
            ir = int(round(curr_r))
            ic = int(round(curr_c))
            if ir < 0 or ir >= rows or ic < 0 or ic >= cols:
                break
            
            c_lat = float(lat_grid[ir, ic])
            c_lon = float(lon_grid[ir, ic])
            c_z = float(elev_grid[ir, ic])
            z_end = c_z

            if len(target_lats) > 0:
                dlat_km = (c_lat - target_lats[-1]) * 111.0
                dlon_km = (c_lon - target_lons[-1]) * 111.0 * np.cos(np.radians(c_lat))
                dist_km += float(np.hypot(dlat_km, dlon_km))

            target_lats.append(c_lat)
            target_lons.append(c_lon)
            target_zs.append(c_z)

            if perm_water_grid[ir, ic]:
                reached_river = True
                break

            dr, dc = vy[ir, ic], vx[ir, ic]
            mag = np.hypot(dr, dc)
            if mag < 1e-4:
                break
            curr_r += (dr / mag) * step_len
            curr_c += (dc / mag) * step_len

        elev_drop = max(0.0, z_start - z_end)
        hydraulic_slope = (elev_drop / (dist_km * 1000.0) * 100.0) if dist_km > 0.01 else 0.0

        selected_path = {
            "lats": target_lats,
            "lons": target_lons,
            "zs": target_zs,
            "dist_km": dist_km,
            "elev_drop": elev_drop,
            "hydraulic_slope": hydraulic_slope,
            "reached_river": reached_river,
            "destination": "Brahmaputra Braided River Channel" if reached_river else "Local Alluvial Floodplain Sink"
        }

    return {
        "streamlines": streamlines,
        "selected_path": selected_path
    }

def build_interactive_map(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    risk_grid: np.ndarray,
    gt_grid: np.ndarray,
    perm_water_grid: np.ndarray,
    df_step: pd.DataFrame = None,
    sim_data: dict = None,
    active_stage_idx: int = 0,
    active_layer_var: str = "risk",
    show_risk: bool = True,
    show_sim_floodwater: bool = True,
    show_flood_front: bool = True,
    show_water: bool = True,
    show_gt: bool = True,
    show_boundaries: bool = True,
    show_contours: bool = True,
    show_inspector: bool = False,
    risk_threshold: float = 20.0,
    risk_opacity: float = 0.85,
    gt_opacity: float = 0.72,
    selected_location: tuple = None,
    **kwargs
) -> folium.Map:
    """
    Constructs a professional 2D interactive geospatial disaster-response GIS map centered on the
    district-scale study area (Golaghat & Nagaon / Kaziranga corridor).
    Features:
    - High-contrast operations basemaps (Esri Dark Canvas, Esri High-Res Satellite, OpenStreetMap)
    - Transparent AI flood-risk surface overlay with project color scheme
    - Permanent Brahmaputra braided channel mask (JRC Water)
    - District boundary lines (Golaghat, Nagaon, Biswanath, Sonitpur) and river centerlines
    - Simulated floodwater layer and glowing advancing flood front overlay
    - Observed satellite flood extent (DFO Event 4924)
    - Selected location marker and regional hydrological stations
    - Compatible with st_folium click event handling
    """
    center = [config.STUDY_AREA["center_lat"], config.STUDY_AREA["center_lon"]]

    # 1. Dark Canvas Basemap (Esri World Dark Gray Base - Clean, high contrast, zero API keys required)
    m = folium.Map(
        location=center,
        zoom_start=config.STUDY_AREA["default_zoom"],
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Dark Canvas (Disaster Operations)",
        control_scale=True
    )

    # 2. High-Resolution Satellite Imagery Basemap (Esri World Imagery)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite Imagery (Esri High-Res)",
        max_zoom=19
    ).add_to(m)

    # 3. Standard Topographic Basemap (OpenStreetMap)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street / Light (OpenStreetMap)",
        max_zoom=19
    ).add_to(m)

    rows, cols = risk_grid.shape
    bounds = [
        [float(np.min(lat_grid)), float(np.min(lon_grid))],
        [float(np.max(lat_grid)), float(np.max(lon_grid))]
    ]

    # 4. Study Domain Geographic Boundary Box
    folium.Rectangle(
        bounds=bounds,
        color="#38BDF8",
        weight=2.0,
        dash_array="6, 4",
        fill=False,
        name="Study Domain Boundary",
        tooltip=f"Study Domain: {config.STUDY_AREA['name']} (26.45°N–26.85°N, 93.05°E–93.65°E)"
    ).add_to(m)

    # 5. District Boundaries & Brahmaputra Fluvial Drainage Network
    if show_boundaries:
        try:
            geo_assets = load_assam_boundary_and_rivers()
            min_lat, max_lat = bounds[0][0] - 0.25, bounds[1][0] + 0.25
            min_lon, max_lon = bounds[0][1] - 0.25, bounds[1][1] + 0.25

            d_group = folium.FeatureGroup(name="District Boundaries", show=True)
            for ring in geo_assets.get("district_rings", []):
                in_bounds = any(min_lat <= pt[1] <= max_lat and min_lon <= pt[0] <= max_lon for pt in ring)
                if in_bounds:
                    folium.PolyLine(
                        locations=[[pt[1], pt[0]] for pt in ring],
                        color="#94A3B8",
                        weight=1.6,
                        dash_array="4, 4",
                        opacity=0.75,
                        tooltip="Assam District Boundary"
                    ).add_to(d_group)
            d_group.add_to(m)

            r_group = folium.FeatureGroup(name="River Network Centerlines", show=True)
            for line in geo_assets.get("river_lines", []):
                in_bounds = any(min_lat <= pt[1] <= max_lat and min_lon <= pt[0] <= max_lon for pt in line)
                if in_bounds:
                    folium.PolyLine(
                        locations=[[pt[1], pt[0]] for pt in line],
                        color="#0284C7",
                        weight=2.2,
                        opacity=0.85,
                        tooltip="Brahmaputra Fluvial Drainage Network"
                    ).add_to(r_group)
            r_group.add_to(m)
        except Exception:
            pass

    # 6. Continuous & Stepped Risk Surface or Environmental Feature Overlay
    if show_risk and risk_grid is not None:
        if active_layer_var == "risk" or df_step is None:
            rgba_img = np.zeros((rows, cols, 4), dtype=np.uint8)
            for r in range(rows):
                for c in range(cols):
                    score = risk_grid[r, c]
                    if score < risk_threshold:
                        continue
                    if score < 25.0:
                        rgba_img[r, c] = [16, 185, 129, int(255 * risk_opacity * 0.70)]
                    elif score < 50.0:
                        rgba_img[r, c] = [245, 158, 11, int(255 * risk_opacity * 0.85)]
                    elif score < 75.0:
                        rgba_img[r, c] = [249, 115, 22, int(255 * risk_opacity * 0.95)]
                    else:
                        rgba_img[r, c] = [239, 68, 68, int(255 * risk_opacity)]
            layer_title = "AI Flood Risk Surface (0–100%)"
        else:
            cmap_dict = {
                "elevation": ("terrain", "SRTM Elevation Surface (m)"),
                "ndwi": ("GnBu", "NDWI Optical Water Index"),
                "mndwi": ("PuBuGn", "MNDWI Moisture Index"),
                "rainfall_7d": ("Blues", "7-Day Cumulative Rain (mm)"),
                "slope": ("YlOrBr", "Topographic Slope (°)"),
                "dist_to_drainage": ("Blues_r", "Distance to Drainage (m)")
            }
            cmap_name, layer_title = cmap_dict.get(active_layer_var, ("viridis", f"{active_layer_var} Surface"))
            vals = df_step[active_layer_var].values.reshape((rows, cols))
            vmin = float(vals.min())
            vmax = float(vals.max())
            if vmin == vmax:
                vmax = vmin + 1.0
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            rgba_img = (mpl.colormaps[cmap_name](norm(vals)) * 255).astype(np.uint8)
            rgba_img[:, :, 3] = int(255 * risk_opacity)

        ImageOverlay(
            image=rgba_img,
            bounds=bounds,
            opacity=1.0,
            name=layer_title,
            interactive=True,
            cross_origin=False
        ).add_to(m)

    # 7. Iso-Probability Risk Contours (50% High Risk, 75% Very High Risk)
    if show_contours and risk_grid is not None:
        try:
            cs = plt.contour(lon_grid, lat_grid, risk_grid, levels=[50.0, 75.0])
            if len(cs.allsegs) >= 2:
                for seg in cs.allsegs[0]:
                    if len(seg) > 2:
                        coords = [[float(pt[1]), float(pt[0])] for pt in seg]
                        folium.PolyLine(
                            locations=coords,
                            color="#F97316",
                            weight=2.2,
                            dash_array="5, 5",
                            opacity=0.90,
                            tooltip="50% High-Risk Iso-Probability Contour",
                            name="50% High-Risk Contour"
                        ).add_to(m)
                for seg in cs.allsegs[1]:
                    if len(seg) > 2:
                        coords = [[float(pt[1]), float(pt[0])] for pt in seg]
                        folium.PolyLine(
                            locations=coords,
                            color="#EF4444",
                            weight=2.8,
                            dash_array="3, 3",
                            opacity=0.95,
                            tooltip="75% Very-High-Risk Iso-Probability Contour",
                            name="75% Very-High-Risk Contour"
                        ).add_to(m)
            plt.close()
        except Exception:
            pass

    # 8. Permanent Riverbed Mask Overlay (Brahmaputra braided mainstem)
    if show_water and perm_water_grid is not None:
        water_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        water_rgba[perm_water_grid == 1] = [2, 132, 199, 215]  # Sky-600 Deep Riverbed
        ImageOverlay(
            image=water_rgba,
            bounds=bounds,
            opacity=0.85,
            name="Permanent Riverbed (JRC Water)",
            interactive=True
        ).add_to(m)

    # 9. Dynamic Simulated Floodwater & Advancing Flood Front
    if sim_data is not None and "stages" in sim_data:
        s_idx = max(0, min(active_stage_idx, len(sim_data["stages"]) - 1))
        stage = sim_data["stages"][s_idx]
        water_mask = stage.get("water_mask", None)
        newly_inundated = stage.get("newly_inundated", None)

        if show_sim_floodwater and water_mask is not None:
            sim_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
            flood_only = water_mask & ~(perm_water_grid == 1)
            sim_rgba[flood_only] = [6, 182, 212, 190]
            ImageOverlay(
                image=sim_rgba,
                bounds=bounds,
                opacity=0.82,
                name=f"Simulated Floodwater ({stage.get('tag', 'T')}: {stage.get('name', '')})",
                interactive=True
            ).add_to(m)

        if show_flood_front and newly_inundated is not None and np.any(newly_inundated):
            front_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
            front_rgba[newly_inundated] = [0, 240, 255, 240]  # Electric Cyan #00F0FF (Glowing Surge Front)
            ImageOverlay(
                image=front_rgba,
                bounds=bounds,
                opacity=0.95,
                name="⚡ Advancing Flood Front (Surge Extent)",
                interactive=True
            ).add_to(m)

    # 10. Observed Historical Flood Extent Overlay (Ground Truth Reference)
    if show_gt and gt_grid is not None:
        gt_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        gt_rgba[gt_grid == 1] = [59, 130, 246, int(255 * gt_opacity)]  # Royal Blue
        ImageOverlay(
            image=gt_rgba,
            bounds=bounds,
            opacity=1.0,
            name="Observed Flood Extent (DFO 4924)",
            interactive=True
        ).add_to(m)

    # 11. Selected Location / Focal Point Target Marker
    if selected_location is not None:
        folium.CircleMarker(
            location=[selected_location[0], selected_location[1]],
            radius=8.5,
            color="#00F0FF",
            weight=3.0,
            fill=True,
            fill_color="#EF4444",
            fill_opacity=0.95,
            tooltip=f"Selected Target: {selected_location[0]:.3f}°N, {selected_location[1]:.3f}°E",
            popup=f"<div style='font-family: Inter, sans-serif; font-size: 11px; color: #0F172A;'><b>Selected Target Cell</b><br>{selected_location[0]:.4f}°N, {selected_location[1]:.4f}°E</div>"
        ).add_to(m)

    # 12. Key Ground Stations & Landmark Markers
    landmarks = [
        {"name": "Kaziranga National Park HQ (Kohora)", "lat": 26.585, "lon": 93.355, "type": "park", "desc": "UNESCO World Heritage flood-dependent ecosystem"},
        {"name": "Brahmaputra Silghat Hydrological Gauge", "lat": 26.780, "lon": 93.120, "type": "gauge", "desc": "CWC Primary Water Level Monitoring Station"},
        {"name": "Bokakhat Emergency Operations Center", "lat": 26.620, "lon": 93.590, "type": "hq", "desc": "Sub-divisional disaster evacuation command"},
        {"name": "Diphlu River Wetland Confluence", "lat": 26.680, "lon": 93.280, "type": "wetland", "desc": "Critical backwater inundation choke point"},
        {"name": "Numaligarh Riverine Reach", "lat": 26.600, "lon": 93.630, "type": "hq", "desc": "Dhansiri-Brahmaputra confluence sector"}
    ]

    for lm in landmarks:
        is_gauge = (lm["type"] == "gauge")
        folium.CircleMarker(
            location=[lm["lat"], lm["lon"]],
            radius=5.5,
            color="#FFFFFF",
            weight=1.5,
            fill=True,
            fill_color="#EF4444" if is_gauge else "#38BDF8",
            fill_opacity=0.95,
            popup=f"<div style='font-family: Inter, sans-serif; font-size: 12px; color: #0F172A; min-width: 180px;'><b>{lm['name']}</b><br><span style='color: #64748B;'>{lm['desc']}</span><br><b>Coordinates:</b> {lm['lat']:.3f}°N, {lm['lon']:.3f}°E</div>",
            tooltip=f"{lm['name']} ({'Gauge' if is_gauge else 'Station'})"
        ).add_to(m)

    # 13. Map Navigation Controls & Layer Switcher
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MousePosition(position="bottomleft", separator=" | ", empty_string="Coordinates: Lat, Lon").add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    return m

def plot_risk_timeline(timeline_df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Renders an interactive command-center Plotly chart showing Mean Floodplain Risk (%)
    and 7-day Cumulative Rainfall (mm) approaching and during the historical flood event.
    """
    fig = go.Figure()

    # 7-Day Rainfall Bars (secondary y-axis)
    fig.add_trace(go.Bar(
        x=timeline_df["time_tag"],
        y=timeline_df["rainfall_7d"],
        name="7-Day Antecedent Rainfall (mm)",
        marker=dict(
            color="rgba(56, 189, 248, 0.45)",
            line=dict(color="rgba(56, 189, 248, 0.85)", width=1.2)
        ),
        yaxis="y2"
    ))

    # Mean Floodplain Risk Line (primary y-axis)
    fig.add_trace(go.Scatter(
        x=timeline_df["time_tag"],
        y=timeline_df["mean_risk"],
        mode="lines+markers+text",
        name="Predicted Floodplain Risk (%)",
        line=dict(color="#EF4444", width=3.2),
        marker=dict(size=8, color="#EF4444", symbol="circle", line=dict(color="#FFFFFF", width=1.2)),
        text=[f"{v:.1f}%" for v in timeline_df["mean_risk"]],
        textposition="top center",
        textfont=dict(color="#F8FAFC", size=11, family="Inter, sans-serif"),
        yaxis="y1"
    ))

    # Operational Hazard Threshold Lines
    fig.add_hline(
        y=60, line_dash="dot", line_color="#F97316", line_width=1.5,
        annotation_text="WARNING (60%)", annotation_position="bottom right",
        annotation_font=dict(color="#F97316", size=10, family="Inter, sans-serif")
    )
    fig.add_hline(
        y=80, line_dash="dash", line_color="#EF4444", line_width=1.8,
        annotation_text="HIGH RISK (80%)", annotation_position="top right",
        annotation_font=dict(color="#EF4444", size=10, family="Inter, sans-serif")
    )

    fig.update_layout(
        title=dict(
            text="<b>TEMPORAL RISK PROGRESSION & ANTECEDENT RAINFALL</b>",
            font=dict(color="#F8FAFC", size=13, family="Inter, sans-serif")
        ),
        xaxis=dict(
            title=dict(text="Lead Time Window (Antecedent T-7 to Peak Inundation T)", font=dict(color="#94A3B8", size=11)),
            tickfont=dict(color="#CBD5E1", size=11, family="Inter, sans-serif"),
            gridcolor="#1E293B",
            showline=True,
            linecolor="#334155"
        ),
        yaxis=dict(
            title=dict(text="<b>Predicted Floodplain Risk (%)</b>", font=dict(color="#EF4444", size=11)),
            range=[0, 110],
            tickfont=dict(color="#EF4444", size=11, family="Inter, sans-serif"),
            gridcolor="#1E293B",
            showline=True,
            linecolor="#334155"
        ),
        yaxis2=dict(
            title=dict(text="<b>7-Day Rainfall (mm)</b>", font=dict(color="#38BDF8", size=11)),
            overlaying="y",
            side="right",
            range=[0, 420],
            tickfont=dict(color="#38BDF8", size=11, family="Inter, sans-serif"),
            showgrid=False
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B111E",
        legend=dict(
            x=0.02, y=0.96,
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#E2E8F0", size=11, family="Inter, sans-serif")
        ),
        margin=dict(l=50, r=50, t=55, b=45),
        height=390
    )

    return fig

def plot_feature_importance_chart(df_imp: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Renders an executive horizontal bar chart displaying ranked environmental drivers.
    """
    feature_labels = {
        "b8": "Near-Infrared Reflectance (B8)",
        "ndwi": "Normalized Difference Water Index (NDWI)",
        "b12": "Shortwave-Infrared 2 (B12)",
        "ndvi": "Vegetation Vigor Index (NDVI)",
        "dist_to_drainage": "Distance to River Channel (m)",
        "permanent_water": "Permanent Riverbed Mask (JRC)",
        "elevation": "Topographic Elevation (SRTM DEM)",
        "mndwi": "Modified NDWI (Suspended Silt Index)",
        "rainfall_7d": "7-Day Cumulative Rainfall (CHIRPS)",
        "rainfall_1d": "24h Instantaneous Rainfall (CHIRPS)",
        "rainfall_3d": "3-Day Cumulative Rainfall (CHIRPS)",
        "rainfall_14d": "14-Day Cumulative Rainfall (CHIRPS)",
        "slope": "Topographic Slope Gradient (deg)",
        "b3": "Green Band Surface Reflectance (B3)",
        "b4": "Red Band Surface Reflectance (B4)",
        "b11": "Shortwave-Infrared 1 (B11)"
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
        text=[f"{v:.3f}" for v in df_sorted["importance"]],
        textposition="outside",
        textfont=dict(color="#CBD5E1", size=10, family="Inter, sans-serif")
    ))

    fig.update_layout(
        title=dict(
            text="<b>ENVIRONMENTAL DRIVER IMPORTANCE (GINI CONTRIBUTIONS)</b>",
            font=dict(color="#F8FAFC", size=13, family="Inter, sans-serif")
        ),
        xaxis=dict(
            title=dict(text="Relative Gini Impurity Reduction", font=dict(color="#94A3B8", size=11)),
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
        margin=dict(l=10, r=40, t=50, b=40),
        height=430
    )

    return fig

def plot_confusion_matrix_chart(cm_dict: dict, **kwargs) -> go.Figure:
    """
    Renders an annotated confusion matrix with command-center aesthetic.
    """
    z = [[cm_dict["tn"], cm_dict["fp"]],
         [cm_dict["fn"], cm_dict["tp"]]]
    x = ["Predicted Non-Flood", "Predicted Flood"]
    y = ["Actual Non-Flood", "Actual Flood"]

    fig = px.imshow(
        z,
        x=x,
        y=y,
        color_continuous_scale=[[0, "#0F172A"], [0.5, "#1E3A8A"], [1.0, "#2563EB"]],
        text_auto=True,
        aspect="auto"
    )

    fig.update_layout(
        title=dict(
            text="<b>HOLDOUT VALIDATION CONFUSION MATRIX (PEAK EVENT T)</b>",
            font=dict(color="#F8FAFC", size=12, family="Inter, sans-serif")
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B111E",
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color="#CBD5E1", size=11, family="Inter, sans-serif")),
        yaxis=dict(tickfont=dict(color="#CBD5E1", size=11, family="Inter, sans-serif")),
        margin=dict(l=20, r=20, t=45, b=20),
        height=320
    )

    return fig
