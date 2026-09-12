"""
Geospatial & Analytical Visualization Engine
Renders interactive Folium risk heatmaps, Plotly temporal escalation timelines,
and feature importance charts.
"""

import folium
from folium import plugins
import branca.colormap as cm
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import config

__all__ = [
    "build_hybrid_flood_simulation_map",
    "compute_downhill_flow_paths",
    "build_interactive_map",
    "build_3d_terrain_view",
    "plot_risk_timeline",
    "plot_feature_importance_chart",
    "plot_confusion_matrix_chart"
]

def compute_downhill_flow_paths(
    elev_grid: np.ndarray,
    risk_grid: np.ndarray,
    perm_water_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    risk_threshold: float = 25.0,
    selected_cell: tuple = None,
    step_len: float = 0.85,
    max_steps: int = 35
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
    # Regular seed sampling across flood-prone alluvial terrain
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
            target_lons.append(float(lon_grid[ir, ic]))
            target_lats.append(float(lat_grid[ir, ic]))
            target_zs.append(float(elev_grid[ir, ic]))
            z_end = float(elev_grid[ir, ic])

            if step > 0:
                d_lat = (target_lats[-1] - target_lats[-2]) * 111.0
                d_lon = (target_lons[-1] - target_lons[-2]) * 111.0 * np.cos(np.radians(target_lats[-1]))
                dist_km += np.hypot(d_lat, d_lon)

            if perm_water_grid[ir, ic] and step > 0:
                reached_river = True
                break

            dr, dc = vy[ir, ic], vx[ir, ic]
            mag = np.hypot(dr, dc)
            if mag < 1e-4:
                break
            curr_r += (dr / mag) * step_len
            curr_c += (dc / mag) * step_len

        elev_drop = max(0.0, z_start - z_end)
        hydraulic_slope = (elev_drop / (dist_km * 1000.0)) * 100.0 if dist_km > 0.05 else 0.0
        est_travel_hours = (dist_km * 1000.0 / 0.4) / 3600.0 if dist_km > 0 else 0.0

        selected_path = {
            "start_r": sr,
            "start_c": sc,
            "start_lat": float(lat_grid[sr, sc]),
            "start_lon": float(lon_grid[sr, sc]),
            "lons": target_lons,
            "lats": target_lats,
            "zs": target_zs,
            "dist_km": dist_km,
            "elev_drop": elev_drop,
            "hydraulic_slope": hydraulic_slope,
            "travel_hours": est_travel_hours,
            "destination": "Brahmaputra Mainstem Braided Channel" if reached_river else "Alluvial Lowland Depression",
            "reached_river": reached_river
        }

    return {
        "streamlines": streamlines,
        "selected_path": selected_path
    }

def build_hybrid_flood_simulation_map(
    elev_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    risk_grid: np.ndarray,
    perm_water_grid: np.ndarray,
    gt_grid: np.ndarray = None,
    df_step: pd.DataFrame = None,
    active_layer_var: str = "risk",
    show_risk: bool = True,
    show_streamlines: bool = True,
    show_water: bool = True,
    show_gt: bool = True,
    show_selected_path: bool = True,
    vertical_exaggeration: float = 2.5,
    camera_preset: str = "perspective",
    selected_cell: tuple = (18, 35),
    risk_threshold: float = 20.0,
    surface_opacity: float = 0.95,
    **kwargs
) -> go.Figure:
    """
    Renders the unified HYBRID 2D + 3D geospatial flood-intelligence and terrain-guided
    propagation simulation map, integrating:
    1. Foundational 3D SRTM Digital Elevation Model
    2. Overlaid AI-Predicted Flood Risk Inundation Surface
    3. Permanent Brahmaputra Braided River Channel Trace (JRC Water)
    4. Observed Historical Inundation Extent (DFO Event 4924 Ground Truth)
    5. Terrain-Guided Downhill Flood Propagation Streamlines (-∇z Overland Flow)
    6. Interactive Target Cell Highlight & Downstream Drainage Path
    7. Regional Hydrological Landmarks & Gauging Stations
    """
    lats = lat_grid[:, 0]
    lons = lon_grid[0, :]
    z_terrain = elev_grid * vertical_exaggeration

    # 1. Surface Texture / Colormap Setup
    if active_layer_var == "risk" or df_step is None:
        if show_risk:
            surface_data = np.copy(risk_grid)
            surface_title = "AI Flood Risk (%)"
            colorscale = [
                [0.0, "#10B981"],
                [0.25, "#10B981"],
                [0.25, "#F59E0B"],
                [0.50, "#F59E0B"],
                [0.50, "#F97316"],
                [0.75, "#F97316"],
                [0.75, "#EF4444"],
                [1.0, "#EF4444"]
            ]
            cmin, cmax = 0.0, 100.0
        else:
            surface_data = elev_grid
            surface_title = "SRTM Elevation (m)"
            colorscale = "Viridis"
            cmin = float(np.min(elev_grid))
            cmax = float(np.max(elev_grid))
    else:
        # Environmental feature layer
        cmap_mapping = {
            "elevation": ("Viridis", "SRTM Elevation (m)"),
            "ndwi": ("GnBu", "NDWI Water Index"),
            "mndwi": ("PuBuGn", "MNDWI Moisture Index"),
            "rainfall_7d": ("Blues", "7-Day Cumulative Rain (mm)"),
            "slope": ("YlOrBr", "Topographic Slope (°)"),
            "dist_to_drainage": ("Blues_r", "Distance to River (m)")
        }
        colorscale, surface_title = cmap_mapping.get(active_layer_var, ("Viridis", f"{active_layer_var}"))
        rows, cols = elev_grid.shape
        vals = df_step[active_layer_var].values.reshape((rows, cols))
        surface_data = vals
        cmin = float(np.min(vals))
        cmax = float(np.max(vals))
        if cmin == cmax:
            cmax = cmin + 1.0

    # Format custom hoverdata
    rows, cols = elev_grid.shape
    hover_text = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            gt_status = "Inundated (DFO 4924)" if (gt_grid is not None and gt_grid[r, c] == 1) else "Dry Ground"
            water_status = "Riverbed Channel" if perm_water_grid[r, c] else "Floodplain Lowland"
            hover_text[r, c] = (
                f"<b>ASSAM BRAHMAPUTRA BASIN</b><br>"
                f"<b>Coords:</b> {lat_grid[r, c]:.3f}°N, {lon_grid[r, c]:.3f}°E<br>"
                f"<b>SRTM Elevation:</b> {elev_grid[r, c]:.1f} m<br>"
                f"<b>AI Flood Risk:</b> {risk_grid[r, c]:.1f}%<br>"
                f"<b>Land Cover:</b> {water_status}<br>"
                f"<b>Observed DFO 4924:</b> {gt_status}"
            )

    fig = go.Figure()

    # TRACE 1: 3D SRTM Terrain Mesh with Projected AI Risk Overlay
    fig.add_trace(go.Surface(
        x=lons,
        y=lats,
        z=z_terrain,
        surfacecolor=surface_data,
        cmin=cmin,
        cmax=cmax,
        colorscale=colorscale,
        opacity=surface_opacity,
        colorbar=dict(
            title=dict(text=f"<b>{surface_title}</b>", font=dict(color="#F8FAFC", size=11, family="Inter, sans-serif")),
            tickfont=dict(color="#CBD5E1", size=10, family="Inter, sans-serif"),
            len=0.72,
            thickness=14,
            x=1.02
        ),
        text=hover_text,
        hoverinfo="text",
        contours=dict(
            z=dict(show=True, usecolormap=False, highlightcolor="#38BDF8", project=dict(z=False), width=1)
        ),
        lighting=dict(ambient=0.68, diffuse=0.82, roughness=0.45, specular=0.25),
        name="Topographic Terrain"
    ))

    # TRACE 2: Permanent Brahmaputra Braided River Channel Mask (JRC Water)
    if show_water and np.any(perm_water_grid):
        w_mask = perm_water_grid.astype(bool)
        w_lons = lon_grid[w_mask]
        w_lats = lat_grid[w_mask]
        w_elev = elev_grid[w_mask] * vertical_exaggeration + 0.35

        fig.add_trace(go.Scatter3d(
            x=w_lons,
            y=w_lats,
            z=w_elev,
            mode="markers",
            marker=dict(
                color="#0284C7",
                size=3.8,
                opacity=0.85,
                symbol="circle"
            ),
            name="Permanent Riverbed (JRC Water)",
            hoverinfo="text",
            hovertext=[f"<b>Brahmaputra River Mainstem</b><br>Permanent Braided Channel<br>Elevation: {z/vertical_exaggeration:.1f} m" for z in w_elev]
        ))

    # TRACE 3: Observed Historical Flood Extent (DFO Event 4924 Ground Truth)
    if show_gt and gt_grid is not None:
        gt_inundated = (gt_grid == 1) & (~perm_water_grid.astype(bool))
        if np.any(gt_inundated):
            gt_lons = lon_grid[gt_inundated]
            gt_lats = lat_grid[gt_inundated]
            gt_elev = elev_grid[gt_inundated] * vertical_exaggeration + 0.55

            fig.add_trace(go.Scatter3d(
                x=gt_lons,
                y=gt_lats,
                z=gt_elev,
                mode="markers",
                marker=dict(
                    color="#06B6D4",
                    size=3.6,
                    opacity=0.78,
                    symbol="diamond"
                ),
                name="Observed Inundation (DFO 4924)",
                hoverinfo="text",
                hovertext=[f"<b>Observed Flood Inundation</b><br>DFO Event 4924 Ground Truth<br>Elevation: {z/vertical_exaggeration:.1f} m" for z in gt_elev]
            ))

    # TRACE 4: Terrain-Guided Flood Flow Streamlines (-∇z Overland Simulation)
    flow_data = compute_downhill_flow_paths(
        elev_grid=elev_grid,
        risk_grid=risk_grid,
        perm_water_grid=perm_water_grid,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        risk_threshold=max(30.0, float(risk_threshold)),
        selected_cell=selected_cell
    )

    if show_streamlines and len(flow_data["streamlines"]) > 0:
        streamline_x = []
        streamline_y = []
        streamline_z = []
        arrow_x = []
        arrow_y = []
        arrow_z = []

        for st_item in flow_data["streamlines"]:
            lx = st_item["lons"]
            ly = st_item["lats"]
            lz = [z * vertical_exaggeration + 0.85 for z in st_item["zs"]]
            streamline_x.extend(lx + [None])
            streamline_y.extend(ly + [None])
            streamline_z.extend(lz + [None])

            if len(lx) >= 3:
                mid_idx = len(lx) // 2
                arrow_x.append(lx[mid_idx])
                arrow_y.append(ly[mid_idx])
                arrow_z.append(lz[mid_idx] + 0.2)

        fig.add_trace(go.Scatter3d(
            x=streamline_x,
            y=streamline_y,
            z=streamline_z,
            mode="lines",
            line=dict(color="#38BDF8", width=3.2),
            name="Predicted Water Flow (-∇z)",
            hoverinfo="text",
            hovertext="<b>Terrain-Guided Overland Flow Streamline</b><br>Trajectory: Downhill toward Brahmaputra<br>Driving Gradient: -∇z Slope"
        ))

        if arrow_x:
            fig.add_trace(go.Scatter3d(
                x=arrow_x,
                y=arrow_y,
                z=arrow_z,
                mode="markers",
                marker=dict(color="#7DD3FC", size=3.0, symbol="circle"),
                name="Downhill Flow Direction",
                hoverinfo="skip"
            ))

    # TRACE 5: Selected Target Cell & Downstream Flow Trajectory
    if show_selected_path and flow_data["selected_path"] is not None:
        sel = flow_data["selected_path"]
        sel_x = sel["lons"]
        sel_y = sel["lats"]
        sel_z = [z * vertical_exaggeration + 1.3 for z in sel["zs"]]

        if len(sel_x) >= 2:
            fig.add_trace(go.Scatter3d(
                x=sel_x,
                y=sel_y,
                z=sel_z,
                mode="lines",
                line=dict(color="#FBBF24", width=7.0),
                name=f"Target Flow Path ({sel['dist_km']:.2f} km to River)",
                hoverinfo="text",
                hovertext=f"<b>Selected Cell Drainage Path</b><br>Descent: {sel['elev_drop']:.1f} m<br>Distance: {sel['dist_km']:.2f} km<br>Slope: {sel['hydraulic_slope']:.2f}%"
            ))

        # Start Beacon
        fig.add_trace(go.Scatter3d(
            x=[sel["start_lon"]],
            y=[sel["start_lat"]],
            z=[sel["zs"][0] * vertical_exaggeration + 3.0],
            mode="markers+text",
            marker=dict(color="#EF4444", size=10, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
            text=[f"TARGET [{sel['start_r']},{sel['start_c']}]"],
            textposition="top center",
            textfont=dict(color="#F8FAFC", size=11, family="Inter, sans-serif"),
            name="Selected Target Cell",
            hoverinfo="text",
            hovertext=f"<b>Target Focal Cell [{sel['start_r']},{sel['start_c']}]</b><br>Elev: {sel['zs'][0]:.1f}m<br>Risk: {risk_grid[sel['start_r'], sel['start_c']]:.1f}%"
        ))

        # River Drainage Outfall Marker
        if len(sel_x) >= 2:
            fig.add_trace(go.Scatter3d(
                x=[sel_x[-1]],
                y=[sel_y[-1]],
                z=[sel_z[-1] + 1.0],
                mode="markers",
                marker=dict(color="#0284C7", size=8, symbol="square", line=dict(color="#38BDF8", width=2)),
                name="River Drainage Outfall",
                hoverinfo="text",
                hovertext=f"<b>Riverbed Entry Outfall</b><br>Target: {sel['destination']}<br>Terminal Elevation: {sel['zs'][-1]:.1f}m"
            ))

    # TRACE 6: Hydrological Stations & Regional Landmarks
    landmarks = [
        {"name": "Kaziranga Central Corridor", "lat": 26.60, "lon": 93.35, "desc": "Alluvial Floodplain Wildlife Sanctuary"},
        {"name": "Tezpur Hydrological Station", "lat": 26.62, "lon": 92.80, "desc": "Upstream River Gauging Benchmark"},
        {"name": "Silghat Braided Convergence", "lat": 26.61, "lon": 92.93, "desc": "Constricted Channel Bottleneck"},
        {"name": "Kaliabor Uplands", "lat": 26.55, "lon": 93.18, "desc": "Flood-Free Southern Foothills"},
        {"name": "Numaligarh River Inflow", "lat": 26.60, "lon": 93.60, "desc": "Dhansiri River Confluence"}
    ]
    lm_lats = []
    lm_lons = []
    lm_elevs = []
    lm_labels = []

    for lm in landmarks:
        if (lat_grid.min() <= lm["lat"] <= lat_grid.max()) and (lon_grid.min() <= lm["lon"] <= lon_grid.max()):
            r_idx = int(round((lm["lat"] - lat_grid.min()) / (lat_grid.max() - lat_grid.min()) * (rows - 1)))
            c_idx = int(round((lm["lon"] - lon_grid.min()) / (lon_grid.max() - lon_grid.min()) * (cols - 1)))
            r_idx = max(0, min(r_idx, rows - 1))
            c_idx = max(0, min(c_idx, cols - 1))
            lm_lats.append(lm["lat"])
            lm_lons.append(lm["lon"])
            lm_elevs.append(elev_grid[r_idx, c_idx] * vertical_exaggeration + 3.2)
            lm_labels.append(f"{lm['name']}")

    if lm_lats:
        fig.add_trace(go.Scatter3d(
            x=lm_lons,
            y=lm_lats,
            z=lm_elevs,
            mode="markers+text",
            marker=dict(color="#38BDF8", size=5, symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
            text=lm_labels,
            textposition="top right",
            textfont=dict(color="#CBD5E1", size=9, family="Inter, sans-serif"),
            name="Hydrological Landmarks",
            hoverinfo="text",
            hovertext=[f"<b>{lm_labels[i]}</b>" for i in range(len(lm_labels))]
        ))

    # Camera Preset Configuration
    camera_dict = {
        "perspective": dict(
            eye=dict(x=-1.45, y=-1.55, z=1.15),
            up=dict(x=0, y=0, z=1)
        ),
        "topdown": dict(
            eye=dict(x=0.001, y=0.001, z=2.40),
            up=dict(x=0, y=1, z=0)
        ),
        "corridor": dict(
            eye=dict(x=-1.80, y=-0.35, z=0.70),
            up=dict(x=0, y=0, z=1)
        ),
        "cross_section": dict(
            eye=dict(x=0.10, y=-2.15, z=0.55),
            up=dict(x=0, y=0, z=1)
        )
    }
    chosen_camera = camera_dict.get(camera_preset, camera_dict["perspective"])

    fig.update_layout(
        title=dict(
            text="<b>HYBRID GEOSPATIAL FLOOD INTELLIGENCE & TERRAIN-GUIDED FLOW PROPAGATION SIMULATION</b>",
            font=dict(color="#F8FAFC", size=13, family="Inter, sans-serif")
        ),
        scene=dict(
            xaxis=dict(title="Longitude (°E)", backgroundcolor="#0B111E", gridcolor="#1E293B", showbackground=True, color="#94A3B8"),
            yaxis=dict(title="Latitude (°N)", backgroundcolor="#0B111E", gridcolor="#1E293B", showbackground=True, color="#94A3B8"),
            zaxis=dict(title="Elevation (m)", backgroundcolor="#0B111E", gridcolor="#1E293B", showbackground=True, color="#94A3B8"),
            camera=chosen_camera,
            aspectratio=dict(x=1.5, y=1.2, z=0.42)
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        margin=dict(l=10, r=10, t=40, b=20),
        height=680,
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(15, 23, 42, 0.88)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#E2E8F0", size=10, family="Inter, sans-serif")
        )
    )

    return fig

def build_3d_terrain_view(
    elev_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    surface_data: np.ndarray = None,
    surface_name: str = "Flood Risk (%)",
    vertical_exaggeration: float = 2.5,
    **kwargs
) -> go.Figure:
    """
    Backward-compatible wrapper for 3D terrain visualization.
    Forwards to build_hybrid_flood_simulation_map when full simulation parameters are provided.
    """
    if "risk_grid" in kwargs or "perm_water_grid" in kwargs:
        return build_hybrid_flood_simulation_map(
            elev_grid=elev_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            vertical_exaggeration=vertical_exaggeration,
            **kwargs
        )
    lats = lat_grid[:, 0]
    lons = lon_grid[0, :]
    z_terrain = elev_grid * vertical_exaggeration

    if surface_data is None:
        surface_data = elev_grid
        surface_name = "Elevation (m)"
        colorscale = "Viridis"
        cmin = float(np.min(elev_grid))
        cmax = float(np.max(elev_grid))
    else:
        colorscale = [
            [0.0, "#10B981"],
            [0.25, "#10B981"],
            [0.25, "#F59E0B"],
            [0.50, "#F59E0B"],
            [0.50, "#F97316"],
            [0.75, "#F97316"],
            [0.75, "#EF4444"],
            [1.0, "#EF4444"]
        ]
        cmin = 0.0
        cmax = 100.0

    fig = go.Figure(data=[
        go.Surface(
            x=lons,
            y=lats,
            z=z_terrain,
            surfacecolor=surface_data,
            cmin=cmin,
            cmax=cmax,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text=f"<b>{surface_name}</b>", font=dict(color="#F8FAFC", size=11, family="Inter, sans-serif")),
                tickfont=dict(color="#CBD5E1", size=10, family="Inter, sans-serif"),
                len=0.75,
                thickness=16
            ),
            hovertemplate=(
                "<b>Longitude:</b> %{x:.3f}°E<br>" +
                "<b>Latitude:</b> %{y:.3f}°N<br>" +
                "<b>SRTM Elevation:</b> %{customdata:.1f} m<br>" +
                "<b>" + surface_name + ":</b> %{surfacecolor:.1f}%<extra></extra>"
            ),
            customdata=elev_grid,
            lighting=dict(ambient=0.65, diffuse=0.8, roughness=0.5, specular=0.25)
        )
    ])

    fig.update_layout(
        title=dict(
            text="<b>3D TOPOGRAPHIC DIGITAL ELEVATION MODEL (SRTM) & INUNDATION PROFILE</b>",
            font=dict(color="#F8FAFC", size=13, family="Inter, sans-serif")
        ),
        scene=dict(
            xaxis=dict(title="Longitude (°E)", backgroundcolor="#0B111E", gridcolor="#1E293B", showbackground=True, color="#94A3B8"),
            yaxis=dict(title="Latitude (°N)", backgroundcolor="#0B111E", gridcolor="#1E293B", showbackground=True, color="#94A3B8"),
            zaxis=dict(title="Elevation (m)", backgroundcolor="#0B111E", gridcolor="#1E293B", showbackground=True, color="#94A3B8"),
            camera=dict(
                eye=dict(x=-1.5, y=-1.6, z=1.2)
            ),
            aspectratio=dict(x=1.5, y=1.2, z=0.45)
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        margin=dict(l=10, r=10, t=40, b=20),
        height=640
    )

    return fig

def build_interactive_map(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    risk_grid: np.ndarray,
    gt_grid: np.ndarray,
    perm_water_grid: np.ndarray,
    df_step: pd.DataFrame = None,
    active_layer_var: str = "risk",
    show_risk: bool = True,
    show_gt: bool = True,
    show_water: bool = True,
    show_contours: bool = True,
    show_inspector: bool = True,
    risk_threshold: float = 20.0,
    risk_opacity: float = 0.85,
    gt_opacity: float = 0.72,
    **kwargs
) -> folium.Map:
    """
    Constructs an interactive satellite intelligence Folium map featuring:
    - Multi-basemap selector (Carto Dark Command, Esri High-Res Satellite, Standard Topo)
    - Semantic continuous/stepped risk surface overlay (0-25% Low, 25-50% Mod, 50-75% High, 75-100% Very High)
    - Environmental layer switching (AI Risk, SRTM Elevation, NDWI, MNDWI, Rainfall, Slope, River Proximity)
    - 50% and 75% iso-probability risk contour boundaries
    - Distinct observed historical flood extent overlay (ground truth)
    - Permanent Brahmaputra braided river channel mask (JRC surface water)
    - Interactive 3,750-cell GeoJSON risk inspector with hover tooltips and click popups from real data
    - Regional hydrological stations, landmarks, study boundary, scale bar, and coordinate tracker
    """
    from folium.raster_layers import ImageOverlay

    center = [config.STUDY_AREA["center_lat"], config.STUDY_AREA["center_lon"]]
    
    # 1. High-Resolution Satellite Imagery Basemap (Esri World Imagery - Primary Reliable Basemap)
    m = folium.Map(
        location=center,
        zoom_start=config.STUDY_AREA["default_zoom"],
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        name="Satellite Imagery (Esri High-Res)",
        control_scale=True
    )

    # 2. Standard Topographic Basemap (OpenStreetMap)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Standard Topographic (OSM)",
        max_zoom=19
    ).add_to(m)

    rows, cols = risk_grid.shape
    bounds = [
        [float(np.min(lat_grid)), float(np.min(lon_grid))],
        [float(np.max(lat_grid)), float(np.max(lon_grid))]
    ]

    # 4. Study Area Geographic Boundary Box
    folium.Rectangle(
        bounds=bounds,
        color="#38BDF8",
        weight=2.0,
        dash_array="6, 4",
        fill=False,
        name="Study Domain Boundary",
        tooltip=f"Study Domain: {config.STUDY_AREA['name']} (26.45°N–26.85°N, 93.05°E–93.65°E)"
    ).add_to(m)

    # 5. Continuous & Stepped Risk Surface or Environmental Feature Overlay
    if show_risk:
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
            layer_title = "Predicted Flood Risk Surface"
        else:
            # Render Environmental Feature Layer
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

    # 6. Iso-Probability Risk Contours (50% High Risk, 75% Very High Risk)
    if show_contours and risk_grid is not None:
        try:
            cs = plt.contour(lon_grid, lat_grid, risk_grid, levels=[50.0, 75.0])
            if len(cs.allsegs) >= 2:
                # 50% High-Risk contour (Orange dashed)
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
                # 75% Very-High-Risk contour (Crimson dashed)
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

    # 7. Permanent Riverbed Mask Overlay (Brahmaputra braided mainstem)
    if show_water:
        water_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        water_rgba[perm_water_grid == 1] = [2, 132, 199, 215]  # Sky-600 Deep Riverbed
        ImageOverlay(
            image=water_rgba,
            bounds=bounds,
            opacity=0.82,
            name="Permanent Riverbed (JRC Water)",
            interactive=True
        ).add_to(m)

    # 7. Observed Historical Flood Extent Overlay (Ground Truth Reference)
    if show_gt:
        gt_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        gt_rgba[gt_grid == 1] = [6, 182, 212, int(255 * gt_opacity)]  # Electric Cyan
        ImageOverlay(
            image=gt_rgba,
            bounds=bounds,
            opacity=1.0,
            name="Observed Historical Flood Extent",
            interactive=True
        ).add_to(m)

    # 8. Interactive Risk Cell Inspector (Hover Tooltips & Click Popups with Real Data)
    if show_inspector:
        dlat = (np.max(lat_grid) - np.min(lat_grid)) / rows / 2.0
        dlon = (np.max(lon_grid) - np.min(lon_grid)) / cols / 2.0

        features = []
        if df_step is not None and not df_step.empty:
            for _, row in df_step.iterrows():
                lat, lon = row["lat"], row["lon"]
                r_idx = int(row["row"])
                c_idx = int(row["col"])
                score = float(row.get("risk_score", risk_grid[r_idx, c_idx]))
                
                if score >= 75.0:
                    risk_cat = "VERY HIGH"
                elif score >= 50.0:
                    risk_cat = "HIGH"
                elif score >= 25.0:
                    risk_cat = "MODERATE"
                else:
                    risk_cat = "LOW"
                    
                gt_val = int(row.get("is_flooded", gt_grid[r_idx, c_idx]))
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [round(lon - dlon, 5), round(lat - dlat, 5)],
                            [round(lon + dlon, 5), round(lat - dlat, 5)],
                            [round(lon + dlon, 5), round(lat + dlat, 5)],
                            [round(lon - dlon, 5), round(lat + dlat, 5)],
                            [round(lon - dlon, 5), round(lat - dlat, 5)]
                        ]]
                    },
                    "properties": {
                        "cell_id": f"Cell [{r_idx}, {c_idx}]",
                        "prob": f"{score:.1f}%",
                        "risk": risk_cat,
                        "elev": f"{float(row['elevation']):.0f} m",
                        "rain7d": f"{float(row['rainfall_7d']):.1f} mm",
                        "ndwi": f"{float(row['ndwi']):.2f}",
                        "mndwi": f"{float(row['mndwi']):.2f}",
                        "dist_drainage": f"{float(row['dist_to_drainage']):.0f} m",
                        "gt_status": "Observed Inundated (DFO 4924)" if gt_val == 1 else "Observed Dry Ground"
                    }
                })
        else:
            for r in range(rows):
                for c in range(cols):
                    lat, lon = float(lat_grid[r, c]), float(lon_grid[r, c])
                    score = float(risk_grid[r, c])
                    if score >= 75.0:
                        risk_cat = "VERY HIGH"
                    elif score >= 50.0:
                        risk_cat = "HIGH"
                    elif score >= 25.0:
                        risk_cat = "MODERATE"
                    else:
                        risk_cat = "LOW"
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [round(lon - dlon, 5), round(lat - dlat, 5)],
                                [round(lon + dlon, 5), round(lat - dlat, 5)],
                                [round(lon + dlon, 5), round(lat + dlat, 5)],
                                [round(lon - dlon, 5), round(lat + dlat, 5)],
                                [round(lon - dlon, 5), round(lat - dlat, 5)]
                            ]]
                        },
                        "properties": {
                            "cell_id": f"Cell [{r}, {c}]",
                            "prob": f"{score:.1f}%",
                            "risk": risk_cat,
                            "elev": "N/A",
                            "rain7d": "N/A",
                            "ndwi": "N/A",
                            "mndwi": "N/A",
                            "dist_drainage": "N/A",
                            "gt_status": "Observed Inundated" if gt_grid[r, c] == 1 else "Observed Dry"
                        }
                    })

        geojson_obj = {"type": "FeatureCollection", "features": features}

        tooltip = folium.GeoJsonTooltip(
            fields=["cell_id", "prob", "risk", "elev", "rain7d", "ndwi", "mndwi"],
            aliases=["Grid Cell:", "Flood Probability:", "Risk Level:", "Elevation:", "7-Day Rain:", "NDWI:", "MNDWI:"],
            style="""
                background: #0F172A;
                color: #F8FAFC;
                font-family: 'Inter', -apple-system, sans-serif;
                font-size: 11px;
                padding: 8px 12px;
                border: 1px solid #38BDF8;
                border-radius: 6px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
            """,
            localize=True,
            sticky=True
        )

        popup = folium.GeoJsonPopup(
            fields=["cell_id", "prob", "risk", "elev", "rain7d", "ndwi", "mndwi", "dist_drainage", "gt_status"],
            aliases=["Spatial Cell:", "Flood Risk Probability:", "Risk Level:", "SRTM Elevation:", "7-Day Precipitation:", "NDWI (Water Index):", "MNDWI (Moisture Index):", "Distance to River:", "Ground Truth Extent:"],
            style="""
                background-color: #0F172A;
                color: #F8FAFC;
                font-family: 'Inter', -apple-system, sans-serif;
                font-size: 11px;
                border: 1px solid #38BDF8;
                border-radius: 8px;
                padding: 10px;
            """
        )

        gj = folium.GeoJson(
            geojson_obj,
            name="Interactive Cell Inspector (Hover/Click)",
            style_function=lambda x: {
                "fillColor": "#000000",
                "color": "#334155",
                "fillOpacity": 0.0,
                "weight": 0.15
            },
            highlight_function=lambda x: {
                "fillColor": "#38BDF8",
                "color": "#38BDF8",
                "fillOpacity": 0.35,
                "weight": 1.8
            },
            tooltip=tooltip,
            popup=popup
        )
        gj.add_to(m)

    # 9. Key Ground Stations & Landmark Markers
    landmarks = [
        {"name": "Kaziranga National Park HQ (Kohora)", "lat": 26.585, "lon": 93.355, "type": "park", "desc": "UNESCO World Heritage flood-dependent ecosystem"},
        {"name": "Brahmaputra Silghat Hydrological Gauge", "lat": 26.780, "lon": 93.120, "type": "gauge", "desc": "CWC Primary Water Level Monitoring Station"},
        {"name": "Bokakhat Emergency Operations Center", "lat": 26.620, "lon": 93.590, "type": "hq", "desc": "Sub-divisional disaster evacuation command"},
        {"name": "Diphlu River Wetland Confluence", "lat": 26.680, "lon": 93.280, "type": "wetland", "desc": "Critical backwater inundation choke point"}
    ]

    for lm in landmarks:
        is_gauge = (lm["type"] == "gauge")
        folium.CircleMarker(
            location=[lm["lat"], lm["lon"]],
            radius=6.0,
            color="#FFFFFF",
            weight=1.5,
            fill=True,
            fill_color="#EF4444" if is_gauge else "#38BDF8",
            fill_opacity=0.95,
            popup=f"<div style='font-family: Inter, sans-serif; font-size: 12px; color: #0F172A; min-width: 180px;'><b>{lm['name']}</b><br><span style='color: #64748B;'>{lm['desc']}</span><br><b>Coordinates:</b> {lm['lat']:.3f}°N, {lm['lon']:.3f}°E</div>",
            tooltip=f"{lm['name']} ({'Gauge' if is_gauge else 'Station'})"
        ).add_to(m)

    # 10. Map Navigation Controls & Layer Switcher
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MousePosition(position="bottomleft", separator=" | ", empty_string="Coordinates: Lat, Lon").add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    return m

def plot_risk_timeline(timeline_df: pd.DataFrame) -> go.Figure:
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

def plot_feature_importance_chart(df_imp: pd.DataFrame) -> go.Figure:
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

def plot_confusion_matrix_chart(cm_dict: dict) -> go.Figure:
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
