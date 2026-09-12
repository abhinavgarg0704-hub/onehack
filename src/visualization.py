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
import config

def build_interactive_map(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    risk_grid: np.ndarray,
    gt_grid: np.ndarray,
    perm_water_grid: np.ndarray,
    df_step: pd.DataFrame = None,
    show_risk: bool = True,
    show_gt: bool = True,
    show_water: bool = True,
    show_inspector: bool = True,
    risk_threshold: float = 20.0,
    risk_opacity: float = 0.85,
    gt_opacity: float = 0.72
) -> folium.Map:
    """
    Constructs an interactive satellite intelligence Folium map featuring:
    - Multi-basemap selector (Carto Dark Command, Esri High-Res Satellite, Standard Topo)
    - Semantic continuous/stepped risk surface overlay (0-25% Low, 25-50% Mod, 50-75% High, 75-100% Very High)
    - Distinct observed historical flood extent overlay (ground truth)
    - Permanent Brahmaputra braided river channel mask (JRC surface water)
    - Interactive 3,750-cell GeoJSON risk inspector with hover tooltips and click popups from real data
    - Regional hydrological stations, landmarks, study boundary, scale bar, and coordinate tracker
    """
    from folium.raster_layers import ImageOverlay

    center = [config.STUDY_AREA["center_lat"], config.STUDY_AREA["center_lon"]]
    
    # 1. Initialize Map with Carto Dark Matter as primary command-center basemap
    m = folium.Map(
        location=center,
        zoom_start=config.STUDY_AREA["default_zoom"],
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Carto Dark Command",
        control_scale=True
    )

    # 2. High-Resolution Satellite Imagery Basemap (Esri World Imagery)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        name="Satellite Imagery (Esri High-Res)",
        max_zoom=19
    ).add_to(m)

    # 3. Standard Topographic Basemap (OpenStreetMap)
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

    # 5. Continuous & Stepped Risk Surface Overlay
    # Semantic color palette matching the locked legend:
    # 0–25%    LOW: Emerald (#10B981)
    # 25–50%   MODERATE: Amber (#F59E0B)
    # 50–75%   HIGH: Orange (#F97316)
    # 75–100%  VERY HIGH: Crimson (#EF4444)
    if show_risk:
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

        ImageOverlay(
            image=rgba_img,
            bounds=bounds,
            opacity=1.0,
            name="Predicted Flood Risk Surface",
            interactive=True,
            cross_origin=False
        ).add_to(m)

    # 6. Permanent Riverbed Mask Overlay (Brahmaputra braided mainstem)
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
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
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
