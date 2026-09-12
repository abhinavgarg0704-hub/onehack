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
    show_risk: bool = True,
    show_gt: bool = True,
    show_water: bool = True,
    risk_threshold: float = 25.0
) -> folium.Map:
    """
    Constructs an interactive Folium map with raster overlays and vector markers.
    """
    from folium.raster_layers import ImageOverlay

    center = [config.STUDY_AREA["center_lat"], config.STUDY_AREA["center_lon"]]
    
    # Initialize Folium Map with Carto Dark Matter as primary basemap
    m = folium.Map(
        location=center,
        zoom_start=config.STUDY_AREA["default_zoom"],
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Carto Dark Command",
        control_scale=True
    )

    # Optional OpenStreetMap basemap
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap Light",
        max_zoom=19
    ).add_to(m)

    rows, cols = risk_grid.shape
    bounds = [
        [float(np.min(lat_grid)), float(np.min(lon_grid))],
        [float(np.max(lat_grid)), float(np.max(lon_grid))]
    ]

    # 1. Study Area Boundary Box (Subtle slate border)
    folium.Rectangle(
        bounds=bounds,
        color="#38BDF8",
        weight=1.8,
        dash_array="6, 4",
        fill=False,
        popup=f"<b>Study Domain:</b> {config.STUDY_AREA['name']}"
    ).add_to(m)

    # 2. Risk Heatmap / Surface Overlay
    if show_risk:
        rgba_img = np.zeros((rows, cols, 4), dtype=np.uint8)
        
        # Professional geospatial hazard colormap:
        # Low (0-25): Emerald (#10B981)
        # Moderate (25-50): Amber (#F59E0B)
        # High (50-75): Orange (#F97316)
        # Very High (75-100): Crimson (#EF4444)
        
        for r in range(rows):
            for c in range(cols):
                score = risk_grid[r, c]
                if score < risk_threshold:
                    continue
                if score < 25:
                    rgba_img[r, c] = [16, 185, 129, 130]
                elif score < 50:
                    rgba_img[r, c] = [245, 158, 11, 160]
                elif score < 75:
                    rgba_img[r, c] = [249, 115, 22, 195]
                else:
                    rgba_img[r, c] = [239, 68, 68, 225]

        ImageOverlay(
            image=rgba_img,
            bounds=bounds,
            opacity=0.85,
            name="Predicted Flood Risk Heatmap",
            interactive=True,
            cross_origin=False
        ).add_to(m)

    # 3. Permanent Water Mask Overlay (Brahmaputra braided mainstem)
    if show_water:
        water_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        water_rgba[perm_water_grid == 1] = [2, 132, 199, 210]  # Sky-600 deep water
        ImageOverlay(
            image=water_rgba,
            bounds=bounds,
            opacity=0.80,
            name="Permanent River Channel (JRC Water)",
            interactive=True
        ).add_to(m)

    # 4. Ground Truth Inundation Overlay
    if show_gt:
        gt_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        gt_rgba[gt_grid == 1] = [6, 182, 212, 175]  # Electric Cyan
        ImageOverlay(
            image=gt_rgba,
            bounds=bounds,
            opacity=0.72,
            name="Observed Flood Extent (Ground Truth)",
            interactive=True
        ).add_to(m)

    # 5. Key Ground Stations & Landmark Markers
    landmarks = [
        {"name": "Kaziranga National Park HQ (Kohora)", "lat": 26.585, "lon": 93.355, "type": "park"},
        {"name": "Brahmaputra River Silghat Hydrological Gauge", "lat": 26.780, "lon": 93.120, "type": "gauge"},
        {"name": "Bokakhat Sub-Divisional Operations Center", "lat": 26.620, "lon": 93.590, "type": "hq"},
        {"name": "Diphlu River Wetland Confluence", "lat": 26.680, "lon": 93.280, "type": "wetland"},
    ]

    for lm in landmarks:
        is_gauge = (lm["type"] == "gauge")
        folium.CircleMarker(
            location=[lm["lat"], lm["lon"]],
            radius=5.5,
            color="#FFFFFF",
            weight=1.2,
            fill=True,
            fill_color="#EF4444" if is_gauge else "#38BDF8",
            fill_opacity=0.95,
            popup=f"<div style='font-family: Inter, sans-serif; font-size: 12px; color: #0F172A;'><b>{lm['name']}</b><br>Coordinates: {lm['lat']:.3f}°N, {lm['lon']:.3f}°E</div>"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
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
