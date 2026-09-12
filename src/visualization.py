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
    m = folium.Map(
        location=center,
        zoom_start=config.STUDY_AREA["default_zoom"],
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Optional Carto Dark Matter basemap
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Carto Dark Mode",
        max_zoom=19
    ).add_to(m)

    rows, cols = risk_grid.shape
    bounds = [
        [float(np.min(lat_grid)), float(np.min(lon_grid))],
        [float(np.max(lat_grid)), float(np.max(lon_grid))]
    ]

    # 1. Study Area Boundary Box
    folium.Rectangle(
        bounds=bounds,
        color="#F39C12",
        weight=2,
        dash_array="5, 5",
        fill=False,
        popup=f"{config.STUDY_AREA['name']} Study Boundary"
    ).add_to(m)

    # 2. Risk Heatmap / Surface Overlay
    if show_risk:
        # Create an RGBA image array representing the risk surface
        rgba_img = np.zeros((rows, cols, 4), dtype=np.uint8)
        
        # Color mapping:
        # Low (0-25): Green (#2ECC71 -> 46, 204, 113)
        # Moderate (25-50): Yellow (#F1C40F -> 241, 196, 15)
        # High (50-75): Orange (#E67E22 -> 230, 126, 34)
        # Very High (75-100): Red (#E74C3C -> 231, 76, 60)
        
        for r in range(rows):
            for c in range(cols):
                score = risk_grid[r, c]
                if score < risk_threshold:
                    continue  # transparent
                if score < 25:
                    rgba_img[r, c] = [46, 204, 113, 140]
                elif score < 50:
                    rgba_img[r, c] = [241, 196, 15, 170]
                elif score < 75:
                    rgba_img[r, c] = [230, 126, 34, 200]
                else:
                    rgba_img[r, c] = [231, 76, 60, 230]

        ImageOverlay(
            image=rgba_img,
            bounds=bounds,
            opacity=0.82,
            name="Predicted Flood Risk Heatmap",
            interactive=True,
            cross_origin=False
        ).add_to(m)

    # 3. Permanent Water Mask Overlay (Brahmaputra braided channel)
    if show_water:
        water_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        water_rgba[perm_water_grid == 1] = [31, 119, 180, 210]  # Deep river blue
        ImageOverlay(
            image=water_rgba,
            bounds=bounds,
            opacity=0.75,
            name="Permanent River Channel (JRC Water)",
            interactive=True
        ).add_to(m)

    # 4. Ground Truth Inundation Overlay
    if show_gt:
        gt_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        gt_rgba[gt_grid == 1] = [0, 240, 255, 180]  # Electric Cyan
        ImageOverlay(
            image=gt_rgba,
            bounds=bounds,
            opacity=0.70,
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
        folium.CircleMarker(
            location=[lm["lat"], lm["lon"]],
            radius=6,
            color="#FFFFFF",
            weight=1.5,
            fill=True,
            fill_color="#FF4136" if lm["type"] == "gauge" else "#0074D9",
            fill_opacity=0.9,
            popup=f"<b>{lm['name']}</b><br>Lat: {lm['lat']}, Lon: {lm['lon']}"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

def plot_risk_timeline(timeline_df: pd.DataFrame) -> go.Figure:
    """
    Renders an interactive Plotly chart showing Mean Regional Risk (%) and 7-day Rainfall (mm)
    approaching and during the historical flood event.
    """
    fig = go.Figure()

    # Rainfall bars (secondary y-axis)
    fig.add_trace(go.Bar(
        x=timeline_df["time_tag"],
        y=timeline_df["rainfall_7d"],
        name="7-Day Antecedent Rainfall (mm)",
        marker_color="rgba(52, 152, 219, 0.45)",
        yaxis="y2"
    ))

    # Mean Risk line
    fig.add_trace(go.Scatter(
        x=timeline_df["time_tag"],
        y=timeline_df["mean_risk"],
        mode="lines+markers+text",
        name="Predicted Flood Risk (%)",
        line=dict(color="#E74C3C", width=3.5),
        marker=dict(size=9, color="#E74C3C"),
        text=[f"{v}%" for v in timeline_df["mean_risk"]],
        textposition="top center",
        yaxis="y1"
    ))

    # Warning threshold reference lines
    fig.add_hline(y=60, line_dash="dot", line_color="#E67E22", annotation_text="WARNING (60%)", annotation_position="bottom right")
    fig.add_hline(y=80, line_dash="dash", line_color="#C0392B", annotation_text="HIGH RISK (80%)", annotation_position="top right")

    fig.update_layout(
        title="<b>Temporal Risk Escalation vs Antecedent Precipitation (July 2020 Event)</b>",
        xaxis=dict(title="Timeline Step (T-7 to Peak Event T)", tickfont=dict(color="#E0E0E0")),
        yaxis=dict(
            title="<b>Predicted Flood Risk (%)</b>",
            range=[0, 105],
            tickfont=dict(color="#E74C3C"),
            gridcolor="#2C3E50"
        ),
        yaxis2=dict(
            title="<b>7-Day Rainfall (mm)</b>",
            overlaying="y",
            side="right",
            range=[0, 420],
            tickfont=dict(color="#3498DB"),
            showgrid=False
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(30, 41, 59, 0.8)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )

    return fig

def plot_feature_importance_chart(df_imp: pd.DataFrame) -> go.Figure:
    """
    Renders a horizontal bar chart displaying ranked feature contributions.
    """
    df_sorted = df_imp.sort_values(by="importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted["importance"],
        y=df_sorted["feature"],
        orientation="h",
        marker=dict(
            color=df_sorted["importance"],
            colorscale="Viridis",
            showscale=False
        ),
        text=[f"{v:.3f}" for v in df_sorted["importance"]],
        textposition="auto"
    ))

    fig.update_layout(
        title="<b>Predictive Feature Importance (Gini / Gain Contribution)</b>",
        xaxis=dict(title="Relative Importance Score", gridcolor="#2C3E50"),
        yaxis=dict(title=""),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=10, r=20, t=45, b=30),
        height=400
    )

    return fig

def plot_confusion_matrix_chart(cm_dict: dict) -> go.Figure:
    """
    Renders an annotated 2x2 confusion matrix heatmap.
    """
    z = [[cm_dict["tn"], cm_dict["fp"]],
         [cm_dict["fn"], cm_dict["tp"]]]
    x = ["Predicted Non-Flood", "Predicted Flood"]
    y = ["Actual Non-Flood", "Actual Flood"]

    fig = px.imshow(
        z,
        x=x,
        y=y,
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto"
    )

    fig.update_layout(
        title="<b>Holdout Validation Confusion Matrix (Peak Event T)</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=20, r=20, t=45, b=20),
        height=320
    )

    return fig
