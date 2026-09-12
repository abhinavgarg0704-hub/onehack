"""
Early Warning and Threat Assessment System
Threshold-based alert generation, advisory checklists, and executive KPI summaries.
"""

import config

def evaluate_flood_alert(mean_risk: float, high_risk_area_km2: float, rainfall_7d: float) -> dict:
    """
    Evaluates early-warning alert tier:
    - NORMAL: Mean Risk < 40
    - WATCH: 40 <= Mean Risk < 60
    - WARNING: 60 <= Mean Risk < 80
    - HIGH FLOOD RISK: Mean Risk >= 80 (or severe localized area trigger)
    """
    # Early-warning tier triggers:
    # High Flood Risk: Mean floodplain risk >= 75% OR high-risk area exceeds 150 km²
    # Warning: Mean floodplain risk >= 55% OR high-risk area exceeds 75 km²
    # Watch: Mean floodplain risk >= 35% OR high-risk area exceeds 30 km²
    if mean_risk >= 75.0 or high_risk_area_km2 >= 150.0:
        key = "HIGH_RISK"
    elif mean_risk >= 55.0 or high_risk_area_km2 >= 75.0:
        key = "WARNING"
    elif mean_risk >= 35.0 or high_risk_area_km2 >= 30.0:
        key = "WATCH"
    else:
        key = "NORMAL"

    cfg = config.ALERT_LEVELS[key]

    return {
        "alert_key": key,
        "label": cfg["label"],
        "badge": cfg["badge"],
        "color": cfg["color"],
        "icon": cfg["icon"],
        "action": cfg["action"],
        "mean_risk": mean_risk,
        "high_risk_area_km2": high_risk_area_km2,
        "rainfall_7d": rainfall_7d,
        "prediction_horizon": config.PREDICTION_HORIZON,
        "region": config.STUDY_AREA["name"],
        "disclaimer": "Analytical demonstration using historical satellite & rainfall archives. Not an operational emergency warning."
    }
