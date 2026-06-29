"""Feature engineering for water rise prediction."""

import pandas as pd

from config import NUMERIC_FEATURES, TARGET_COLUMN
from src.dataset import get_feature_columns


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw inputs."""
    df = df.copy()

    if {"rainfall_mm", "soil_moisture_pct"}.issubset(df.columns):
        df["rain_soil_interaction"] = df["rainfall_mm"] * df["soil_moisture_pct"] / 100

    if {"river_level_m", "rainfall_mm"}.issubset(df.columns):
        df["flood_risk_index"] = df["river_level_m"] * 10 + df["rainfall_mm"] * 0.2

    if {"humidity_pct", "temperature_c"}.issubset(df.columns):
        df["heat_moisture_ratio"] = df["humidity_pct"] / (df["temperature_c"].abs() + 1)

    if {"wind_speed_kmh", "evaporation_mm"}.issubset(df.columns):
        df["drying_index"] = df["wind_speed_kmh"] * df["evaporation_mm"]

    return df


def select_model_features(df: pd.DataFrame) -> list[str]:
    """Return feature columns available for modeling."""
    base = get_feature_columns()
    engineered = [c for c in df.columns if c not in base + [TARGET_COLUMN]]
    return base + engineered
