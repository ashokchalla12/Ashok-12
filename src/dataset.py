"""Dataset loading, validation, and import utilities."""

import io

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)
from src.utils import ensure_dir, save_csv

VALID_SEASONS = {"spring", "summer", "autumn", "winter"}
VALID_LAND_USES = {"urban", "agricultural", "forest", "wetland"}

REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]


def generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic water-level data for development and demos."""
    rng = np.random.default_rng(random_state)
    seasons = list(VALID_SEASONS)
    land_uses = list(VALID_LAND_USES)

    rainfall = rng.uniform(0, 150, n_samples)
    temperature = rng.uniform(-5, 35, n_samples)
    humidity = rng.uniform(30, 100, n_samples)
    wind_speed = rng.uniform(0, 60, n_samples)
    river_level = rng.uniform(0.5, 8.0, n_samples)
    soil_moisture = rng.uniform(10, 90, n_samples)
    evaporation = rng.uniform(0, 10, n_samples)
    pressure = rng.uniform(980, 1040, n_samples)
    season = rng.choice(seasons, n_samples)
    land_use = rng.choice(land_uses, n_samples)

    water_rise = (
        0.35 * rainfall
        + 0.8 * river_level
        + 0.15 * soil_moisture
        - 0.05 * evaporation
        - 0.02 * wind_speed
        + rng.normal(0, 5, n_samples)
    )
    water_rise = np.clip(water_rise, 0, None)

    return pd.DataFrame(
        {
            "rainfall_mm": rainfall.round(2),
            "temperature_c": temperature.round(1),
            "humidity_pct": humidity.round(1),
            "wind_speed_kmh": wind_speed.round(1),
            "river_level_m": river_level.round(2),
            "soil_moisture_pct": soil_moisture.round(1),
            "evaporation_mm": evaporation.round(2),
            "pressure_hpa": pressure.round(1),
            "season": season,
            "land_use": land_use,
            TARGET_COLUMN: water_rise.round(2),
        }
    )


def load_or_create_dataset() -> pd.DataFrame:
    """Load raw data or create synthetic data if missing."""
    ensure_dir(RAW_DATA_DIR)
    if not RAW_DATA_FILE.exists():
        df = generate_synthetic_data()
        save_csv(df, RAW_DATA_FILE)
        return df
    return pd.read_csv(RAW_DATA_FILE)


def get_feature_columns() -> list[str]:
    """Return all input feature column names."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate_row(data: dict) -> dict:
    """Validate and normalize a single dataset record."""
    missing = [col for col in REQUIRED_COLUMNS if col not in data or data[col] == ""]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    row = {}
    for col in NUMERIC_FEATURES + [TARGET_COLUMN]:
        try:
            row[col] = float(data[col])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{col}' must be a number") from exc

    season = str(data["season"]).strip().lower()
    if season not in VALID_SEASONS:
        raise ValueError(f"season must be one of: {', '.join(sorted(VALID_SEASONS))}")
    row["season"] = season

    land_use = str(data["land_use"]).strip().lower()
    if land_use not in VALID_LAND_USES:
        raise ValueError(f"land_use must be one of: {', '.join(sorted(VALID_LAND_USES))}")
    row["land_use"] = land_use

    return row


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an imported DataFrame has required columns and values."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing columns: {', '.join(missing_cols)}")

    cleaned_rows = []
    for idx, row in df[REQUIRED_COLUMNS].iterrows():
        try:
            cleaned_rows.append(validate_row(row.to_dict()))
        except ValueError as exc:
            raise ValueError(f"Row {idx + 2}: {exc}") from exc

    return pd.DataFrame(cleaned_rows)


def append_row(data: dict) -> int:
    """Append a validated row to the dataset. Returns total row count."""
    row = validate_row(data)
    df = load_or_create_dataset()
    updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_csv(updated, RAW_DATA_FILE)
    return len(updated)


def import_csv(file_bytes: bytes) -> dict:
    """Import rows from a CSV file and merge with existing data."""
    try:
        incoming = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError("Could not read CSV file. Ensure it is a valid comma-separated file.") from exc

    if incoming.empty:
        raise ValueError("Uploaded CSV contains no data rows.")

    validated = validate_dataframe(incoming)
    existing = load_or_create_dataset()
    updated = pd.concat([existing, validated], ignore_index=True)
    save_csv(updated, RAW_DATA_FILE)

    return {"added": len(validated), "total": len(updated)}


def get_dataset_preview(n: int = 8) -> list[dict]:
    """Return the most recent dataset rows for display."""
    df = load_or_create_dataset()
    preview = df.tail(n).iloc[::-1]
    return preview.round(2).to_dict(orient="records")
