"""Inference utilities for water rise prediction."""

import pandas as pd

from config import BEST_MODEL_PATH, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.preprocessing import transform_for_inference
from src.utils import load_artifact


def _validate_input(data: dict) -> pd.DataFrame:
    """Validate and normalize a single prediction payload."""
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in required if col not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    row = {col: data[col] for col in required}
    return pd.DataFrame([row])


def predict_water_rise(data: dict) -> dict:
    """Predict water rise in centimeters from input features."""
    if not BEST_MODEL_PATH.exists():
        raise RuntimeError("Model artifacts not found. Run training first: python -m src.train")

    model = load_artifact(BEST_MODEL_PATH)
    df = _validate_input(data)
    X = transform_for_inference(df)
    prediction = float(model.predict(X)[0])

    risk = "low"
    if prediction >= 50:
        risk = "high"
    elif prediction >= 25:
        risk = "moderate"

    return {
        "water_rise_cm": round(prediction, 2),
        "risk_level": risk,
    }
