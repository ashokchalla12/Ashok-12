"""Utility helpers for the Rising Water Prediction project."""

import logging
from pathlib import Path

import joblib
import pandas as pd


def setup_logging(name: str = "rising_water", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_artifact(obj, path: Path) -> None:
    """Persist a model or transformer to disk."""
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_artifact(path: Path):
    """Load a persisted model or transformer."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file as a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
