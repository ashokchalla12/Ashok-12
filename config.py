"""Central configuration for the Rising Water Prediction project."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "water_level_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"

# Model artifacts
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"

# Feature columns
NUMERIC_FEATURES = [
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "river_level_m",
    "soil_moisture_pct",
    "evaporation_mm",
    "pressure_hpa",
]

CATEGORICAL_FEATURES = ["season", "land_use"]

TARGET_COLUMN = "water_rise_cm"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Flask
class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
    PORT = int(os.environ.get("FLASK_PORT", 5000))
