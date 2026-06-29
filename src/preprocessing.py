"""Data cleaning and preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    ENCODER_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_FILE,
    SCALER_PATH,
    TARGET_COLUMN,
)
from src.dataset import get_feature_columns, load_or_create_dataset
from src.utils import load_artifact, save_artifact, save_csv, setup_logging

logger = setup_logging(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning steps to the raw dataset."""
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COLUMN])

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            q1, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(q1, q99)

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def preprocess_and_save() -> pd.DataFrame:
    """Clean raw data and save processed CSV."""
    df = load_or_create_dataset()
    cleaned = clean_data(df)
    save_csv(cleaned, PROCESSED_DATA_FILE)
    logger.info("Saved processed data to %s (%d rows)", PROCESSED_DATA_FILE, len(cleaned))
    return cleaned


def fit_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Fit preprocessor on training features and persist artifacts."""
    preprocessor = build_preprocessor()
    X = df[get_feature_columns()]
    preprocessor.fit(X)
    save_artifact(preprocessor, PREPROCESSOR_PATH)
    save_artifact(preprocessor.named_transformers_["num"], SCALER_PATH)
    save_artifact(preprocessor.named_transformers_["cat"], ENCODER_PATH)
    return preprocessor


def transform_for_inference(df: pd.DataFrame) -> np.ndarray:
    """Transform input features using persisted fitted preprocessors."""
    X = df[get_feature_columns()]

    if PREPROCESSOR_PATH.exists():
        preprocessor = load_artifact(PREPROCESSOR_PATH)
        return preprocessor.transform(X)

    numeric_pipeline = load_artifact(SCALER_PATH)
    categorical_pipeline = load_artifact(ENCODER_PATH)
    X_num = numeric_pipeline.transform(df[NUMERIC_FEATURES])
    X_cat = categorical_pipeline.transform(df[CATEGORICAL_FEATURES])
    return np.hstack([X_num, X_cat])


def transform_features(df: pd.DataFrame, preprocessor: ColumnTransformer | None = None) -> pd.DataFrame:
    """Transform features using a fitted preprocessor."""
    if preprocessor is not None:
        transformed = preprocessor.transform(df[get_feature_columns()])
    else:
        transformed = transform_for_inference(df)
    return pd.DataFrame(transformed)
