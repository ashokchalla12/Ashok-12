"""Model evaluation utilities."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import BEST_MODEL_PATH, RANDOM_STATE, REPORTS_DIR, TARGET_COLUMN, TEST_SIZE
from src.dataset import get_feature_columns, load_or_create_dataset
from src.feature_engineering import add_engineered_features
from src.preprocessing import build_preprocessor, clean_data
from src.utils import ensure_dir, load_artifact, setup_logging

logger = setup_logging(__name__)


def evaluate_model() -> dict:
    """Evaluate the persisted model on a held-out test split."""
    model = load_artifact(BEST_MODEL_PATH)
    raw = load_or_create_dataset()
    df = clean_data(raw)
    df = add_engineered_features(df)

    base_cols = get_feature_columns()
    X = df[base_cols]
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    X_test_t = preprocessor.transform(X_test)

    y_pred = model.predict(X_test_t)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "r2": float(r2_score(y_test, y_pred)),
        "samples": int(len(y_test)),
    }

    ensure_dir(REPORTS_DIR)
    report_path = REPORTS_DIR / "metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Evaluation metrics saved to %s", report_path)

    _save_residual_plot(y_test.values, y_pred)
    return metrics


def _save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save a residual plot to reports/."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Water Rise (cm)")
    plt.ylabel("Residual (cm)")
    plt.title("Residual Plot")
    plt.tight_layout()
    ensure_dir(REPORTS_DIR)
    plt.savefig(REPORTS_DIR / "residual_plot.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    evaluate_model()
