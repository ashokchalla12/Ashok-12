"""Model training pipeline."""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.dataset import get_feature_columns, load_or_create_dataset
from src.feature_engineering import add_engineered_features
from src.preprocessing import build_preprocessor, clean_data, fit_preprocessor
from src.utils import save_artifact, setup_logging

logger = setup_logging(__name__)


def get_candidate_models() -> dict:
    """Return models and hyperparameter grids for tuning."""
    return {
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        ),
        "ridge": (
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0]},
        ),
    }


def train_best_model() -> dict:
    """Train and persist the best model based on cross-validated MAE."""
    raw = load_or_create_dataset()
    df = add_engineered_features(clean_data(raw))

    base_cols = get_feature_columns()
    X = df[base_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    best_score = float("inf")
    best_model = None
    best_name = ""

    for name, (model, param_grid) in get_candidate_models().items():
        search = GridSearchCV(
            model,
            param_grid,
            cv=CV_FOLDS,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        search.fit(X_train_t, y_train)
        mae = -search.best_score_
        logger.info("%s CV MAE: %.3f", name, mae)
        if mae < best_score:
            best_score = mae
            best_model = search.best_estimator_
            best_name = name

    y_pred = best_model.predict(X_test_t)
    metrics = {
        "model": best_name,
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }

    save_artifact(best_model, BEST_MODEL_PATH)
    fit_preprocessor(df)

    logger.info("Best model: %s | Test MAE: %.3f | R2: %.3f", best_name, metrics["mae"], metrics["r2"])
    return metrics


if __name__ == "__main__":
    train_best_model()
