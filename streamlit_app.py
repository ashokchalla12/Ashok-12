"""
Streamlit app for Rising Water Prediction.
Deploy this file on Streamlit Cloud (NOT app.py — that is the Flask server).
"""

import json
import sys
from pathlib import Path

# Fix imports on Streamlit Cloud — project root must be on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.dataset import (
    REQUIRED_COLUMNS,
    append_row,
    import_csv,
    load_or_create_dataset,
)
from src.predict import predict_water_rise

st.set_page_config(
    page_title="Rising Water Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
      h1, h2, h3 { font-family: 'DM Serif Display', serif !important; font-weight: 400 !important; }
      .block-container { padding-top: 2rem; max-width: 1100px; }
      div[data-testid="stMetricValue"] { font-family: 'DM Serif Display', serif; color: #2aa8c4; }
    </style>
    """,
    unsafe_allow_html=True,
)


def model_ready() -> bool:
    return BEST_MODEL_PATH.exists()


def train_model():
    from src.train import train_best_model

    with st.spinner("Training model — this may take a minute..."):
        return train_best_model()


def risk_badge(level: str) -> str:
    colors = {"low": "#4ade80", "moderate": "#fbbf24", "high": "#f87171"}
    color = colors.get(level, "#94a3b8")
    return f"<span style='color:{color};font-weight:700;'>{level.upper()}</span>"


def page_home():
    st.markdown("##### Hydrological Intelligence")
    st.title("Predict Rising Water Levels")
    st.markdown(
        "A machine learning platform that estimates water rise from rainfall, "
        "river level, soil moisture, and environmental signals."
    )

    if model_ready():
        st.success("Model is ready for predictions.")
    else:
        st.warning("Model not trained yet. Use the sidebar to train, or run `python -m src.train` locally and push `models/` to GitHub.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Platform", "Streamlit Cloud")
    c2.metric("Features", len(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
    c3.metric("Model", "Ready" if model_ready() else "Pending")


def page_predict():
    st.markdown("##### Forecast")
    st.title("Water Rise Prediction")

    if not model_ready():
        st.error("Train the model first using the **Train Model** button in the sidebar.")
        return

    defaults = {
        "rainfall_mm": 45.0,
        "temperature_c": 22.0,
        "humidity_pct": 70.0,
        "wind_speed_kmh": 15.0,
        "river_level_m": 3.5,
        "soil_moisture_pct": 55.0,
        "evaporation_mm": 2.5,
        "pressure_hpa": 1013.0,
    }

    with st.form("predict_form"):
        st.subheader("Environmental Inputs")
        cols = st.columns(2)
        data = {}
        for i, feature in enumerate(NUMERIC_FEATURES):
            label = feature.replace("_", " ").title()
            with cols[i % 2]:
                data[feature] = st.number_input(label, value=float(defaults.get(feature, 0.0)))

        c1, c2 = st.columns(2)
        with c1:
            data["season"] = st.selectbox("Season", ["spring", "summer", "autumn", "winter"], index=1)
        with c2:
            data["land_use"] = st.selectbox("Land Use", ["urban", "agricultural", "forest", "wetland"], index=1)

        submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

    if submitted:
        try:
            result = predict_water_rise(data)
            st.markdown("---")
            st.subheader("Forecast Result")
            st.metric("Predicted Water Rise", f"{result['water_rise_cm']} cm")
            st.markdown(f"Risk Level: {risk_badge(result['risk_level'])}", unsafe_allow_html=True)
        except Exception as exc:
            st.error(str(exc))


def page_dataset():
    st.markdown("##### Data Management")
    st.title("Dataset Manager")

    df = load_or_create_dataset()
    st.metric("Total Records", len(df))

    tab_upload, tab_manual, tab_preview = st.tabs(["Upload CSV", "Add Record", "Preview"])

    with tab_upload:
        st.caption(f"Required columns: {', '.join(REQUIRED_COLUMNS)}")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded and st.button("Upload Dataset", type="primary"):
            try:
                result = import_csv(uploaded.getvalue())
                st.success(f"Added {result['added']} row(s). Total: {result['total']}. Retrain the model after adding data.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    with tab_manual:
        with st.form("manual_form"):
            cols = st.columns(2)
            row = {}
            for i, feature in enumerate(NUMERIC_FEATURES):
                with cols[i % 2]:
                    row[feature] = st.number_input(feature.replace("_", " ").title(), key=f"m_{feature}")
            row["season"] = st.selectbox("Season", ["spring", "summer", "autumn", "winter"], key="m_season")
            row["land_use"] = st.selectbox("Land Use", ["urban", "agricultural", "forest", "wetland"], key="m_land")
            row[TARGET_COLUMN] = st.number_input("Water Rise (cm)", key="m_target")

            if st.form_submit_button("Add Record", type="primary"):
                try:
                    total = append_row(row)
                    st.success(f"Record added. Total rows: {total}. Retrain the model to use new data.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with tab_preview:
        st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)


def page_dashboard():
    st.markdown("##### Insights")
    st.title("Data Dashboard")

    df = load_or_create_dataset()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Avg Rainfall (mm)", round(df["rainfall_mm"].mean(), 2) if "rainfall_mm" in df.columns else "—")
    c3.metric("Avg Water Rise (cm)", round(df[TARGET_COLUMN].mean(), 2) if TARGET_COLUMN in df.columns else "—")
    c4.metric("Model", "Ready" if model_ready() else "Pending")

    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.subheader("Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", metrics.get("mae"))
        m2.metric("RMSE", metrics.get("rmse"))
        m3.metric("R²", metrics.get("r2"))
        m4.metric("Test Samples", metrics.get("samples"))
    else:
        st.info("No evaluation metrics yet. Train and evaluate the model locally, then commit `reports/metrics.json`.")


# Sidebar
with st.sidebar:
    st.title("Rising Water")
    page = st.radio(
        "Navigate",
        ["Home", "Predict", "Dataset", "Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("Train Model", use_container_width=True):
        try:
            metrics = train_model()
            st.success(f"Done — {metrics['model']} (R² {metrics['r2']:.3f})")
        except Exception as exc:
            st.error(str(exc))

pages = {
    "Home": page_home,
    "Predict": page_predict,
    "Dataset": page_dataset,
    "Dashboard": page_dashboard,
}
pages[page]()
