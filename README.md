# Rising Water Prediction

End-to-end machine learning project for predicting water level rise from environmental and hydrological data, with a Flask web interface for interactive inference.

## Project Structure

```
Rising-Water-Prediction/
├── app.py                  # Flask web application
├── config.py               # Paths, features, and settings
├── requirements.txt
├── data/                   # Raw, processed, and external data
├── notebooks/              # Jupyter notebooks for EDA and modeling
├── models/                 # Trained model artifacts (.pkl)
├── src/                    # Core ML pipeline modules
├── templates/              # HTML templates
├── static/                 # CSS, JS, images
├── reports/                # Evaluation metrics and plots
└── docs/                   # Additional documentation
```

## Features

- Data loading with synthetic fallback for development
- Preprocessing pipeline (imputation, scaling, encoding)
- Feature engineering (interaction and risk indices)
- Model training with GridSearchCV (Random Forest, Gradient Boosting, Ridge)
- Evaluation metrics and residual plots
- Flask web app with prediction form and dashboard
- REST API endpoint: `POST /api/predict`

## Deploy on Streamlit Cloud

**Important:** Use `streamlit_app.py`, not `app.py`. The Flask file will not work on Streamlit.

1. Push the **full project** to GitHub (not just `app.py`):
   - `streamlit_app.py`
   - `config.py`
   - `src/` (entire folder)
   - `requirements.txt`
   - `models/` (optional — or train from the sidebar after deploy)
   - `data/raw/.gitkeep` (data is auto-generated if missing)

2. On [share.streamlit.io](https://share.streamlit.io), create a new app:
   - **Main file path:** `streamlit_app.py`
   - **Requirements file:** `requirements.txt`

3. After deploy, click **Train Model** in the sidebar if predictions fail (no model files yet).

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python -m src.train
python -m src.evaluate
```

### 4. Run the web app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Input Features

| Feature | Type | Description |
|---------|------|-------------|
| rainfall_mm | numeric | Rainfall in millimeters |
| temperature_c | numeric | Temperature in Celsius |
| humidity_pct | numeric | Relative humidity (%) |
| wind_speed_kmh | numeric | Wind speed (km/h) |
| river_level_m | numeric | Current river level (m) |
| soil_moisture_pct | numeric | Soil moisture (%) |
| evaporation_mm | numeric | Evaporation (mm) |
| pressure_hpa | numeric | Atmospheric pressure (hPa) |
| season | categorical | spring, summer, autumn, winter |
| land_use | categorical | urban, agricultural, forest, wetland |

**Target:** `water_rise_cm` — predicted water rise in centimeters.

## API Usage

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall_mm": 45,
    "temperature_c": 22,
    "humidity_pct": 70,
    "wind_speed_kmh": 15,
    "river_level_m": 3.5,
    "soil_moisture_pct": 55,
    "evaporation_mm": 2.5,
    "pressure_hpa": 1013,
    "season": "summer",
    "land_use": "agricultural"
  }'
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| Data_Cleaning.ipynb | Explore and clean raw data |
| EDA.ipynb | Exploratory data analysis |
| Feature_Engineering.ipynb | Build derived features |
| Model_Training.ipynb | Train and compare models |
| Evaluation.ipynb | Assess model performance |

## License

MIT — see [LICENSE](LICENSE).
