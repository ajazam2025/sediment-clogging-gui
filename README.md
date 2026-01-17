# Sediment-Induced Clogging Prediction – ML GUI

A Streamlit-based machine learning application to predict:

- Clogging Index (0–1)
- Derived Clogging State (Low / Moderate / Severe)

## Models Used
- Linear Regression (LR)
- Support Vector Regression (SVR)
- XGBoost (XGB)
- Bayesian Model Averaging (Bayesian Ridge)

## How It Works
1. Predicts clogging index using ML models
2. Converts index into clogging state based on thresholds

## Run Locally
pip install -r requirements.txt  
streamlit run app.py

## Deployment
Deployed using Streamlit Cloud.
