import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Sediment Clogging Prediction",
    page_icon="🌊",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS (colors & cards)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .header-box {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        text-align: center;
    }
    .model-name {
        font-size: 18px;
        font-weight: bold;
        color: #1f77b4;
    }
    .index-value {
        font-size: 26px;
        font-weight: bold;
    }
    .state {
        font-size: 16px;
        color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="header-box">
        <h1>🌊 Sediment-Induced Clogging Prediction</h1>
        <h4>Machine Learning–based estimation of clogging index and state</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("input file 1.csv")

df = load_data()

TARGET = "Clogging_Index_0_to_1"

X = df.drop(columns=[TARGET, "Clogging_State"], errors="ignore")
y = df[TARGET]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train models
# --------------------------------------------------
@st.cache_resource
def train_models():
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=100, gamma=0.1))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ))
        ]),
        "Bayesian Model Averaging": Pipeline([
            ("scaler", StandardScaler()),
            ("model", BayesianRidge())
        ])
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


models = train_models()

# --------------------------------------------------
# Input section
# --------------------------------------------------
st.markdown("### 🔧 Input Parameters")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(
        col,
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([input_data])

st.write("")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def clogging_state(index):
    if index < 0.33:
        return "Low Clogging"
    elif index < 0.66:
        return "Moderate Clogging"
    else:
        return "Severe Clogging"


if st.button("🚀 Predict Clogging", use_container_width=True):

    st.write("")
    st.markdown("### 🔍 Predicted Clogging Index & State")

    for name, model in models.items():
        index = model.predict(input_df)[0]
        state = clogging_state(index)

        st.markdown(
            f"""
            <div class="card">
                <div class="model-name">{name}</div>
                <div class="index-value">{index:.3f}</div>
                <div class="state">{state}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.success("✅ Prediction completed successfully")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: grey;'>
    ML-based sediment clogging prediction • Research & academic use
    </p>
    """,
    unsafe_allow_html=True
)
