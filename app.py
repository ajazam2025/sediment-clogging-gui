import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="GUI Tool for  Clogging Prediction in Pervious Concrete",
    page_icon="🌊",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS (compact, colorful, one-screen)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #f4f7fb; }

    .header {
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        color: white;
        margin-bottom: 15px;
    }

    .header h1 { font-size: 32px; margin-bottom: 5px; }
    .header p { font-size: 15px; opacity: 0.9; }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 5px;
        color: #203a43;
    }

    .card {
        background: white;
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        text-align: center;
    }

    .model {
        font-size: 15px;
        font-weight: 600;
        color: #2c5364;
    }

    .value {
        font-size: 26px;
        font-weight: bold;
        color: #0f2027;
    }

    .state {
        font-size: 14px;
        color: #666;
    }

    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        margin-top: 10px;
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
    <div class="header">
        <h1>🌊 Sediment Clogging Predictor</h1>
        <p>Machine learning–based prediction of clogging index & state</p>
    </div>
    """,
    unsafe_allow_html=True
)

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
        "LR ⚖️": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "SVR 🧠": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=100, gamma=0.1))
        ]),
        "XGB 🚀": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ))
        ]),
        "BMA 📊": Pipeline([
            ("scaler", StandardScaler()),
            ("model", BayesianRidge())
        ])
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


models = train_models()

# --------------------------------------------------
# Input section (2-column compact)
# --------------------------------------------------
st.markdown("<div class='section-title'>🔧 Input Parameters</div>", unsafe_allow_html=True)

cols = st.columns(2)
input_data = {}

for i, col in enumerate(X.columns):
    input_data[col] = cols[i % 2].number_input(
        col,
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([input_data])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def clogging_state(index):
    if index < 0.33:
        return "🟢 Low"
    elif index < 0.66:
        return "🟡 Moderate"
    else:
        return "🔴 Severe"


if st.button("🚀 Predict Clogging", use_container_width=True):

    st.markdown("<div class='section-title'>📊 Prediction Results</div>", unsafe_allow_html=True)

    result_cols = st.columns(4)

    for i, (name, model) in enumerate(models.items()):
        index = model.predict(input_df)[0]
        state = clogging_state(index)

        result_cols[i].markdown(
            f"""
            <div class="card">
                <div class="model">{name}</div>
                <div class="value">{index:.3f}</div>
                <div class="state">{state}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
    Developed by Dr. Mudasir Nazeer, Assistant Professor, National Institute of Technology Srianagar
• GUI for Sediment-induced clogging assessment
    </div>
    """,
    unsafe_allow_html=True
)
