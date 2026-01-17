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
    layout="wide"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🌊 Sediment-Induced Clogging Prediction</h1>
    <h3 style='text-align: center; color: grey;'>
    Machine Learning based prediction of clogging index and state
    </h3>
    <hr>
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

# Train–test split (internal, not shown)
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train models (cached)
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


with st.spinner("🔄 Initializing machine learning models..."):
    models = train_models()

st.success("✅ Models ready for prediction")

# --------------------------------------------------
# Prediction tab
# --------------------------------------------------
st.subheader("🔧 Input Parameters")

with st.sidebar:
    st.markdown("### 🔧 Input Parameters")
    input_data = {}

    for col in X.columns:
        input_data[col] = st.number_input(
            col,
            value=float(X[col].mean())
        )

    predict_btn = st.button("🚀 Predict Clogging")

input_df = pd.DataFrame([input_data])

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
def clogging_state(index):
    if index < 0.33:
        return "Low Clogging"
    elif index < 0.66:
        return "Moderate Clogging"
    else:
        return "Severe Clogging"


if predict_btn:
    st.subheader("🔍 Predicted Clogging Index & State")

    cols = st.columns(len(models))

    for i, (name, model) in enumerate(models.items()):
        index = model.predict(input_df)[0]
        state = clogging_state(index)

        cols[i].metric(
            label=name,
            value=f"{index:.3f}",
            delta=state
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
