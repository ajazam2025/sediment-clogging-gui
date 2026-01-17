import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

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
    Machine Learning based assessment of clogging index and state
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

X_train, X_test, y_train, y_test = train_test_split(
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

    r2_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        r2_scores[name] = r2_score(y_test, model.predict(X_test))

    return models, r2_scores


with st.spinner("🔄 Training ML models..."):
    models, r2_scores = train_models()

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2 = st.tabs(["📊 Model Performance", "🔮 Prediction"])

# --------------------------------------------------
# Tab 1: Performance
# --------------------------------------------------
with tab1:
    st.subheader("📈 Model Performance (R² Score)")

    r2_df = pd.DataFrame.from_dict(
        r2_scores, orient="index", columns=["R² Score"]
    ).sort_values(by="R² Score", ascending=False)

    st.dataframe(
        r2_df.style.format("{:.3f}").background_gradient(cmap="Blues"),
        use_container_width=True
    )

    best_model = r2_df.index[0]
    st.success(f"🏆 Best performing model: **{best_model}**")

# --------------------------------------------------
# Tab 2: Prediction
# --------------------------------------------------
with tab2:
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

    if predict_btn:
        st.subheader("🔍 Predicted Results")

        cols = st.columns(len(models))

        def clogging_state(index):
            if index < 0.33:
                return "Low Clogging"
            elif index < 0.66:
                return "Moderate Clogging"
            else:
                return "Severe Clogging"

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
    ML-based prediction of sediment-induced clogging • Research & academic use
    </p>
    """,
    unsafe_allow_html=True
)
