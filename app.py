
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "heart_disease_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

st.set_page_config(
    page_title="Cardiac Digital Twin",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. PAGE HEADER
# --------------------------------------------------

st.title("❤️ Digital Twin Model for Personalized Cardiac Health Monitoring")

st.caption(
    "Academic prototype | Machine Learning | "
    "Personalized Health Visualization"
)

st.warning(
    "EDUCATIONAL RESEARCH PROTOTYPE ONLY. This application uses a "
    "historical dataset and is not a medical device. Its output is "
    "not a diagnosis or validated individual clinical risk estimate. "
    "Never use it to make medical decisions."
)


# --------------------------------------------------
# 3. LOAD TRAINED MODEL AND METRICS
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error("Trained model not found.")

    st.write(
        "Run the following commands in your project terminal:"
    )

    st.code(
        "python train_model.py\n"
        "streamlit run app.py",
        language="bash"
    )

    st.stop()

try:
    model = joblib.load(MODEL_PATH)

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        metrics = {}

except Exception as exc:
    st.error(f"Could not load the model or metrics: {exc}")
    st.stop()


# --------------------------------------------------
# 4. PATIENT INPUT SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("👤 Patient Profile")

    st.caption(
        "Enter values using the historical dataset's "
        "feature definitions. These are demonstration inputs."
    )

    age = st.slider(
        "Age (years)", 20, 85, 50
    )

    sex = st.selectbox(
        "Sex code",
        options=[1, 0],
        format_func=lambda x: (
            "Male (1)" if x == 1 else "Female (0)"
        )
    )

    cp = st.selectbox(
        "Chest pain type (cp)",
        options=[1, 2, 3, 4]
    )

    trestbps = st.slider(
        "Resting blood pressure (mm Hg)",
        80, 220, 125
    )

    chol = st.slider(
        "Serum cholesterol (mg/dL)",
        100, 600, 240
    )

    fbs = st.selectbox(
        "Fasting blood sugar >120 mg/dL (fbs)",
        options=[0, 1]
    )

    restecg = st.selectbox(
        "Resting ECG code (restecg)",
        options=[0, 1, 2]
    )

    thalach = st.slider(
        "Maximum heart rate achieved (thalach)",
        60, 220, 150
    )

    exang = st.selectbox(
        "Exercise-induced angina (exang)",
        options=[0, 1]
    )

    oldpeak = st.slider(
        "ST depression (oldpeak)",
        0.0, 6.5, 1.0, 0.1
    )

    slope = st.selectbox(
        "ST segment slope code (slope)",
        options=[1, 2, 3]
    )

    ca = st.selectbox(
        "Major vessels code (ca)",
        options=[0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thal code (thal)",
        options=[3, 6, 7]
    )

    update = st.button(
        "🔄 Update Digital Twin",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.caption(
        "This prototype is for academic demonstration, "
        "not medical diagnosis."
    )


# --------------------------------------------------
# 5. PREPARE INPUT DATA
# --------------------------------------------------

row = pd.DataFrame(
    [{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }],
    columns=FEATURES
)


# --------------------------------------------------
# 6. PREDICTION
# --------------------------------------------------

# Streamlit reruns the script after input changes.
# The button confirms an explicit request to refresh.
# The prediction also updates automatically when inputs change.

try:
    prediction = int(model.predict(row)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]

        classes = list(model.classes_)

        if 1 in classes:
            positive_index = classes.index(1)
            probability = float(probabilities[positive_index])

except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()


# --------------------------------------------------
# 7. DIGITAL TWIN HEALTH SNAPSHOT
# --------------------------------------------------

st.subheader("🫀 Personalized Cardiac-Health Snapshot")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Patient Measurements")

    st.metric(
        "Age",
        f"{age} years"
    )

    st.metric(
        "Resting Blood Pressure",
        f"{trestbps} mm Hg"
    )

    st.metric(
        "Serum Cholesterol",
        f"{chol} mg/dL"
    )

    st.metric(
        "Maximum Heart Rate",
        f"{thalach} bpm"
    )

with col2:
    st.markdown("### Model Output")

    # Explicit conditional statements prevent the result
    # from being accidentally rendered as Python source text.
    if prediction == 1:
        st.error(
            "Model class: disease-positive",
            icon="⚠️"
        )
    else:
        st.success(
            "Model class: disease-negative",
            icon="✅"
        )

    if probability is not None:
        st.metric(
            "Model-estimated positive-class probability",
            f"{probability:.1%}"
        )

        st.caption(
            "This is a model score, not a calibrated personal "
            "clinical risk estimate."
        )

    st.info(
        "This experimental classification cannot confirm or rule "
        "out heart disease. Consult a qualified healthcare "
        "professional about health concerns."
    )


# --------------------------------------------------
# 8. SIMULATED HEALTH TREND VISUALIZATION
# --------------------------------------------------

st.divider()

st.subheader("📈 Illustrative Health Trend")

st.caption(
    "SIMULATED DATA: These 14-day trends are generated for "
    "interface demonstration only. They are not actual patient "
    "measurements, wearable data, or evidence of continuous monitoring."
)

rng = np.random.default_rng(17)

dates = pd.date_range(
    end=pd.Timestamp.today().normalize(),
    periods=14,
    freq="D"
)

heart_rate = np.clip(
    75 + np.cumsum(rng.normal(0, 1.8, 14)),
    55,
    120
).round()

systolic_bp = np.clip(
    trestbps + np.cumsum(rng.normal(0, 2.2, 14)),
    90,
    210
).round()

trend = pd.DataFrame({
    "Date": dates,
    "Heart Rate (simulated)": heart_rate,
    "Systolic BP (simulated)": systolic_bp
})

fig = px.line(
    trend,
    x="Date",
    y=[
        "Heart Rate (simulated)",
        "Systolic BP (simulated)"
    ],
    markers=True,
    title="Simulated 14-Day Health Trend"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Value",
    legend_title="Measurement",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 9. MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.subheader("🧪 Held-Out Model Evaluation")

if metrics:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics.get('accuracy', 0):.2%}"
    )

    c2.metric(
        "Precision",
        f"{metrics.get('precision', 0):.2%}"
    )

    c3.metric(
        "Recall",
        f"{metrics.get('recall', 0):.2%}"
    )

    c4.metric(
        "F1 Score",
        f"{metrics.get('f1', 0):.2%}"
    )

    if "roc_auc" in metrics:
        st.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.2%}"
        )

    st.write(
        f"**Dataset:** {metrics.get('dataset', 'Unknown')}"
    )

    st.write(
        f"**Records used:** "
        f"{metrics.get('dataset_rows_used', 'Unknown')}"
    )

    st.write(
        f"**Training samples:** "
        f"{metrics.get('train_samples', 'Unknown')}"
    )

    st.write(
        f"**Test samples:** "
        f"{metrics.get('test_samples', 'Unknown')}"
    )

    st.write(
        "**Confusion matrix** "
        "(rows = actual class; columns = predicted class)"
    )

    cm = metrics.get("confusion_matrix")

    if cm is not None:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

else:
    st.info(
        "Evaluation metrics are not available. "
        "Run python train_model.py first."
    )


# --------------------------------------------------
# 10. PROTOTYPE MODULES
# --------------------------------------------------

st.divider()

st.subheader("⚙️ Prototype Modules")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("#### Patient Profile")
    st.write(
        "Collects selected clinical feature values "
        "for the experimental model."
    )

with m2:
    st.markdown("#### ML Classification")
    st.write(
        "Uses a trained baseline model to classify "
        "the input according to the dataset target."
    )

with m3:
    st.markdown("#### Health Visualization")
    st.write(
        "Displays patient inputs, simulated trends, "
        "and held-out evaluation metrics."
    )


# --------------------------------------------------
# 11. LIMITATIONS AND FOOTER
# --------------------------------------------------

st.divider()

with st.expander("Research Limitations"):
    st.markdown("""
    - The model is an educational baseline, not a medical device.
    - The UCI Cleveland dataset is small and historical.
    - The model output is not a clinically validated diagnosis.
    - The displayed probability is not necessarily calibrated.
    - The simulated trend chart does not use real patient monitoring data.
    - The prototype does not currently connect to physical sensors or wearables.
    - Independent external validation would be needed before any clinical use.
    """)

st.caption(
    "Digital Twin Model for Personalized Cardiac Health Monitoring | "
    "Academic prototype | Not intended for clinical use"
)
