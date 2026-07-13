import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Smart Industrial IoT Fault Detection",
    page_icon="🏭",
    layout="wide"
)

# =====================================
# Load Model
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "iot_anomaly_model.pkl"

model = joblib.load(MODEL_PATH)
feature_importance = joblib.load(
    BASE_DIR / "models" / "feature_importance.pkl"
)

# =====================================
# Sidebar
# =====================================

st.sidebar.title("🏭 Smart IoT Dashboard")

st.sidebar.markdown("""
### Project

Industrial IoT Fault Detection

### Internship

Machine Learning Internship 2026

### Developed By

Himanshi Arora

### Model

Random Forest

### Status

🟢 Ready
""")

# =====================================
# Main Title
# =====================================

st.title("🏭 Smart Industrial IoT Fault Detection & Predictive Maintenance")

st.markdown(
"""
Monitor industrial sensor values and detect equipment faults
using Machine Learning.
"""
)

st.divider()

# =====================================
# Dashboard Cards
# =====================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Dataset",
    "1000 Records"
)

col2.metric(
    "Features",
    "5 Sensors"
)

col3.metric(
    "Target",
    "3 Classes"
)

st.divider()

# =====================================
# Live Sensor Inputs
# =====================================

st.header("📡 Live Sensor Monitoring")

col1, col2 = st.columns(2)

with col1:

    temperature = st.slider(
        "🌡 Temperature (°C)",
        20.0,
        150.0,
        90.0
    )

    vibration = st.slider(
        "⚙ Vibration (mm/s)",
        0.0,
        2.0,
        0.50
    )

    pressure = st.slider(
        "💨 Pressure (bar)",
        5.0,
        12.0,
        8.5
    )

with col2:

    rms = st.slider(
        "📈 RMS Vibration",
        0.0,
        2.0,
        0.60
    )

    mean_temp = st.slider(
        "🌡 Mean Temperature",
        20.0,
        150.0,
        90.0
    )

st.divider()

predict = st.button(
    "🚨 Detect Fault",
    use_container_width=True
)

# =====================================
# Equipment Health Score
# =====================================

health = 100

if temperature > 100:
    health -= 25

if vibration > 0.8:
    health -= 20

if pressure > 9.5:
    health -= 20

if rms > 0.8:
    health -= 15

if mean_temp > 105:
    health -= 20

health = max(0, health)

st.subheader("💚 Equipment Health")

col1, col2 = st.columns([1,2])

with col1:
    st.metric(
        "Health Score",
        f"{health}%"
    )

with col2:

    if health >= 85:
        st.success("🟢 Excellent")

    elif health >= 65:
        st.warning("🟡 Good")

    elif health >= 40:
        st.warning("🟠 Needs Maintenance")

    else:
        st.error("🔴 Critical")

st.progress(health/100)

st.subheader("📡 Sensor Status")

sensor_df = pd.DataFrame({
    "Sensor":[
        "Temperature",
        "Vibration",
        "Pressure",
        "RMS",
        "Mean Temp"
    ],
    "Current Value":[
        temperature,
        vibration,
        pressure,
        rms,
        mean_temp
    ]
})

st.dataframe(
    sensor_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("🤖 Model Information")

c1,c2,c3 = st.columns(3)

c1.metric("Algorithm","Random Forest")

c2.metric("Classes","3")

c3.metric("Training Samples","1000")
# =====================================
# Prediction
# =====================================

if predict:

    # Create input dataframe
    input_data = pd.DataFrame(
        [[
            vibration,
            temperature,
            pressure,
            rms,
            mean_temp
        ]],
        columns=[
            "Vibration (mm/s)",
            "Temperature (°C)",
            "Pressure (bar)",
            "RMS Vibration",
            "Mean Temp"
        ]
    )

    # Make prediction
    prediction = int(model.predict(input_data)[0])

    # Prediction confidence
    confidence = model.predict_proba(input_data).max() * 100

    st.divider()
    st.subheader("📋 Prediction Result")

    # Normal
    if prediction == 0:

        st.success("🟢 NORMAL")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.info("""
✅ Equipment is operating normally.

No anomaly detected.

Continue regular monitoring.
""")

    # Warning
    elif prediction == 1:

        st.warning("🟡 WARNING")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.warning("""
⚠ Abnormal sensor behaviour detected.

Preventive maintenance is recommended.
""")

    # Critical
    else:

        st.error("🔴 CRITICAL FAULT")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.error("""
🚨 Critical equipment fault detected.

Immediate inspection required.

Possible equipment failure.
""")

        st.balloons()

# =====================================
# Feature Importance
# =====================================

st.divider()
st.subheader("📊 Feature Importance")

feature_names = [
    "Vibration",
    "Temperature",
    "Pressure",
    "RMS Vibration",
    "Mean Temperature"
]


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text="Importance",
    title="Random Forest Feature Importance"
)

fig.update_layout(
    template="plotly_dark",
    height=450,
    margin=dict(l=20, r=20, t=50, b=20)
)

fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

st.plotly_chart(fig, use_container_width=True)
