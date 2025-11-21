import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(page_title="Traffic Duration Predictor", layout="wide")

# ------------------------------------
# CUSTOM CSS (With Background Image)
# ------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* FULL PAGE BACKGROUND IMAGE */
body {
    background-image: url("time.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Transparent white overlay for readability */
.block-container {
    background: rgba(255, 255, 255, 0.55) !important;
    padding: 20px;
    border-radius: 16px;
}

/* HEADER GRADIENT BOX */
.header-box {
    background: rgba(30, 58, 138, 0.85);
    padding: 40px 20px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

.header-title {
    font-size: 40px;
    font-weight: 700;
    margin: 0;
}

.header-sub {
    font-size: 18px;
    margin-top: 10px;
    opacity: 0.9;
}

/* PREDICTION CARD */
.pred-card {
    background: rgba(255,255,255,0.85);
    border-left: 6px solid #2563eb;
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
}

/* BUTTON */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    padding: 0.8rem 1.4rem;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    width: 100%;
    transition: all 0.25s;
}

div.stButton > button:hover {
    background-color: #1e40af;
    transform: translateY(-2px);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------
# HEADER
# ------------------------------------
st.markdown("""
<div class="header-box">
    <h1 class="header-title">⏱️ Traffic Duration Prediction</h1>
    <p class="header-sub">Predict accurate travel time using traffic, road & weather conditions</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------
# LOAD MODEL
# ------------------------------------
MODEL_PATH = "traffic_duration_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

# ------------------------------------
# WEATHER OPTIONS
# ------------------------------------
weather_conditions = [
    "Clear", "Sunny", "Mostly Sunny", "Partly Sunny", "Partly Cloudy", "Mostly Cloudy",
    "Cloudy", "Overcast", "Fair", "Hazy", "Smoke", "Fog", "Foggy", "Mist", "Misty",
    "Rain", "Light Rain", "Moderate Rain", "Heavy Rain", "Snow", "Light Snow", "Heavy Snow"
]

weather_map = {w: i for i, w in enumerate(weather_conditions)}

# ------------------------------------
# INPUT SECTION
# ------------------------------------
st.markdown("### 🚦 Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    Distance = st.number_input("Distance (mi)", value=5.0)
    Severity = st.slider("Severity (0–4)", 0, 4, 2)
    Temperature = st.number_input("Temperature (F)", value=75.0)

with col2:
    Congestion_Speed = st.number_input("Congestion Speed (mph)", value=20.0)
    StartHour = st.slider("Start Hour", 0, 23, 9)
    Visibility = st.number_input("Visibility (mi)", value=8.0)

with col3:
    DelayTypical = st.number_input("Delay From Typical Traffic (mins)", value=5.0)
    StartDayOfWeek = st.slider("Day of Week (0=Monday)", 0, 6, 2)
    Weather_Conditions = st.selectbox("Weather Conditions", weather_conditions)
    

# DataFrame for prediction
df = pd.DataFrame({
    "Distance(mi)": [Distance],
    "Congestion_Speed": [Congestion_Speed],
    "DelayFromTypicalTraffic(mins)": [DelayTypical],
    "DelayFromFreeFlowSpeed(mins)": [0],  # You can add if needed
    "Severity": [Severity],
    "StartHour": [StartHour],
    "StartDayOfWeek": [StartDayOfWeek],
    "Temperature(F)": [Temperature],
    "Visibility(mi)": [Visibility],
    "Weather_Conditions": [weather_map[Weather_Conditions]]
})

# ------------------------------------
# PREDICTION
# ------------------------------------
st.markdown("### 🔮 Prediction")

if st.button("Predict Duration (mins)"):

    try:
        prediction = model.predict(df)[0][0]

        st.markdown(
            f"""
            <div class="pred-card">
                <h3>🕒 Predicted Duration</h3>
                <h1 style="color:#2563eb;">{prediction:.2f} mins</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
