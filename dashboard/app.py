
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json


st.set_page_config(
    page_title="Delhi AQI Forecasting System",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* GLOBAL BACKGROUND — applies to all pages */
.stApp {
    background-color: #0f172a;   /* deep bluish */
    color: #e2e8f0;
}

/* Remove default top padding / strip */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp { background-color: #0f172a; color: #e2e8f0; }
.section { background:#1e293b; border-radius:10px; padding:1.25rem; margin-bottom:1rem; border:1px solid #334155; }
.metric-label { color:#94a3b8; font-size:0.8rem; text-transform:uppercase; }
.metric-value { font-size:1.8rem; font-weight:700; }
.aqi-value { font-size:3.5rem; font-weight:800; }
.footer { text-align:center; color:#64748b; font-size:0.8rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent


@st.cache_resource
def load_assets():
    model = pickle.load(open(BASE_DIR / "models/aqi_linear_model.pkl", "rb"))
    metadata = json.load(open(BASE_DIR / "models/model_metadata.json"))
    df = pd.read_csv(BASE_DIR / "data/processed/final_model_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return model, metadata, df

model, metadata, df = load_assets()
FEATURE_COLS = metadata["feature_columns"]


def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "#10b981"
    if aqi <= 100: return "Satisfactory", "#f59e0b"
    if aqi <= 200: return "Moderate", "#ea580c"
    if aqi <= 300: return "Poor", "#dc2626"
    return "Very Poor", "#991b1b"


with st.sidebar:
    st.divider()
    selected_date = st.date_input(
        "Prediction Date",
        min_value=datetime.now().date() + timedelta(days=1),
        max_value=datetime.now().date() + timedelta(days=7),
        value=datetime.now().date() + timedelta(days=1)
    )
    st.divider()
    st.subheader("Model Performance")
    st.metric("MAE", "32.8")
    st.metric("Improvement", "20.6%")
    st.caption("Linear Regression")


st.title("Delhi Air Quality Index Forecast")
st.caption("Next-day AQI prediction using ML")


latest = df.iloc[-1]
X_pred = pd.DataFrame([latest[FEATURE_COLS]])
predicted_aqi = round(float(model.predict(X_pred)[0]), 1)

category, color = get_aqi_category(predicted_aqi)


st.markdown(f"""
<div class="section" style="border-left:6px solid {color};">
    <div class="aqi-value" style="color:{color};">{predicted_aqi:.0f}</div>
    <div style="font-size:1.3rem;font-weight:600;color:{color};">{category}</div>
    <div style="color:#94a3b8;">Predicted AQI for {selected_date.strftime('%B %d, %Y')}</div>
</div>
""", unsafe_allow_html=True)

st.subheader("Weather Snapshot")
cols = st.columns(4)

weather_items = [
    ("Temperature", f"{latest.get('temperature_2m_mean', 0):.1f}°C"),
    ("Wind Speed", f"{latest.get('wind_speed', 0):.1f} m/s"),
    ("Humidity", f"{latest.get('humidity_percent', 0):.0f}%"),
    ("Rain Yesterday", "Yes" if latest.get("had_rain_yesterday", 0) else "No")
]

for col, (label, value) in zip(cols, weather_items):
    with col:
        st.markdown(f"""
        <div class="section">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


st.subheader("Recent AQI Trend (Last 7 Days)")

if "target_aqi" in df.columns:
    recent = df.tail(7)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"],
        y=recent["target_aqi"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=3)
    ))
    fig.update_layout(
        height=300,
        plot_bgcolor="#1e293b",
        paper_bgcolor="#1e293b",
        font_color="#e2e8f0",
        xaxis_title=None,
        yaxis_title="AQI"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="footer">
Delhi AQI Forecasting System • Linear Regression • Educational Project
</div>
""", unsafe_allow_html=True)
