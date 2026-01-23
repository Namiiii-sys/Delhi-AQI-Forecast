# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Delhi AQI Forecasting System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CLEAN CSS
# -------------------------------------------------
st.markdown("""
<style>
    Base styling 
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    Compact containers
    .compact-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    
    /* Header */
    .dashboard-title {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        font-family: 'Inter', sans-serif;
    }
    
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* AQI Display */
    .aqi-display-compact {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1.5rem;
        border: 2px solid;
        margin-bottom: 1.25rem;
    }
    
    .aqi-value-compact {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.25rem;
        font-family: 'Inter', sans-serif;
    }
    
    .aqi-category-compact {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    /* Small metric cards */
    .metric-card-small {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
        height: 100%;
    }
    
    .metric-label-small {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value-small {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1;
    }
    
    /* Section headers */
    .section-header-compact {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    
    /* Health advisory */
    .advisory-compact {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    
    .advisory-title-compact {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .advisory-message-compact {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    
    /* Confidence bar */
    .confidence-bar {
        background-color: #0f172a;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #334155;
    }
    
    .progress-bar {
        height: 6px;
        background: #334155;
        border-radius: 3px;
        margin: 0.75rem 0;
        position: relative;
    }
    
    .progress-fill-small {
        height: 100%;
        background: linear-gradient(90deg, #dc2626 0%, #d97706 50%, #059669 100%);
        border-radius: 3px;
    }
    
    .progress-marker-small {
        position: absolute;
        top: -4px;
        width: 2px;
        height: 14px;
        background: #f8fafc;
        border-radius: 1px;
    }
    
    /* Chart container */
    .chart-container-compact {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1.25rem;
        border: 1px solid #334155;
    }
    
    /* Simple sidebar */
    .sidebar-simple {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    }
    
    /* Compact list */
    .compact-list {
        padding-left: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .compact-list li {
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    
    /* Footer */
    .footer-compact {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding: 1rem 0;
        border-top: 1px solid #334155;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_model_and_data():
    model_path = BASE_DIR / "models" / "aqi_linear_model.pkl"
    metadata_path = BASE_DIR / "models" / "model_metadata.json"
    data_path = BASE_DIR / "data" / "processed" / "final_model_features.csv"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    return model, metadata, df


def get_aqi_category(aqi):
    """Return AQI category and corresponding color"""
    if aqi <= 50:
        return "Good", "#10b981"
    elif aqi <= 100:
        return "Satisfactory", "#f59e0b"
    elif aqi <= 200:
        return "Moderate", "#ea580c"
    elif aqi <= 300:
        return "Poor", "#dc2626"
    elif aqi <= 400:
        return "Very Poor", "#991b1b"
    else:
        return "Severe", "#7f1d1d"

def get_health_advisory(aqi):
    """Return health advisory based on AQI"""
    if aqi <= 50:
        return {
            "title": "Good Air Quality",
            "message": "Air quality is satisfactory with minimal risk to health.",
            "recommendations": [
                "Ideal for outdoor activities",
                "No restrictions needed",
                "Normal ventilation recommended"
            ],
            "color": "#10b981"
        }
    elif aqi <= 100:
        return {
            "title": "Moderate Air Quality",
            "message": "Air quality is acceptable; sensitive individuals may experience minor effects.",
            "recommendations": [
                "Generally safe for outdoor activities",
                "Sensitive individuals should reduce exertion",
                "Monitor symptoms if vulnerable"
            ],
            "color": "#f59e0b"
        }
    elif aqi <= 200:
        return {
            "title": "Poor Air Quality",
            "message": "Sensitive groups may experience health effects.",
            "recommendations": [
                "Reduce prolonged outdoor exertion",
                "Sensitive groups limit outdoor activities",
                "Keep windows closed during peak hours"
            ],
            "color": "#ea580c"
        }
    elif aqi <= 300:
        return {
            "title": "Very Poor Air Quality",
            "message": "Everyone may experience health effects.",
            "recommendations": [
                "Avoid prolonged outdoor activities",
                "Sensitive groups avoid outdoor exposure",
                "Keep windows closed",
                "Consider using air purifiers"
            ],
            "color": "#dc2626"
        }
    else:
        return {
            "title": "Severe Air Quality",
            "message": "Health warning of emergency conditions.",
            "recommendations": [
                "Avoid all outdoor activities",
                "Remain indoors",
                "Use air purifiers",
                "Wear N95 masks if going outside"
            ],
            "color": "#991b1b"
        }

model, metadata, df = load_model_and_data()
FEATURE_COLS = metadata["feature_columns"]


with st.sidebar:
    st.title("Delhi AQI Forecast")
    st.caption("ML-based next-day AQI prediction")
    
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


# Header
st.markdown("""
<div class="dashboard-title">Delhi Air Quality Index Forecast</div>
<div class="dashboard-subtitle">Next-day AQI prediction using Linear Regression model</div>
""", unsafe_allow_html=True)

latest = df.iloc[-1].copy()

# Make prediction
X_pred = pd.DataFrame([latest[FEATURE_COLS]])
predicted_aqi = float(model.predict(X_pred)[0])
predicted_aqi = round(predicted_aqi, 1)

# Get AQI info
category, color = get_aqi_category(predicted_aqi)

# Compact AQI Display
st.markdown(f"""
<div class="aqi-display-compact" style="border-color: {color};">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="aqi-value-compact" style="color: {color};">{predicted_aqi:.0f}</div>
            <div class="aqi-category-compact" style="color: {color};">{category}</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">
                Predicted AQI for {selected_date.strftime('%B %d, %Y')}
            </div>
        </div>
        <div style="text-align: right;">
            <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.25rem;">
                Model Confidence
            </div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">
                ±32.8 AQI
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

lower_bound = max(0, predicted_aqi - 32.8)
upper_bound = predicted_aqi + 32.8
left_position = min(100, max(0, (predicted_aqi - 32.8) / 500 * 100))

st.markdown(f"""
<div class="confidence-bar">
    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
        <div style="color: #94a3b8; font-size: 0.85rem;">Prediction Range</div>
        <div style="color: #f8fafc; font-size: 0.85rem; font-weight: 600;">±32.8 AQI points</div>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="color: #f8fafc; font-weight: 600;">{lower_bound:.0f}</div>
        <div style="flex-grow: 1; margin: 0 1rem; position: relative;">
            <div class="progress-bar">
                <div class="progress-fill-small" style="width: {(upper_bound/500)*100}%"></div>
                <div class="progress-marker-small" style="left: {left_position}%"></div>
            </div>
        </div>
        <div style="color: #f8fafc; font-weight: 600;">{upper_bound:.0f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header-compact">Current Weather Conditions</div>', unsafe_allow_html=True)

weather_data = [
    {"label": "Temperature", "value": f"{latest.get('temperature_2m_mean', 'N/A'):.1f}°C", "color": "#ef4444"},
    {"label": "Wind Speed", "value": f"{latest.get('wind_speed', 'N/A'):.1f} m/s", "color": "#0ea5e9"},
    {"label": "Humidity", "value": f"{latest.get('humidity_percent', 'N/A'):.0f}%", "color": "#3b82f6"},
    {"label": "Rain Status", "value": "Yes" if latest.get("had_rain_yesterday", 0) else "No", 
     "color": "#10b981" if latest.get("had_rain_yesterday", 0) else "#94a3b8"},
]

weather_cols = st.columns(4)
for idx, weather in enumerate(weather_data):
    with weather_cols[idx]:
        st.markdown(f"""
        <div class="metric-card-small">
            <div class="metric-label-small">{weather['label']}</div>
            <div class="metric-value-small" style="color: {weather['color']};">{weather['value']}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown('<div class="section-header-compact">Health Advisory</div>', unsafe_allow_html=True)

advisory = get_health_advisory(predicted_aqi)

st.markdown(f"""
<div class="advisory-compact" style="border-left-color: {advisory['color']};">
    <div class="advisory-title-compact" style="color: {advisory['color']};">{advisory['title']}</div>
    <div class="advisory-message-compact">{advisory['message']}</div>
    <ul class="compact-list">
        {''.join([f'<li>{rec}</li>' for rec in advisory['recommendations']])}
    </ul>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="compact-card">
    <div style="font-weight: 600; color: #f8fafc; margin-bottom: 1rem; font-size: 0.95rem;">
        Risk Assessment by Group
    </div>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem;">
        <div style="display: flex; justify-content: space-between; padding: 0.5rem; 
                    border-bottom: 1px solid #334155;">
            <span style="color: #cbd5e1; font-size: 0.85rem;">General Population</span>
            <span style="font-weight: 600; color: #f59e0b; font-size: 0.85rem;">Moderate</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.5rem; 
                    border-bottom: 1px solid #334155;">
            <span style="color: #cbd5e1; font-size: 0.85rem;">Children</span>
            <span style="font-weight: 600; color: #dc2626; font-size: 0.85rem;">High</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.5rem; 
                    border-bottom: 1px solid #334155;">
            <span style="color: #cbd5e1; font-size: 0.85rem;">Elderly</span>
            <span style="font-weight: 600; color: #dc2626; font-size: 0.85rem;">High</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.5rem; 
                    border-bottom: 1px solid #334155;">
            <span style="color: #cbd5e1; font-size: 0.85rem;">Respiratory Conditions</span>
            <span style="font-weight: 600; color: #dc2626; font-size: 0.85rem;">High</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# RECENT AQI TREND - COMPACT
# -------------------------------------------------
st.markdown('<div class="section-header-compact">AQI Trend Analysis</div>', unsafe_allow_html=True)

if "target_aqi" in df.columns and len(df) >= 7:
    recent = df.tail(7).copy()
    
    # Create compact chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recent["date"],
        y=recent["target_aqi"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=6, color="#3b82f6"),
        name="Actual AQI",
        hovertemplate="Date: %{x|%b %d}<br>AQI: %{y:.0f}<extra></extra>"
    ))
    
    # Compact layout
    fig.update_layout(
        height=300,
        plot_bgcolor="#1e293b",
        paper_bgcolor="#1e293b",
        font_color="#e2e8f0",
        xaxis=dict(
            title=None,
            gridcolor="#334155",
            showline=True,
            linecolor="#475569",
            tickfont=dict(color="#94a3b8", size=10)
        ),
        yaxis=dict(
            title="AQI",
            gridcolor="#334155",
            showline=True,
            linecolor="#475569",
            tickfont=dict(color="#94a3b8", size=10)
        ),
        showlegend=False,
        hovermode="x unified",
        margin=dict(t=10, b=10, l=50, r=10)
    )
    
    # Chart container
    st.markdown('<div class="chart-container-compact">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Compact trend summary
    trend_change = recent["target_aqi"].iloc[-1] - recent["target_aqi"].iloc[0]
    trend_direction = "increasing" if trend_change > 0 else "decreasing" if trend_change < 0 else "stable"
    trend_color = "#ef4444" if trend_change > 0 else "#10b981" if trend_change < 0 else "#94a3b8"
    
    st.markdown(f"""
    <div class="compact-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.9rem; color: #94a3b8;">7-Day Trend</div>
            <div style="font-weight: 600; color: {trend_color}; font-size: 0.9rem;">
                {abs(trend_change):.0f} points {trend_direction}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# COMPACT FOOTER
# -------------------------------------------------
st.markdown('<div class="footer-compact">', unsafe_allow_html=True)
st.markdown(f"""
<div>
    <div style="margin-bottom: 0.25rem;">
        <span style="color: #f8fafc; font-weight: 600; font-size: 0.9rem;">
            Delhi Air Quality Forecasting System
        </span>
    </div>
    <div style="display: flex; justify-content: center; gap: 1.5rem; margin-bottom: 0.5rem; font-size: 0.8rem;">
        <span>Linear Regression Model</span>
        <span>MAE: 32.8</span>
        <span>+20.6% over baseline</span>
    </div>
    <div style="color: #64748b; font-size: 0.75rem;">
        Forecast period: {selected_date.strftime('%Y-%m-%d')}
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)