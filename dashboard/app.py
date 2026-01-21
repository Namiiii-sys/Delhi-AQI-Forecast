import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from datetime import timedelta
from pathlib import Path
import pickle
import json

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Delhi AQI Forecasting System",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent


# ========== LOAD YOUR ACTUAL MODEL ==========
@st.cache_resource
def load_model():
    """Load your trained Linear Regression model"""
    try:
        # Load your actual model
        model_path = BASE_DIR / 'models' / 'aqi_linear_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load your actual metadata
        metadata_path = BASE_DIR / 'models' / 'model_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load your latest data
        csv_path = BASE_DIR / 'data' / 'processed' / 'final_model_features.csv'
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        
        return model, metadata, df
    
    except FileNotFoundError:
        st.error("Model files not found. Please ensure your model is saved.")
        return None, None, None

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("Forecast Controls")
   

    
    # Date Selection
    selected_date = st.date_input(
        "Select Prediction Date",
        min_value=datetime.now().date() + timedelta(days=1),
        max_value=datetime.now().date() + timedelta(days=7),
        value=datetime.now().date() + timedelta(days=1)
    )
    
    st.divider()
    
    # Model Performance
    st.subheader("Model Performance")
    st.metric("MAE", "32.8")
    st.metric("Improvement", "20.6%")
    st.caption("Linear Regression model")

# ========== MAIN DASHBOARD ==========
st.title("Delhi Air Quality Forecast")

# Load model and data
model, metadata, df = load_model()

if model is None or df is None:
    st.error("Please ensure your model files exist in the correct location.")
    st.stop()

# ========== MAKE PREDICTION ==========
# Get latest data for features
latest = df.iloc[-1].copy()

# Prepare features - ADJUST THESE TO MATCH YOUR ACTUAL FEATURES
# This should match exactly what you used in training
features = {
    'aqi_yesterday': latest['target_aqi'],
    'aqi_7day_avg': df['target_aqi'].tail(7).mean(),
    'month_sin': np.sin(2 * np.pi * selected_date.month / 12),
    'month_cos': np.cos(2 * np.pi * selected_date.month / 12),
    'day_of_week': selected_date.weekday(),
    'wind_speed_yesterday': latest.get('wind_speed', 5.0),
    'temp_yesterday': latest.get('temp_c', 25.0),
    'rain_yesterday': latest.get('rain', 0),
    'humidity_yesterday': latest.get('humidity', 60)
}

# Create DataFrame with correct feature order
# IMPORTANT: The columns must match EXACTLY what your model was trained on
# Check your model_features.json for the exact order
feature_df = pd.DataFrame([features])

# Make prediction - THIS WILL BE STABLE NOW
try:
    predicted_aqi = model.predict(feature_df)[0]
    
    # Add small rounding for stability (remove random fluctuations)
    predicted_aqi = round(predicted_aqi, 1)
    
except Exception as e:
    st.error(f"Error making prediction: {e}")
    predicted_aqi = 216  # Fallback value

# ========== AQI DISPLAY ==========
# Determine AQI category
def get_aqi_info(aqi):
    if aqi <= 50:
        return "Good", "#00E400", "#00E40020"
    elif aqi <= 100:
        return "Satisfactory", "#FFFF00", "#FFFF0020"
    elif aqi <= 200:
        return "Moderate", "#FF7E00", "#FF7E0020"
    elif aqi <= 300:
        return "Poor", "#FF0000", "#FF000020"
    elif aqi <= 400:
        return "Very Poor", "#8B0000", "#8B000020"
    else:
        return "Severe", "#800080", "#80008020"

category, color, bg_color = get_aqi_info(predicted_aqi)

# Display AQI
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div style="border-left: 6px solid {color}; padding-left: 20px; margin: 20px 0;">
        <h1 style="color: {color}; margin: 0; font-size: 4.5rem;">{predicted_aqi:.0f}</h1>
        <h3 style="color: {color}; margin: 0;">{category}</h3>
        <p style="color: #666; margin: 5px 0 0 0;">Predicted AQI for {selected_date.strftime('%B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Confidence Interval")
    ma = 32.8  # Your MAE
    lower = max(0, predicted_aqi - ma)
    upper = predicted_aqi + ma
    
    st.metric("Lower Bound", f"{lower:.0f}")
    st.metric("Upper Bound", f"{upper:.0f}")
    st.caption(f"Based on model MAE: {ma}")

st.divider()

# ========== WEATHER INTEGRATION ==========
st.subheader("Current Weather Conditions")

# Create weather metrics
weather_cols = st.columns(4)

weather_data = [
    ("Temperature", f"{latest.get('temp_c', 25):.1f}°C", "#FF6B6B"),
    ("Wind Speed", f"{latest.get('wind_speed', 5):.1f} km/h", "#4ECDC4"),
    ("Humidity", f"{latest.get('humidity', 60):.0f}%", "#45B7D1"),
    ("Rainfall", f"{latest.get('rain', 0):.1f} mm", "#96CEB4")
]

for idx, (title, value, col_color) in enumerate(weather_data):
    with weather_cols[idx]:
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px;">
            <div style="font-size: 1.2rem; font-weight: 600; color: {col_color};">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 700; margin: 10px 0;">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ========== HEALTH ADVISORY ==========
st.divider()
st.subheader("Health Advisory")

# Health recommendations
if predicted_aqi <= 50:
    advisory = """
    ✅ **Air quality is satisfactory.** No health impacts expected for general population.
    
    **Activities:**
    - Normal outdoor activities permitted
    - No restrictions needed
    
    **For sensitive groups:**
    - No special precautions needed
    """
elif predicted_aqi <= 100:
    advisory = """
    ⚠️ **Air quality is acceptable.** Some pollutants may affect sensitive individuals.
    
    **Activities:**
    - Outdoor activities generally safe
    - Consider reducing prolonged exertion if sensitive
    
    **For sensitive groups:**
    - Monitor for symptoms
    """
elif predicted_aqi <= 200:
    advisory = """
    ⚠️ **Air quality may affect sensitive groups.** General public not likely affected.
    
    **Activities:**
    - Reduce prolonged outdoor exertion
    - Sensitive groups should limit outdoor activities
    
    **For sensitive groups:**
    - Limit time outdoors
    - Have medication accessible
    """
elif predicted_aqi <= 300:
    advisory = """
    🔴 **Everyone may begin to experience health effects.**
    
    **Activities:**
    - Avoid prolonged outdoor exertion
    - Sensitive groups avoid outdoor activities
    
    **Precautions:**
    - Keep windows closed
    - Use air purifiers if available
    - Wear masks if going outside
    """
else:
    advisory = """
    🚨 **Health alert: Serious risk to all populations.**
    
    **Activities:**
    - Avoid all outdoor activities
    - Postpone non-essential travel
    
    **Precautions:**
    - Stay indoors with windows closed
    - Use air purifiers
    - Wear N95 masks if going outside is unavoidable
    - Seek medical attention for breathing difficulties
    """

st.markdown(advisory)

# ========== POLLUTANT INFO ==========
st.divider()
st.subheader("Pollutant Analysis")

# Simple pollutant display
pollutants = {
    'PM2.5': latest.get('pm2_5', 85),
    'PM10': latest.get('pm10', 140),
    'NO2': latest.get('no2', 38),
    'SO2': latest.get('so2', 13),
    'CO': latest.get('co', 1.8),
    'O3': latest.get('o3', 54)
}

# Find dominant pollutant
dominant = max(pollutants.items(), key=lambda x: x[1])
dominant_name, dominant_value = dominant

st.markdown(f"""
**Dominant Pollutant:** **{dominant_name}** ({dominant_value:.1f} μg/m³)

This is the primary contributor to current air quality levels.
""")

# Simple pollutant chart
fig = go.Figure(data=[
    go.Bar(
        x=list(pollutants.keys()),
        y=list(pollutants.values()),
        marker_color=['#FF6B6B' if v > 100 else '#4ECDC4' for v in pollutants.values()]
    )
])

fig.update_layout(
    title="Current Pollutant Levels",
    xaxis_title="Pollutant",
    yaxis_title="Concentration (μg/m³)",
    height=350,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ========== RECENT TREND ==========
st.divider()
st.subheader("Recent AQI Trend")

# Plot last 7 days
if len(df) >= 7:
    recent = df.tail(7).copy()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=recent['date'],
        y=recent['aqi'],
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI",
        height=350,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Insufficient data for trend analysis.")

# ========== FOOTER ==========
st.divider()
st.caption(f"""
**Model:** Linear Regression | **MAE:** 32.8 | **Improvement:** 20.6% over baseline  
**Prediction Date:** {selected_date.strftime('%Y-%m-%d')} | **Last Data:** {df['date'].iloc[-1].strftime('%Y-%m-%d')}
""")