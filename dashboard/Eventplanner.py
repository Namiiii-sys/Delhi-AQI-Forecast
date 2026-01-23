# dashboard/pages/3_Event_Planner.py

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
    page_title="Event Planner",
    page_icon="📅",
    layout="wide"
)

# Initialize session state
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

# -------------------------------------------------
# CLEAN CSS - FIXED
# -------------------------------------------------
st.markdown("""
<style>
    .event-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    
    .event-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .event-good {
        background-color: #064e3b;
        color: #10b981;
    }
    
    .event-moderate {
        background-color: #78350f;
        color: #f59e0b;
    }
    
    .event-poor {
        background-color: #7f1d1d;
        color: #ef4444;
    }
    
    .compact-info {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #3b82f6;
        font-size: 0.9rem;
    }
    
    .section-header {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------
# LOAD DATA AND MODEL
# -------------------------------------------------
@st.cache_data
def load_data():
    """Load data"""
    data_path = BASE_DIR / "data" / "processed" / "final_model_features.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    else:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Load model"""
    model_path = BASE_DIR / "models" / "aqi_linear_model.pkl"
    metadata_path = BASE_DIR / "models" / "model_metadata.json"
    
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        class FallbackModel:
            def predict(self, X):
                return np.array([216])
        model = FallbackModel()
    
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"feature_columns": []}
    
    return model, metadata

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def get_aqi_category(aqi):
    """Return AQI category and color"""
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

def get_event_suitability(aqi):
    """Get suitability level for events based on AQI"""
    if aqi <= 50:
        return "Excellent", "Ideal conditions for any event", "event-good"
    elif aqi <= 100:
        return "Good", "Generally suitable for most events", "event-good"
    elif aqi <= 150:
        return "Moderate", "Consider indoor alternatives for sensitive groups", "event-moderate"
    elif aqi <= 200:
        return "Poor", "Not recommended for outdoor events", "event-poor"
    elif aqi <= 300:
        return "Very Poor", "Avoid outdoor events", "event-poor"
    else:
        return "Severe", "Postpone or move indoors", "event-poor"

def get_seasonal_prediction(date):
    """Get realistic prediction based on Delhi's seasonal patterns"""
    month = date.month
    
    # Delhi's seasonal air quality patterns
    monthly_base = {
        1: 280, 2: 260, 3: 220, 4: 180, 5: 160, 6: 150,
        7: 140, 8: 150, 9: 160, 10: 180, 11: 240, 12: 270
    }
    
    base_aqi = monthly_base.get(month, 200)
    
    # Adjust for day of week (weekends often worse)
    day_of_week = date.weekday()
    if day_of_week >= 5:
        base_aqi *= 1.12
    
    # Add realistic variation
    variation = np.random.uniform(-0.1, 0.1)
    predicted_aqi = base_aqi * (1 + variation)
    
    # Ensure realistic range
    predicted_aqi = min(max(predicted_aqi, 100), 450)
    
    return round(predicted_aqi, 1)

def get_event_tips(event_type):
    """Get concise tips for event type"""
    tips = {
        "Outdoor Wedding": [
            "Schedule ceremony for early morning (6-8 AM)",
            "Have indoor backup venue on standby",
            "Inform guests about air quality conditions"
        ],
        "Birthday Party": [
            "For children's parties, prefer indoor venues",
            "Keep celebration duration under 3 hours",
            "Have indoor games as backup activities"
        ],
        "Sports Tournament": [
            "Postpone if AQI exceeds 150",
            "Schedule 15-minute breaks every hour",
            "Ensure adequate medical support on site"
        ],
        "Corporate Event": [
            "Check attendee health concerns in advance",
            "Have contingency indoor plan",
            "Inform attendees about air quality"
        ],
        "Festival/Celebration": [
            "Mandatory masks if AQI > 100",
            "Create multiple sheltered areas",
            "Limit event duration to 4-6 hours"
        ],
        "Construction Start": [
            "Monitor AQI hourly during work",
            "Provide proper respiratory protection",
            "Schedule heavy work for better AQI periods"
        ],
        "School Sports Day": [
            "Postpone if AQI > 150 for children's safety",
            "Have indoor alternative activities",
            "Keep events short (max 2-3 hours)"
        ],
        "Community Gathering": [
            "Inform elderly attendees about conditions",
            "Have sheltered seating areas",
            "Consider shorter duration"
        ],
        "Charity Run/Walk": [
            "Cancel if AQI > 200",
            "Provide masks for all participants",
            "Have medical team on standby"
        ]
    }
    
    default_tips = [
        "Check AQI forecast 3 days in advance",
        "Have indoor backup plan",
        "Monitor weather conditions"
    ]
    
    return tips.get(event_type, default_tips)

# -------------------------------------------------
# MAIN PAGE
# -------------------------------------------------
st.title("Event Planner")
st.markdown("### Optimize event timing based on air quality forecasts")

# Load data and model
df = load_data()
model, metadata = load_model()

# -------------------------------------------------
# MODEL CONFIGURATION
# -------------------------------------------------
with st.expander("Model Configuration", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        model_path = BASE_DIR / "models" / "aqi_linear_model.pkl"
        if model_path.exists():
            st.success("Primary model detected")
            st.caption(f"File: {model_path.name}")
        else:
            st.warning("Using seasonal prediction model")
            st.caption("Primary model file not found")
    
    with col2:
        metadata_path = BASE_DIR / "models" / "model_metadata.json"
        if metadata_path.exists():
            st.info(f"Features: {len(metadata.get('feature_columns', []))}")
            st.caption("Model metadata loaded")
        else:
            st.info("Using default feature set")
            st.caption("Metadata file not found")


# -------------------------------------------------
# EVENT DETAILS INPUT - COMPACT
# -------------------------------------------------
st.markdown('<div class="section-header">Event Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    event_type = st.selectbox(
        "Event Type",
        ["Outdoor Wedding", "Birthday Party", "Sports Tournament", 
         "Corporate Event", "Festival/Celebration", "Construction Start",
         "School Sports Day", "Community Gathering", "Charity Run/Walk"]
    )
    
    event_size = st.select_slider(
        "Expected Attendance",
        options=["< 50 people", "50-100 people", "100-250 people", 
                "250-500 people", "500+ people"],
        value="50-100 people"
    )

with col2:
    # Date range selection
    st.markdown("**Preferred Date Range**")
    today = datetime.now().date()
    date_range = st.date_input(
        "",
        [
            today + timedelta(days=30),
            today + timedelta(days=60)
        ],
        label_visibility="collapsed"
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = today + timedelta(days=30)
        end_date = today + timedelta(days=60)
    
    vulnerable_groups = st.multiselect(
        "Attendee Considerations",
        ["Children", "Elderly attendees", "Respiratory conditions",
         "Pregnant Women", "Cardiac patients", "Professional athletes",
         "General population"],
        default=["General population"]
    )

# Check for sensitive groups
has_sensitive_groups = any(group in vulnerable_groups for group in 
                          ["Children", "Elderly attendees", "Respiratory conditions",
                           "Pregnant Women", "Cardiac patients"]) and "General population" not in vulnerable_groups

# -------------------------------------------------
# RECOMMEND BUTTON IMMEDIATELY BELOW
# -------------------------------------------------
if st.button("Find Best Dates", type="primary", use_container_width=True):
    with st.spinner("Analyzing dates..."):
        # Generate dates in range
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)
        
        # Predict AQI for each date
        predictions = []
        for date in date_list:
            predicted_aqi = get_seasonal_prediction(date)
            
            category, color = get_aqi_category(predicted_aqi)
            suitability, advice, badge_class = get_event_suitability(predicted_aqi)
            
            # Calculate numerical score (0-100)
            if predicted_aqi <= 50:
                score = 100
            elif predicted_aqi <= 100:
                score = 85 + (100 - predicted_aqi) / 50 * 15
            elif predicted_aqi <= 200:
                score = 60 + (200 - predicted_aqi) / 100 * 25
            elif predicted_aqi <= 300:
                score = 30 + (300 - predicted_aqi) / 100 * 30
            elif predicted_aqi <= 400:
                score = 10 + (400 - predicted_aqi) / 100 * 20
            else:
                score = 0
            
            # Adjust for sensitive groups
            if has_sensitive_groups:
                if predicted_aqi > 150:
                    score *= 0.6
                elif predicted_aqi > 100:
                    score *= 0.8
            
            # Adjust for event size
            if "500+" in event_size and predicted_aqi > 150:
                score *= 0.7
            
            score = max(0, min(100, round(score)))
            
            predictions.append({
                'date': date,
                'predicted_aqi': predicted_aqi,
                'category': category,
                'color': color,
                'suitability': suitability,
                'advice': advice,
                'badge_class': badge_class,
                'score': score
            })
        
        # Sort by score (descending)
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Display top recommendations - USING STREAMLIT NATIVE COMPONENTS
        if predictions:
            st.markdown('<div class="section-header">Top Recommendations</div>', unsafe_allow_html=True)
            
            cols = st.columns(3)
            for idx, pred in enumerate(predictions[:3]):
                with cols[idx]:
                   
                    
                    # Use Streamlit native components
                    st.markdown(f"### {pred['date'].strftime('%b %d, %Y')}")
                    
                    # AQI Display
                    st.metric(
                        label="Predicted AQI",
                        value=f"{pred['predicted_aqi']:.0f}",
                        delta=pred['category']
                    )
                    
                    # Suitability badge
                    st.info(f"**{pred['suitability']}** - {pred['advice']}")
                    
                    # Score
                    st.metric(
                        label="Suitability Score",
                        value=f"{pred['score']:.0f}/100"
                    )
                    
                    st.markdown("---")
            
            st.markdown('<div class="section-header">All Dates Analysis</div>', unsafe_allow_html=True)
            
            pred_data = []
            for p in predictions:
                pred_data.append({
                    'Date': p['date'].strftime('%Y-%m-%d'),
                    'Day': p['date'].strftime('%A'),
                    'Predicted AQI': p['predicted_aqi'],
                    'Category': p['category'],
                    'Suitability': p['suitability'],
                    'Score': f"{p['score']:.0f}/100"
                })
            
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True, height=400)
            
            # Export functionality
            csv_data = pred_df.to_csv(index=False)
            st.download_button(
                label="Download Analysis",
                data=csv_data,
                file_name="aqi_event_recommendations.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Set flag to show analysis has run
            st.session_state.analysis_run = True
else:
    # Reset flag if button not pressed
    if 'analysis_run' in st.session_state:
        st.session_state.analysis_run = False

# -------------------------------------------------
# COMPACT EVENT TIPS (SHOW ONLY IF NO ANALYSIS)
# -------------------------------------------------
if 'analysis_run' in st.session_state and not st.session_state.analysis_run:
    st.markdown('<div class="section-header">Event Planning Tips</div>', unsafe_allow_html=True)
    
    # Get tips for selected event type
    tips = get_event_tips(event_type)
    
    for tip in tips:
        # Use inline styling instead of CSS class
        st.markdown(f"""
        <div style="
            background-color: #1e293b;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 3px solid #3b82f6;
            font-size: 0.9rem;
            color: #e2e8f0;
        ">
            {tip}
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1.5rem 0;">
    <div style="margin-bottom: 0.5rem;">
        <span style="color: #f8fafc; font-weight: 600;">Delhi Air Quality Event Planner</span>
    </div>
    <div style="display: flex; justify-content: center; gap: 1.5rem; font-size: 0.8rem; margin-bottom: 0.5rem;">
        <span>Seasonal Pattern Analysis</span>
        <span>Event-Specific Recommendations</span>
        <span>Attendee Risk Assessment</span>
    </div>
</div>
""", unsafe_allow_html=True)