import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

st.markdown("""
<style>
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    margin: 1.5rem 0 0.75rem 0;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.25rem;
}

</style>
""", unsafe_allow_html=True)

def get_seasonal_prediction(date):
    monthly_base = {
        1: 280, 2: 260, 3: 220, 4: 180, 5: 160, 6: 150,
        7: 140, 8: 150, 9: 160, 10: 180, 11: 240, 12: 270
    }
    base = monthly_base.get(date.month, 200)

    if date.weekday() >= 5:
        base *= 1.12

    variation = np.random.uniform(-0.15, 0.15)
    return round(min(max(base * (1 + variation), 100), 450), 1)


def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    return "Very Poor"


def get_suitability_score(aqi, sensitive, large_event):
    if aqi <= 50: score = 100
    elif aqi <= 100: score = 85
    elif aqi <= 150: score = 65
    elif aqi <= 200: score = 40
    else: score = 15

    if sensitive and aqi > 100:
        score *= 0.7
    if large_event and aqi > 150:
        score *= 0.7

    return int(max(0, min(100, score)))


def get_confidence(score):
    if score >= 80:
        return "High", "conf-high"
    elif score >= 50:
        return "Medium", "conf-medium"
    else:
        return "Low", "conf-low"


st.title("Event Planner")
st.caption(
    "Recommendations are based on historical seasonal AQI patterns — not real-time forecasts."
)

st.markdown('<div class="section-header">Event Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    event_type = st.selectbox(
        "Event Type",
        [
            "Outdoor Wedding",
            "Birthday Party",
            "Sports Tournament",
            "Corporate Event",
            "Festival / Celebration",
            "Community Gathering"
        ]
    )

    event_size = st.selectbox(
        "Expected Attendance",
        ["< 50", "50–100", "100–250", "250+"]
    )

with col2:
    today = datetime.now().date()
    date_range = st.date_input(
        "Preferred Date Range",
        [today + timedelta(days=30), today + timedelta(days=60)]
    )

    vulnerable = st.multiselect(
        "Vulnerable Groups Present",
        ["Children", "Elderly", "Respiratory Conditions", "Pregnant Women"],
        default=[]
    )

has_sensitive = len(vulnerable) > 0
large_event = event_size == "250+"

if st.button("Find Best Dates", use_container_width=True):
    st.session_state.analysis_run = True

    with st.spinner("Analyzing historical patterns..."):
        start, end = date_range
        days = pd.date_range(start, end)

        rows = []
        for d in days:
            aqi = get_seasonal_prediction(d)
            score = get_suitability_score(aqi, has_sensitive, large_event)
            conf, conf_class = get_confidence(score)

            rows.append({
                "Date": d,
                "Day": d.strftime("%A"),
                "AQI": aqi,
                "Category": get_aqi_category(aqi),
                "Score": score,
                "Confidence": conf,
                "Confidence_Class": conf_class
            })

        df_results = pd.DataFrame(rows).sort_values("Score", ascending=False)

   
    st.markdown('<div class="section-header">Top Recommended Dates</div>', unsafe_allow_html=True)

    top3 = df_results.head(3)
    cols = st.columns(3)

    for idx, row in enumerate(top3.itertuples()):
        with cols[idx]:
            st.metric(
                label=row.Date.strftime("%b %d, %Y"),
                value=int(row.AQI),
                delta=row.Category
            )
            st.markdown(
                f"""
                Suitability Score: **{row.Score}/100**  
                Confidence: <span class="{row.Confidence_Class}">{row.Confidence}</span>
                """,
                unsafe_allow_html=True
            )

   
    st.markdown('<div class="section-header">All Dates Analysis</div>', unsafe_allow_html=True)

    st.dataframe(
        df_results.drop(columns=["Confidence_Class"]),
        use_container_width=True,
        height=420
    )

    st.download_button(
        "Download Recommendations",
        df_results.drop(columns=["Confidence_Class"]).to_csv(index=False),
        file_name="event_aqi_recommendations.csv",
        mime="text/csv",
        use_container_width=True
    )

if not st.session_state.analysis_run:
    st.markdown('<div class="section-header">Planning Tips</div>', unsafe_allow_html=True)
    st.markdown(
        "- Prefer early morning (6–8 AM)\n"
        "- Always keep an indoor backup plan\n"
        "- Avoid large outdoor events if AQI > 150\n"
        "- Inform attendees in advance\n"
    )

st.divider()
st.caption(
    "Delhi Air Quality Event Planner • Pattern-based recommendations • Educational project"
)
