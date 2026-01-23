# Delhi Air Quality Forecasting System

A machine learning–driven system for **predicting Delhi’s Air Quality Index (AQI)** and enabling **informed decision-making for event planning** based on air pollution risk.

The project combines **supervised learning**, **feature engineering**, and **human-centric dashboards** to translate environmental data into actionable insights.

---

##  Machine Learning Overview

- **Problem Type:** Time-series regression  
- **Target Variable:** Next-day AQI  
- **Model Used:** Linear Regression  
- **Evaluation Metric:** Mean Absolute Error (MAE)  
- **MAE:** ~32.8 AQI points  
- **Improvement over baseline:** ~20.6%

### Feature Engineering
The model was trained using engineered features derived from historical AQI and weather data, including:

- Previous day AQI
- Rolling AQI averages (7-day, 14-day)
- Seasonal encodings (month sine/cosine)
- Weather indicators:
  - Wind speed
  - Humidity
  - Rain occurrence
- Temporal context (day of week)

### Model Artifacts
- `aqi_linear_model.pkl` — trained regression model  
- `model_metadata.json` — feature order & configuration (ensures stable inference)

---

##  Application Features

### AQI Forecast Dashboard
- Predicts **next-day AQI**
- Displays:
  - AQI value with category (Good → Severe)
  - Confidence interval using model MAE
  - Health advisories based on pollution level
  - Weather context (wind, humidity, rain)
  - Recent AQI trend visualization

### Event Planner (Decision Support Tool)
A planning assistant that recommends **optimal dates for outdoor events** using **historical seasonal AQI patterns**.

- Ranks dates using a **suitability score (0–100)**
- Adjusts scores based on:
  - Event size
  - Presence of vulnerable groups (children, elderly, respiratory conditions)
- Provides:
  - Top 3 recommended dates
  - Confidence labels (High / Medium / Low)
  - Full date-range analysis
  - CSV export for planning & reporting

> The Event Planner is **pattern-based**, not a real-time forecast — this is explicitly communicated in the UI.

---



## Limitations

- The AQI prediction is **limited to short-term (next-day) forecasting**
- Linear Regression assumes **linear relationships**, which may not fully capture pollution dynamics
- Model performance depends on **historical data quality and feature coverage**
- Event Planner uses **seasonal historical patterns**, not real-time AQI forecasts
- Predictions are **city-level**, not station-specific

---


## Future Enhancements

- Extend prediction horizon to **multi-day forecasting**
- Experiment with **tree-based models** (Random Forest, XGBoost)
- Integrate **real-time weather APIs** for dynamic forecasting
- Add **location-wise AQI predictions** across Delhi
- Implement **probabilistic confidence intervals**
- Deploy dashboard publicly using **Streamlit Cloud**

---


## Additional Notes

- This project prioritizes **interpretability and clarity** over black-box accuracy
- Feature engineering and evaluation were treated as **first-class components**
- Designed to be **portfolio-ready**, reproducible, and easy to extend

---

