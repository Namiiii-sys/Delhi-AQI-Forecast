# Model Evaluation: AQI Prediction (2022–2025)

This section evaluates the performance of machine learning models used to predict **next-day AQI**.  
The goal is to measure whether ML models provide meaningful improvement over a naive baseline.

---

## 1. Baseline Performance

**Baseline assumption:**  
Tomorrow’s AQI = Today’s AQI

- **Baseline MAE:** ~41.3 AQI points

This serves as the minimum benchmark. Any useful model must outperform this.

---

## 2. Model Performance Comparison

### Linear Regression
- **MAE:** ~32.8 AQI points  
- **Improvement over baseline:** ~20.6%

This model successfully beats the baseline and captures temporal and weather-driven patterns.

---

### Random Forest
- **MAE:** ~35.8 AQI points  
- Performs worse than Linear Regression in this setup

The added model complexity does not translate into better generalization.

---

## 3. Error Behavior

- Most prediction errors lie within **0–40 AQI points**
- A small number of days show very large errors (>100 AQI)
- Large errors are concentrated on sudden pollution spikes and abrupt weather changes

This indicates stable performance on normal days but reduced accuracy during extreme events.

---

## 4. Predicted vs Actual Analysis

- Predictions closely follow actual AQI values overall
- High AQI values tend to be **underestimated**
- Low AQI values tend to be **slightly overestimated**

The model smooths extremes rather than predicting sharp peaks or drops.

---

## 5. Time-Series Evaluation

- The model tracks overall AQI trends well
- Directional changes are usually correct
- Sudden spikes or drops are predicted with delay and reduced magnitude

This confirms strong trend-following behavior but limited shock sensitivity.

---

## 6. High Pollution Day Performance

- **Overall MAE:** ~32.8 AQI  
- **MAE for AQI > 300:** ~43 AQI  

Prediction accuracy decreases during severe pollution episodes, which are inherently harder to model.

---

## 7. Model Limitations

- Heavy reliance on lag-based features
- Limited ability to anticipate sudden external events
- No spatial spillover information from neighboring regions

---

## 8. Conclusion

- Linear Regression improves prediction accuracy by ~20% over the baseline
- The model is reliable for trend estimation but less effective during extreme pollution events
- Results align with strong seasonal and weather-driven patterns observed in exploratory analysis

This model provides a solid baseline for AQI forecasting and establishes scope for future improvements.
