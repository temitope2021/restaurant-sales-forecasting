import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/best_sales_model.pkl")
model_columns = joblib.load("models/model_columns.pkl") # add this

sales_lag_7 = st.number_input("Sales lag 7", value=100)
weekday = st.selectbox("Weekday", range(7))
is_holiday = st.checkbox("Is Holiday")

if st.button("Predict"):
    # Build a dict with all possible features, set missing ones to 0
    input_dict = {
        "sales_lag_7": sales_lag_7,
        "weekday": weekday,
        "is_holiday": int(is_holiday) # make sure it's 0/1 not True/False
    }
    
    # Convert to DataFrame with the exact columns the model expects
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_df)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")

     ### Create a simple 7-day trend chart

         # Stimulate next 7 days by decaying the lag value slightly
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    trend_values = [predicted * (0.98 ** i) for i in range(7)]

    trend_df = pd.DataFrame({'Day': days, 'Predicted Sales': trend_values})
    st.line_chart(trend_df.set_index('Day'))