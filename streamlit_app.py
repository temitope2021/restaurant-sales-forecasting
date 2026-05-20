import streamlit as st

st.title("Restaurant Sales Forecast")



sales_lag_7 = st.number_input("Enter Sales Lag 7", value=100.0, min_value=0.0)

weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
    "Friday": 4, "Saturday": 5, "Sunday": 6
    }
weekday_name = st.selectbox("weekday", options=list(weekday_map.keys()))
weekday=weekday_map[weekday_name] # convert "monday" -> 0
is_holiday = st.checkbox("Is Holiday")

    
    
if st.button("Predict"):
    payload = {"sales_lag_7": sales_lag_7,
               "weekday": weekday, # this is now 0-6
               "is_holiday": is_holiday
              }
    
    # Load the model and predict directly
    import joblib
    model = joblib.load("best_sales_model.pkl")
    
    # Convert payload dict to a list in the right order for your model
    features = [[payload["sales_lag_7"], payload["weekday"], payload["is_holiday"]]]
    prediction = model.predict(features)
    
    st.success(f"Predicted Sales: {prediction[0]:.2f}")

    

            # Create a simple 7-day trend chart
            import pandas as pd
            import matplotlib.pyplot as plt

            # Stimulate next 7 days by decaying the lag value slightly
            days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
            trend_values = [predicted * (0.98 ** i) for i in range(7)]

            trend_df = pd.DataFrame({'Day': days, 'Predicted Sales': trend_values})
            st.line_chart(trend_df.set_index('Day'))
         
        
    else:
            st.error(f"Error: {response.text}")