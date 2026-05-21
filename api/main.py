from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Retail Sales Forecasting API")

# Load model and scaler
model = joblib.load("models/best_sales_model.pkl")
target_scaler = joblib.load("models/target_scaler.pkl")
model_columns = joblib.load("models/model_columns.pkl") # you'll need this when using all 110 features

class Features(BaseModel):
    sales_lag_7: float
    weekday: int
    is_holiday: bool

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

@app.post("/predict", tags=["Prediction"])
def predict(features: Features):
    # create dict with all model columns set to 0
    input_dict = {col: 0 for col in model_columns}

    # update with the 3 fratures from Streamlit
    input_dict['sales_lag_7'] = features.sales_lag_7
    input_dict['weekday'] = features.weekday # integer 0-6
    input_dict['is_holiday'] = int(features.is_holiday) # True=1, False=0

    
    # Convert to DataFrame in the same order as training
    input_df = pd.DataFrame([input_dict], columns=model_columns)
    prediction_scaled = model.predict(input_df)
    prediction_real = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

    
    return {"predicted_units_sold": float(prediction_real[0][0])}