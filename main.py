from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import pandas as pd
import numpy as np

with open("final_used_car_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Used Car Price Prediction API")

origins = [
    "http://localhost",
    "http://localhost:5500",  # your frontend dev server
    "http://127.0.0.1:5500"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CarInput(BaseModel):
    # Numeric features
    yr_mfr: Optional[int] = Field(2020, example=2018)
    kms_run: Optional[int] = Field(62624, example=50000)
    broker_quote: Optional[float] = Field(432204, example=432204)
    original_price: Optional[float] = Field(551035, example=551035)
    emi_starts_from: Optional[float] = Field(10565, example=10565)
    booking_down_pymnt: Optional[float] = Field(68233, example=68233)
    total_owners: Optional[int] = Field(1, example=1)

    # Categorical features
    fuel_type: Optional[str] = Field("petrol", example="petrol")
    transmission: Optional[str] = Field("manual", example="manual")
    make: Optional[str] = Field("maruti", example="maruti")
    model: Optional[str] = Field("swift", example="swift")
    body_type: Optional[str] = Field("hatchback", example="hatchback")
    city: Optional[str] = Field("bengaluru", example="bengaluru")
    registered_city: Optional[str] = Field("bengaluru", example="bengaluru")
    registered_state: Optional[str] = Field("karnataka", example="karnataka")
    car_rating: Optional[str] = Field("great", example="great")
    source: Optional[str] = Field("unknown", example="unknown")

    # Boolean features - Added the missing ones
    assured_buy: Optional[bool] = Field(False)
    warranty_avail: Optional[bool] = Field(False)
    fitness_certificate: Optional[bool] = Field(False)  # Added missing column
    reserved: Optional[bool] = Field(False)             # Added missing column
    is_hot: Optional[bool] = Field(False)               # Added missing column


@app.post("/predict")
def predict_car_price(car: CarInput):
    try:
        # Convert input to DataFrame
        car_dict = car.model_dump()
        
        # Feature Engineering: Car age
        car_dict['car_age'] = 2025 - car_dict.pop('yr_mfr')
        
        # Convert to DataFrame
        input_df = pd.DataFrame([car_dict])
        
        # Debug: Print column names to verify
        print("Input columns:", input_df.columns.tolist())
        print("Input data shape:", input_df.shape)

        # Make prediction
        predicted_price = model.predict(input_df)[0]

        return {"predicted_price": float(predicted_price)}
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
def root():
    return {"message": "Used Car Price Prediction API is running!"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}
