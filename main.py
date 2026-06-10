from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = joblib.load("house_price_pipeline.pkl")

class HouseFeatures(BaseModel):
    sqft: float
    beds: int
    age: float

@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([{
        "sqft": features.sqft,
        "beds": features.beds,
        "age": features.age
    }])
    prediction = pipe.predict(input_df)[0]
    return {"predicted_price_rs": round(prediction, 2)}