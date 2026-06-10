from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
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