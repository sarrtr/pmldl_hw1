from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("random_forest.pkl")

class PredictRequest(BaseModel):
    data: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

@app.get("/health", tags=['health'])  
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = np.array(req.data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return {"predictions": model.predict(X).tolist()}