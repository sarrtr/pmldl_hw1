from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import joblib

app = FastAPI()

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
model_path = os.path.join(MODEL_DIR, "random_forest.pkl")

model = joblib.load(model_path)

class PredictRequest(BaseModel):
    data: List[List[float]]
    # round: Optional[int] = None  

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