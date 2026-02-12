# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List

# Load the trained model when the app starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create FastAPI app
app = FastAPI(title="ML Model API", version="1.0.0")

# Define input data structure
class PredictionInput(BaseModel):
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776,
                             0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053,
                             8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587,
                             0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
                             0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
            }
        }

# Define output structure
class PredictionOutput(BaseModel):
    prediction: int
    probability: float

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ML Model API is running",
        "model_type": type(model).__name__
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    """Make a prediction"""
    try:
        # Convert input to numpy array (model expects 2D array)
        features = np.array(data.features).reshape(1, -1)
        
        # Check feature count
        if features.shape[1] != 30:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 30 features, got {features.shape[1]}"
            )
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Kubernetes/ECS health check"""
    return {"status": "healthy"}