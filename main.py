import os
import warnings
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=UserWarning)

from tensorflow.keras.models import load_model

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    try:
        print("[*] Loading models...")
        
        models["ANN"] = load_model("classificationd_model.keras")
        models["DT"] = joblib.load("regression_model.joblib")
        models["X_ENC"] = joblib.load("restaurant_encoder.joblib")
        
        fb_enc = joblib.load("feedback_encoder.joblib")
        models["FB_CLASSES"] = fb_enc.categories_[0]
        
        print("[+] Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"[-] FATAL: Missing file. Check your directory. {e}")
    except Exception as e:
        print(f"[-] FATAL: Model load failed. {e}")
    
    yield
    
    models.clear()


app = FastAPI(
    title="Model-API",
    description="Endpoints for ANN Feedback & DT Sales models.",
    lifespan=lifespan
)


class RestaurantFeatures(BaseModel):
    Resturant_Name: str
    Cuisine: str
    Location: str
    City: str

class SalesFeatures(BaseModel):
    sales_qty: float
    Ratings: float


@app.get("/health")
async def health_check():
    if not models:
        return {"status": "offline", "detail": "Models not loaded"}
    return {"status": "online", "models": list(models.keys())}

@app.post("/predict/feedback")
async def predict_feedback(features: RestaurantFeatures):
    if "ANN" not in models:
        raise HTTPException(status_code=503, detail="ANN Model unavailable")

    try:
        data_df = pd.DataFrame([features.dict()])
        data_df = data_df[['Resturant_Name', 'Cuisine', 'Location', 'City']]
        
        encoded_data = models["X_ENC"].transform(data_df)
        
        prediction_probs = models["ANN"].predict(encoded_data, verbose=0)
        predicted_index = np.argmax(prediction_probs[0])
        predicted_class = models["FB_CLASSES"][predicted_index]
        
        return {"feedback_prediction": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/sales")
async def predict_sales(features: SalesFeatures):
    if "DT" not in models:
        raise HTTPException(status_code=503, detail="DT Model unavailable")

    try:
        data_df = pd.DataFrame([features.dict()])
        data_df = data_df[['sales_qty', 'Ratings']]
        
        prediction = models["DT"].predict(data_df)
        
        return {"high_sales_prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("[*] Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
