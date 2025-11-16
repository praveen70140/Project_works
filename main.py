import os
import warnings
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

from tensorflow.keras.models import load_model

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    try:
        print("[*] Loading models...")

        # --- Legacy Stack (Original) ---
        models["ANN"] = load_model("classificationd_model.keras")
        models["DT"] = joblib.load("regression_model.joblib")
        models["X_ENC"] = joblib.load("restaurant_encoder.joblib")

        fb_enc = joblib.load("feedback_encoder.joblib")
        models["FB_CLASSES"] = fb_enc.categories_[0]

        # --- Business Logic Stack ---

        # 1. Rating Model (Regressor)
        models["RATING_RF"] = joblib.load("model_ratings.pkl")

        # 2. Sales Model (Regressor)
        models["SALES_RF"] = joblib.load("model_sales.pkl")

        # 3. Success Model (Binary Classifier) - NEW ADDITION
        models["SUCCESS_RF"] = joblib.load("model_success.pkl")

        # Shared Encoders
        models["LE_CITY"] = joblib.load("encoder_city.pkl")
        models["LE_CUISINE"] = joblib.load("encoder_cuisine.pkl")

        print("[+] All systems go. Models loaded.")
    except FileNotFoundError as e:
        print(f"[-] FATAL: Missing artifact. Check your directory. {e}")
    except Exception as e:
        print(f"[-] FATAL: Model load failed. {e}")

    yield

    models.clear()


app = FastAPI(
    title="Model-API-Hybrid",
    description="Endpoints for ANN Feedback, DT Sales, RF Ratings, Monthly Sales & Success Prob.",
    lifespan=lifespan,
)

# --- Schemas ---


class RestaurantFeatures(BaseModel):
    Resturant_Name: str
    Cuisine: str
    Location: str
    City: str


class SalesFeatures(BaseModel):
    sales_qty: float
    Ratings: float


class MarketFeatures(BaseModel):
    year: int
    month: int
    sales_qty: float
    sales_amount: float = 0.0
    Ratings: float = 0.0
    City: str
    Cuisine: str


@app.get("/health")
async def health_check():
    if not models:
        return {"status": "offline", "detail": "Models not loaded"}
    return {"status": "online", "models": list(models.keys())}


# --- Original Endpoints ---


@app.post("/predict/feedback")
async def predict_feedback(features: RestaurantFeatures):
    if "ANN" not in models:
        raise HTTPException(status_code=503, detail="ANN Model unavailable")

    try:
        data_df = pd.DataFrame([features.dict()])
        data_df = data_df[["Resturant_Name", "Cuisine", "Location", "City"]]

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
        data_df = data_df[["sales_qty", "Ratings"]]

        prediction = models["DT"].predict(data_df)

        return {"high_sales_prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Business Logic RF Endpoints ---


@app.post("/predict/rf_rating")
async def predict_rf_rating(features: MarketFeatures):
    """Scenario 1: Predicts Rating based on Sales volume."""
    if "RATING_RF" not in models:
        raise HTTPException(status_code=503, detail="Rating RF Model unavailable")

    try:
        city_enc = models["LE_CITY"].transform([features.City])[0]
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

        # Training Order: year, month, sales_qty, sales_amount, City_encoded, Cuisine_encoded
        data = pd.DataFrame(
            [
                {
                    "year": features.year,
                    "month": features.month,
                    "sales_qty": features.sales_qty,
                    "sales_amount": features.sales_amount,
                    "City_encoded": city_enc,
                    "Cuisine_encoded": cuisine_enc,
                }
            ]
        )

        prediction = models["RATING_RF"].predict(data)
        return {"rf_rating_prediction": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RF Rating Error: {str(e)}")


@app.post("/predict/rf_monthly_sales")
async def predict_rf_monthly_sales(features: MarketFeatures):
    """Scenario 2: Predicts Sales Amount based on Rating."""
    if "SALES_RF" not in models:
        raise HTTPException(status_code=503, detail="Sales RF Model unavailable")

    try:
        city_enc = models["LE_CITY"].transform([features.City])[0]
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

        # Training Order: year, month, sales_qty, Ratings, City_encoded, Cuisine_encoded
        data = pd.DataFrame(
            [
                {
                    "year": features.year,
                    "month": features.month,
                    "sales_qty": features.sales_qty,
                    "Ratings": features.Ratings,
                    "City_encoded": city_enc,
                    "Cuisine_encoded": cuisine_enc,
                }
            ]
        )

        prediction = models["SALES_RF"].predict(data)
        return {"rf_sales_prediction": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RF Sales Error: {str(e)}")


@app.post("/predict/rf_success_prob")
async def predict_rf_success_prob(features: MarketFeatures):
    """Scenario 4: Predicts Success Probability (%) based on all stats."""
    if "SUCCESS_RF" not in models:
        raise HTTPException(status_code=503, detail="Success RF Model unavailable")

    try:
        city_enc = models["LE_CITY"].transform([features.City])[0]
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

        # Training Order: Ratings, sales_qty, sales_amount, City_encoded, Cuisine_encoded, year, month
        data = pd.DataFrame(
            [
                {
                    "Ratings": features.Ratings,
                    "sales_qty": features.sales_qty,
                    "sales_amount": features.sales_amount,
                    "City_encoded": city_enc,
                    "Cuisine_encoded": cuisine_enc,
                    "year": features.year,
                    "month": features.month,
                }
            ]
        )

        # Get probability of class 1 (Success)
        probs = models["SUCCESS_RF"].predict_proba(data)[0]
        success_prob = probs[1] * 100  # Convert to percentage

        return {
            "success_probability_percentage": round(float(success_prob), 2),
            "is_successful": bool(success_prob > 50),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Success Model Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("[*] Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
