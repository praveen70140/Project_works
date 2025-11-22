import os
import warnings
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- ENV/Warning Mute ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

# --- Lazy TF Import ---
from tensorflow.keras.models import load_model

# --- Model Cache ---
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models into the 'models' dict on startup.
    Handles artifact loading and clears cache on shutdown.
    """
    try:
        print("[*] Locking and loading models...")

        models["ANN"] = load_model("classificationd_model.keras")
        models["DT"] = joblib.load("regression_model.joblib")
        models["X_ENC"] = joblib.load("restaurant_encoder.joblib")

        fb_enc = joblib.load("feedback_encoder.joblib")
        models["FB_CLASSES"] = fb_enc.categories_[0]

        models["RATING_RF"] = joblib.load("model_ratings.pkl")
        models["SALES_RF"] = joblib.load("model_sales.pkl")
        models["SUCCESS_RF"] = joblib.load("model_success.pkl")
        models["CITY_RF"] = joblib.load("model_city.pkl")
        models["MONTH_RF"] = joblib.load("model_month.pkl")

        models["LE_CITY"] = joblib.load("encoder_city.pkl")
        models["LE_CUISINE"] = joblib.load("encoder_cuisine.pkl")

        print(f"[+] All systems go. {len(models)} artifacts loaded.")
    except FileNotFoundError as e:
        print(f"[-] FATAL: Missing artifact. Check your directory. {e}")
    except Exception as e:
        print(f"[-] FATAL: Model load failed. {e}")

    yield

    # --- Shutdown ---
    print("[*] Clearing model cache...")
    models.clear()


app = FastAPI(
    title="Model-API-Hybrid",
    description="Endpoints for ANN Feedback, DT Sales, RF Ratings, Monthly Sales & Success Prob.",
    lifespan=lifespan,
)


class RestaurantFeatures(BaseModel):
    Resturant_Name: str
    Cuisine: str
    Location: str
    City: str


class SalesFeatures(BaseModel):
    sales_qty: float
    Ratings: float


class RatingFeatures(BaseModel):
    """Features to predict RF_RATING (Scenario 1)"""

    year: int
    month: int
    sales_qty: float
    sales_amount: float
    City: str
    Cuisine: str


class SalesPredictFeatures(BaseModel):
    """Features to predict RF_MONTHLY_SALES (Scenario 2)"""

    year: int
    month: int
    sales_qty: float
    Ratings: float
    City: str
    Cuisine: str


class CityRecommendFeatures(BaseModel):
    """Features to predict RF_CITY_RECOMMEND (Scenario 3)"""

    year: int
    month: int
    sales_qty: float
    sales_amount: float
    Ratings: float
    Cuisine: str


class SuccessFeatures(BaseModel):
    """Features to predict RF_SUCCESS_PROB (Scenario 4)"""

    year: int
    month: int
    sales_qty: float
    sales_amount: float
    Ratings: float
    City: str
    Cuisine: str


class MonthRecommendFeatures(BaseModel):
    """Features to predict RF_MONTH_RECOMMEND (Scenario 5)"""

    year: int
    sales_qty: float
    sales_amount: float
    Ratings: float
    City: str
    Cuisine: str


# --- Core Endpoints ---


@app.get("/health")
async def health_check():
    """Health check to verify model loading."""
    if not models:
        raise HTTPException(
            status_code=503, detail="Models are offline or failed to load"
        )
    return {"status": "online", "models_loaded": list(models.keys())}


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


# --- Business Logic RF Endpoints (Refactored) ---


@app.post("/predict/rf_rating")
async def predict_rf_rating(features: RatingFeatures):
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

    except ValueError as e:
        # Catch bad input (e.g., "City" or "Cuisine" not in encoder)
        raise HTTPException(status_code=422, detail=f"Invalid input data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RF Rating Error: {str(e)}")


@app.post("/predict/rf_monthly_sales")
async def predict_rf_monthly_sales(features: SalesPredictFeatures):
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

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RF Sales Error: {str(e)}")


@app.post("/predict/rf_city_recommend")
async def predict_rf_city_recommend(features: CityRecommendFeatures):
    """Scenario 3: Recommends Top 3 Cities based on market data."""
    if "CITY_RF" not in models or "LE_CITY" not in models:
        raise HTTPException(
            status_code=503, detail="City RF Model or Encoders unavailable"
        )

    try:
        # Note: features.City is NOT used here, as it's the target.
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

        # Training Order: Cuisine_encoded, Ratings, sales_qty, sales_amount, year, month
        data = pd.DataFrame(
            [
                {
                    "Cuisine_encoded": cuisine_enc,
                    "Ratings": features.Ratings,
                    "sales_qty": features.sales_qty,
                    "sales_amount": features.sales_amount,
                    "year": features.year,
                    "month": features.month,
                }
            ]
        )

        probs = models["CITY_RF"].predict_proba(data)[0]

        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_cities = models["LE_CITY"].inverse_transform(top3_idx)
        top3_probs = (probs[top3_idx] * 100).round(2)

        return {
            "top_3_recommendations": [
                {"city": city, "probability_percent": prob}
                for city, prob in zip(top3_cities, top3_probs)
            ]
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"City Model Error: {str(e)}")


@app.post("/predict/rf_success_prob")
async def predict_rf_success_prob(features: SuccessFeatures):
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

        probs = models["SUCCESS_RF"].predict_proba(data)[0]
        success_prob = probs[1] * 100  # Prob of class 1 (Success)

        return {
            "success_probability_percentage": round(float(success_prob), 2),
            "is_successful": bool(success_prob > 50),
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Success Model Error: {str(e)}")


@app.post("/predict/rf_month_recommend")
async def predict_rf_month_recommend(features: MonthRecommendFeatures):
    """Scenario 5: Recommends Top 3 Months based on market data."""
    if "MONTH_RF" not in models:
        raise HTTPException(status_code=503, detail="Month RF Model unavailable")

    try:
        # Note: features.month is NOT used here, as it's the target.
        city_enc = models["LE_CITY"].transform([features.City])[0]
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

        # Training Order: Ratings, sales_qty, sales_amount, City_encoded, Cuisine_encoded, year
        data = pd.DataFrame(
            [
                {
                    "Ratings": features.Ratings,
                    "sales_qty": features.sales_qty,
                    "sales_amount": features.sales_amount,
                    "City_encoded": city_enc,
                    "Cuisine_encoded": cuisine_enc,
                    "year": features.year,
                }
            ]
        )

        probs = models["MONTH_RF"].predict_proba(data)[0]

        # Model predicts classes 1-12. Probs array is 0-indexed.
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_months = top3_idx + 1  # Convert 0-11 index to 1-12 month
        top3_probs = (probs[top3_idx] * 100).round(2)

        return {
            "top_3_month_recommendations": [
                {"month": int(month), "probability_percent": prob}
                for month, prob in zip(top3_months, top3_probs)
            ]
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Month Model Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("[*] Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
