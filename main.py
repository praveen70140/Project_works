import os
import warnings
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# --- ENV/Warning Mute ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
# --- Lazy TF Import ---
from tensorflow.keras.models import load_model

# --- Gemini Integration ---
import google.generativeai as genai

# --- Model Cache ---
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models into the 'models' dict on startup.
    Handles artifact loading and clears cache on shutdown.
    """
    try:
        print("[*] Locking and loading analytical models...")

        # Check for API Key
        if "GEMINI_API_KEY" not in os.environ:
            print(
                "[-] WARNING: GEMINI_API_KEY not found in environment variables. Generative features will fail."
            )
        else:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            print("[+] Gemini configured.")

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
    description="Endpoints for ANN Feedback, DT Sales, RF Ratings, Monthly Sales & Success Prob + Gemini Insight.",
    lifespan=lifespan,
)
origins = ["http://localhost:3000", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],
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
    year: int
    month: int
    sales_qty: float
    sales_amount: float
    City: str
    Cuisine: str


class SalesPredictFeatures(BaseModel):
    year: int
    month: int
    sales_qty: float
    Ratings: float
    City: str
    Cuisine: str


class CityRecommendFeatures(BaseModel):
    year: int
    month: int
    sales_qty: float
    sales_amount: float
    Ratings: float
    Cuisine: str


class SuccessFeatures(BaseModel):
    year: int
    month: int
    sales_qty: float
    sales_amount: float
    Ratings: float
    City: str
    Cuisine: str


class MonthRecommendFeatures(BaseModel):
    year: int
    sales_qty: float
    sales_amount: float
    Ratings: float
    City: str
    Cuisine: str


class MatrixFeatures(BaseModel):
    """Merged Features for Full Grid Prediction"""

    year: int
    sales_qty: float
    sales_amount: float
    Ratings: float
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


# --- Business Logic RF Endpoints ---


@app.post("/predict/rf_rating")
async def predict_rf_rating(features: RatingFeatures):
    if "RATING_RF" not in models:
        raise HTTPException(status_code=503, detail="Rating RF Model unavailable")

    try:
        city_enc = models["LE_CITY"].transform([features.City])[0]
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

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
        raise HTTPException(status_code=422, detail=f"Invalid input data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RF Rating Error: {str(e)}")


@app.post("/predict/rf_monthly_sales")
async def predict_rf_monthly_sales(features: SalesPredictFeatures):
    if "SALES_RF" not in models:
        raise HTTPException(status_code=503, detail="Sales RF Model unavailable")

    try:
        city_enc = models["LE_CITY"].transform([features.City])[0]
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

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
    """
    Scenario 3: Recommends Top 3 Cities.
    (Reverted to Top 3 logic as requested)
    """
    if "CITY_RF" not in models or "LE_CITY" not in models:
        raise HTTPException(status_code=503, detail="City RF Model unavailable")

    try:
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

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

        # Get probabilities
        probs = models["CITY_RF"].predict_proba(data)[0]

        # Sort and slice Top 3
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
    if "SUCCESS_RF" not in models:
        raise HTTPException(status_code=503, detail="Success RF Model unavailable")

    try:
        city_enc = models["LE_CITY"].transform([features.City])[0]
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

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
        success_prob = probs[1] * 100

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
    if "MONTH_RF" not in models:
        raise HTTPException(status_code=503, detail="Month RF Model unavailable")

    try:
        city_enc = models["LE_CITY"].transform([features.City])[0]
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

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
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_months = top3_idx + 1
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


@app.post("/predict/market_matrix")
async def predict_market_matrix(features: MatrixFeatures):
    """
    Merged Scenario: Generates a full probability matrix.
    Iterates all 12 months against the City Model to find the probability
    of EVERY city being the target for EVERY month.
    """
    if "CITY_RF" not in models or "LE_CITY" not in models:
        raise HTTPException(
            status_code=503, detail="City RF Model or Encoder unavailable"
        )

    try:
        # 1. Encode Cuisine Once
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

        # 2. Get all available cities from the encoder classes
        all_cities = models["LE_CITY"].classes_

        # 3. Create a batch DataFrame for 12 months
        batch_data = []
        for m in range(1, 13):
            batch_data.append(
                {
                    "Cuisine_encoded": cuisine_enc,
                    "Ratings": features.Ratings,
                    "sales_qty": features.sales_qty,
                    "sales_amount": features.sales_amount,
                    "year": features.year,
                    "month": m,
                }
            )

        df_batch = pd.DataFrame(batch_data)

        # 4. Run Inference ONCE for the batch
        all_probs = models["CITY_RF"].predict_proba(df_batch)

        # 5. Calculate Global Aggregates
        avg_city_probs = np.mean(all_probs, axis=0)

        global_metrics = {
            city: round(prob * 100, 2) for city, prob in zip(all_cities, avg_city_probs)
        }

        # 6. Construct the Grid
        matrix = {city: {} for city in all_cities}

        for month_idx, month_probs in enumerate(all_probs):
            month_key = f"Month_{month_idx + 1}"
            for city_idx, city_name in enumerate(all_cities):
                prob = month_probs[city_idx]
                matrix[city_name][month_key] = round(prob * 100, 2)

        return {"market_matrix": matrix, "city_global_probabilities": global_metrics}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input data: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Matrix Calculation Error: {str(e)}"
        )


# --- NEW UNIFIED SECTION WITH GEMINI ---


class UnifiedFeatures(BaseModel):
    """Superset Class for the Unified Endpoint"""

    Resturant_Name: str
    Cuisine: str
    Location: str
    City: str
    year: int
    month: int
    sales_qty: float
    sales_amount: float
    Ratings: float


@app.post("/predict/unified")
async def predict_unified(features: UnifiedFeatures):
    """
    Runs all models + Gemini Analysis.
    """
    # Check model availability
    required_models = ["ANN", "DT", "RATING_RF", "SALES_RF", "SUCCESS_RF", "CITY_RF"]
    if any(m not in models for m in required_models):
        raise HTTPException(
            status_code=503, detail="One or more models failed to load."
        )

    results = {}

    try:
        # 1. ANN Feedback
        ann_df = pd.DataFrame(
            [
                {
                    "Resturant_Name": features.Resturant_Name,
                    "Cuisine": features.Cuisine,
                    "Location": features.Location,
                    "City": features.City,
                }
            ]
        )
        ann_encoded = models["X_ENC"].transform(ann_df)
        ann_probs = models["ANN"].predict(ann_encoded, verbose=0)
        ann_index = np.argmax(ann_probs[0])
        results["feedback_prediction"] = {
            "feedback_prediction": models["FB_CLASSES"][ann_index]
        }

        # 2. DT Sales
        dt_df = pd.DataFrame(
            [{"sales_qty": features.sales_qty, "Ratings": features.Ratings}]
        )
        dt_pred = models["DT"].predict(dt_df)
        results["high_sales_prediction"] = {"high_sales_prediction": int(dt_pred[0])}

        # Pre-computation for RFs
        city_enc = models["LE_CITY"].transform([features.City])[0]
        cuisine_enc = models["LE_CUISINE"].transform([features.Cuisine])[0]

        # 3. RF Rating
        rf_rating_df = pd.DataFrame(
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
        rf_rating_pred = models["RATING_RF"].predict(rf_rating_df)
        results["rf_rating_prediction"] = {
            "rf_rating_prediction": float(rf_rating_pred[0])
        }

        # 4. RF Monthly Sales
        rf_sales_df = pd.DataFrame(
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
        rf_sales_pred = models["SALES_RF"].predict(rf_sales_df)
        results["rf_monthly_sales"] = {"rf_sales_prediction": float(rf_sales_pred[0])}

        # 5. RF Success Probability
        rf_success_df = pd.DataFrame(
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
        success_probs = models["SUCCESS_RF"].predict_proba(rf_success_df)[0]
        success_prob_val = success_probs[1] * 100
        results["rf_success_prob"] = {
            "success_probability_percentage": round(float(success_prob_val), 2),
            "is_successful": bool(success_prob_val > 50),
        }

        # 6. Market Matrix (Vectorized)
        all_cities = models["LE_CITY"].classes_
        batch_data = []
        for m in range(1, 13):
            batch_data.append(
                {
                    "Cuisine_encoded": cuisine_enc,
                    "Ratings": features.Ratings,
                    "sales_qty": features.sales_qty,
                    "sales_amount": features.sales_amount,
                    "year": features.year,
                    "month": m,
                }
            )
        df_batch = pd.DataFrame(batch_data)
        all_probs = models["CITY_RF"].predict_proba(df_batch)
        avg_city_probs = np.mean(all_probs, axis=0)
        global_metrics = {
            city: round(prob * 100, 2) for city, prob in zip(all_cities, avg_city_probs)
        }
        matrix = {city: {} for city in all_cities}
        for month_idx, month_probs in enumerate(all_probs):
            month_key = f"Month_{month_idx + 1}"
            for city_idx, city_name in enumerate(all_cities):
                prob = month_probs[city_idx]
                matrix[city_name][month_key] = round(prob * 100, 2)

        results["market_matrix"] = {
            "market_matrix": matrix,
            "city_global_probabilities": global_metrics,
        }

        # --- 7. Gemini AI Analysis ---
        try:
            if "GEMINI_API_KEY" in os.environ:
                # Prepare the prompt
                prompt_text = f"""
                Act as a data-driven business consultant for a restaurant chain. 
                Analyze the following restaurant data and predictive model outputs.
                
                Restaurant Input: {features.dict()}
                
                Model Predictions:
                - Feedback Sentiment: {results["feedback_prediction"]}
                - High Sales Potential: {results["high_sales_prediction"]}
                - Predicted Rating: {results["rf_rating_prediction"]}
                - Monthly Sales Forecast: {results["rf_monthly_sales"]}
                - Success Probability: {results["rf_success_prob"]}
                
                Provide a concise, actionable recommendation (max 3 sentences) on how to improve the business or maintain success. Focus on the relationship between ratings, sales, and cuisine fit for the location.
                """

                model = genai.GenerativeModel("models/gemini-flash-latest")
                gemini_response = await model.generate_content_async(prompt_text)
                results["gemini_recommendation"] = gemini_response.text.strip()
            else:
                results["gemini_recommendation"] = (
                    "API Key missing. AI analysis skipped."
                )

        except Exception as g_ex:
            print(f"[-] Gemini API Error: {g_ex}")
            results["gemini_recommendation"] = (
                "AI analysis failed due to an internal error."
            )

        return results

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Bad Payload: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unified Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("[*] Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
