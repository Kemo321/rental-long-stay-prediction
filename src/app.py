from __future__ import annotations

import datetime
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import shap

app: FastAPI = FastAPI(title="Nocarz Long-Stay Predictor API")

BASE_PATH: Path = Path(__file__).resolve().parent
MODEL_ADV: Any = joblib.load(BASE_PATH / "models/model_advanced.joblib")
MODEL_BASE: Any = joblib.load(BASE_PATH / "models/model_baseline.joblib")
FEATURES: list[str] = joblib.load(BASE_PATH / "models/model_features.joblib")
LOG_FILE = os.getenv("LOG_PATH", "ab_test_logs.jsonl")
# Initialize SHAP explainer at server start for performance
EXPLAINER = shap.TreeExplainer(MODEL_ADV)

class ListingData(BaseModel):
    accommodates: float
    bathrooms: float
    bedrooms: float
    beds: float
    price_num: float
    number_of_reviews: float
    review_scores_rating: float
    availability_rate: float
    comment_length: float
    has_city_info: float
    has_kitchen: int
    has_washer: int
    has_workspace: int
    amenities_count: int
    is_superhost: int
    lead_time_days: int
    reviews_per_month: float
    calculated_host_listings_count: int
    room_type_Hotel_room: bool = False
    room_type_Private_room: bool = False
    room_type_Shared_room: bool = False


class PredictionResponse(BaseModel):
    """Structured response returned by the prediction endpoint."""

    status: str
    request_id: str
    is_long_stay_probability: float
    top_deciding_factors: list[dict[str, Any]] | None = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: ListingData) -> PredictionResponse:
    request_id = str(uuid.uuid4())

    try:
        # 1. Convert validated request payload to a plain dictionary.
        input_dict: dict[str, Any] = data.model_dump()

        # 2. Map room type fields from API-safe names to training feature names.
        # Pydantic field names cannot contain spaces, while model features do.
        mapping: dict[str, str] = {
            "room_type_Hotel_room": "room_type_Hotel room",
            "room_type_Private_room": "room_type_Private room",
            "room_type_Shared_room": "room_type_Shared room"
        }

        final_input: dict[str, Any] = {}
        for key, value in input_dict.items():
            new_key = mapping.get(key, key)
            final_input[new_key] = value

        # 3. Build a single-row DataFrame for model inference.
        input_df: pd.DataFrame = pd.DataFrame([final_input])

        # 4. Ensure all training features exist; fill missing ones with 0.
        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = 0

        # 5. Reorder columns to match the exact order expected by the model.
        input_df = input_df[FEATURES]

        # 6. A/B experiment: randomly assign baseline (A) or advanced (B) model.
        model_group: str = "A" if random.random() < 0.5 else "B"
        model = MODEL_BASE if model_group == "A" else MODEL_ADV

        # 7. Predict positive-class probability (long stay).
        prediction: float = float(model.predict_proba(input_df)[0, 1])

        # Explainability via SHAP - only for advanced model (group B)
        top_factors: list[dict[str, Any]] = []
        if model_group == "B":
            # Compute SHAP values for the current request
            shap_values = EXPLAINER.shap_values(input_df)
            # Depending on SHAP/XGBoost version, shap_values may be list or array
            vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

            # Pair feature names with their SHAP impact
            feature_impacts = [(FEATURES[i], float(vals[i])) for i in range(len(FEATURES))]
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            for f_name, f_val in feature_impacts[:3]:
                effect = "increases chance of long stay" if f_val > 0 else "decreases chance of long stay"
                top_factors.append({
                    "feature": f_name,
                    "impact_value": round(f_val, 4),
                    "business_interpretation": effect
                })

        # 8. Log request metadata for later A/B evaluation.
        log_entry: dict[str, Any] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "request_id": request_id,
            "model_group": model_group,
            "prediction": prediction,
            "is_long_stay": 1 if prediction >= 0.5 else 0  # Preliminary classification label
        }

        # Append logs to a JSONL file (mounted via Docker volume in deployment).
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return PredictionResponse(
            status="success",
            request_id=request_id,
            is_long_stay_probability=round(prediction, 4),
            top_deciding_factors=top_factors if model_group == "B" else None
        )
        

    except Exception as e:
        # Log runtime errors to container stdout for easier debugging.
        print(f"Błąd przetwarzania: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd wewnętrzny modelu: {str(e)}")