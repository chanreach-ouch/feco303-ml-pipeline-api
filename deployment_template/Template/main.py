from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("week4_model.joblib")

app = FastAPI(title="Week 4 Model API")


class PredictionRequest(BaseModel):
    # These match the default model trained in lab/week4/README.md.
    # Replace them only if you retrain with a different feature set.
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float


@app.get("/health")
def health():
    return {"status": "ok", "model_file_exists": MODEL_PATH.exists()}


@app.post("/predict")
def predict(payload: PredictionRequest):
    if not MODEL_PATH.exists():
        return {"error": "Model file not found. Save week4_model.joblib in this folder."}

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([payload.model_dump()])
    prediction = model.predict(row)[0]

    return {"prediction": int(prediction)}
