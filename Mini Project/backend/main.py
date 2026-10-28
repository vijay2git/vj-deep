
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "cnn_best.keras"

ACTIVITIES = [
    "Walking",
    "Walking upstairs",
    "Walking downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

# Load the trained model once when the server starts
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(
    title="ActivitySense AI",
    description="Sensor-based human activity recognition API",
    version="1.0.0",
)

# Allow the React frontend to call the backend locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    # 128 time steps, with 9 sensor channels at each step
    sequence: list[list[float]] = Field(
        ...,
        min_length=128,
        max_length=128,
    )


@app.get("/")
def home():
    return {
        "message": "ActivitySense AI API is running",
        "model": "1D CNN",
        "classes": ACTIVITIES,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    sequence = np.asarray(request.sequence, dtype=np.float32)

    # Validate the nine sensor channels per time step.
    if sequence.shape != (128, 9):
        raise HTTPException(
            status_code=422,
            detail="Expected exactly 128 time steps and 9 sensor channels.",
        )

    if not np.isfinite(sequence).all():
        raise HTTPException(
            status_code=422,
            detail="Sensor values must be finite numbers.",
        )

    # Add batch dimension: (128, 9) -> (1, 128, 9)
    model_input = np.expand_dims(sequence, axis=0)

    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "activity": ACTIVITIES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            activity: float(probabilities[index])
            for index, activity in enumerate(ACTIVITIES)
        },
    }


from pathlib import Path

@app.post("/predict/sample")
def predict_sample():
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "ml"
        / "data"
        / "processed"
        / "har_sequences.npz"
    )

    if not dataset_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Processed HAR dataset was not found.",
        )

    data = np.load(dataset_path)
    sequence = data["X_test"][0]

    probabilities = model.predict(
        np.expand_dims(sequence, axis=0),
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))

    activities = [
        "Walking",
        "Walking upstairs",
        "Walking downstairs",
        "Sitting",
        "Standing",
        "Laying",
    ]

    return {
        "activity": activities[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            activity: float(probability)
            for activity, probability in zip(activities, probabilities)
        },
    }
