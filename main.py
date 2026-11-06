"""
FastAPI application for Human Activity Recognition (HAR) using LSTM Deep Learning.
"""
import io
import json
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from .config import ACTIVITIES, SAVED_MODELS_DIR, SEQUENCE_LENGTH, CHANNELS, SAMPLING_RATE_HZ
from .database import init_db, get_predictions, delete_prediction, clear_all_predictions, get_prediction_statistics
from .schemas import PredictionRequest, PredictionResponse, BatchPredictionResponse, RetrainRequest
from .services.model_service import model_service
from .services.dataset_service import dataset_service
from .services.simulator import sensor_simulator
from .services.preprocessor import preprocessor

app = FastAPI(
    title="Human Activity Recognition LSTM API",
    description="Deep Learning API for classifying human physical activities from smartphone sensor data using LSTM neural networks.",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()
    dataset_service.ensure_samples_exist()
    print("Application initialized successfully.")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Human Activity Recognition (HAR) LSTM API",
        "model_loaded": model_service.is_model_loaded(),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "window_timesteps": SEQUENCE_LENGTH,
        "num_classes": len(ACTIVITIES)
    }

@app.get("/api/activities")
def get_activities():
    return {
        "activities": ACTIVITIES,
        "channels": CHANNELS,
        "sequence_length": SEQUENCE_LENGTH,
        "sampling_rate_hz": SAMPLING_RATE_HZ
    }

@app.get("/api/model-info")
def get_model_info():
    return model_service.get_model_info()

@app.get("/api/model/confusion-matrix-image")
def get_confusion_matrix_image():
    cm_path = SAVED_MODELS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        return FileResponse(cm_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Confusion matrix image not yet generated. Please train the model.")

@app.get("/api/model/curves-image")
def get_curves_image():
    curves_path = SAVED_MODELS_DIR / "loss_accuracy_curves.png"
    if curves_path.exists():
        return FileResponse(curves_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Curves image not yet generated. Please train the model.")

@app.get("/api/samples")
def list_samples():
    return {"samples": dataset_service.get_sample_list()}

@app.get("/api/samples/{activity}")
def get_sample(activity: str):
    data = dataset_service.get_sample_data(activity)
    if not data:
        raise HTTPException(status_code=404, detail=f"Sample for activity '{activity}' not found.")
    return data

@app.post("/api/predict", response_model=PredictionResponse)
def predict_sensor_sequence(payload: PredictionRequest):
    try:
        arr = np.array(payload.data, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != len(CHANNELS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D array of shape (N, {len(CHANNELS)}). Received {arr.shape}."
            )
            
        result = model_service.predict_window(arr, source=payload.source or "api_direct")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict/upload", response_model=BatchPredictionResponse)
async def upload_and_predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename or "uploaded_data.csv"
        
        if filename.endswith(".json"):
            json_obj = json.loads(contents.decode("utf-8"))
            if "data" in json_obj:
                raw_matrix = np.array(json_obj["data"], dtype=np.float32)
            else:
                raw_matrix = np.array(json_obj, dtype=np.float32)
        else:
            # CSV or space-delimited text
            try:
                df = pd.read_csv(io.BytesIO(contents))
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), sep=r'\s+', header=None)
                
            raw_matrix = preprocessor.parse_tabular_data(df)
            
        if len(raw_matrix) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        result = model_service.predict_batch(raw_matrix, filename=filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")

@app.get("/api/simulate/timestep")
def simulate_timestep(activity: str = "WALKING", t: float = 0.0):
    readings = sensor_simulator.generate_single_timestep(activity, t)
    return {
        "activity": activity,
        "t": t,
        "readings": readings,
        "channels": CHANNELS
    }

@app.post("/api/simulate/window")
def simulate_and_predict_window(activity: str = "WALKING"):
    window = sensor_simulator.generate_full_window(activity, SEQUENCE_LENGTH)
    arr = np.array(window, dtype=np.float32)
    result = model_service.predict_window(arr, source=f"simulator:{activity}")
    return {
        "simulated_target": activity,
        "prediction": result,
        "raw_window": window
    }

@app.get("/api/history")
def get_prediction_history(limit: int = 50, offset: int = 0):
    items = get_predictions(limit=limit, offset=offset)
    return {"history": items}

@app.delete("/api/history/{prediction_id}")
def delete_history_item(prediction_id: int):
    deleted = delete_prediction(prediction_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    return {"success": True, "deleted_id": prediction_id}

@app.delete("/api/history")
def clear_history():
    count = clear_all_predictions()
    return {"success": True, "records_deleted": count}

@app.get("/api/stats")
def get_system_stats():
    stats = get_prediction_statistics()
    return stats

def background_train_task(epochs: int, batch_size: int, learning_rate: float, use_cache: bool):
    from ..ml.train import run_training
    try:
        run_training(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, use_cache=use_cache)
        model_service.reload()
    except Exception as e:
        print(f"Background training failed: {e}")

@app.post("/api/train")
def trigger_training(payload: RetrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        background_train_task,
        epochs=payload.epochs or 20,
        batch_size=payload.batch_size or 64,
        learning_rate=payload.learning_rate or 0.001,
        use_cache=payload.use_cached_dataset
    )
    return {
        "status": "training_started",
        "message": f"Training job scheduled with {payload.epochs} epochs.",
        "config": payload.dict()
    }

# Mount static frontend build if compiled
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

