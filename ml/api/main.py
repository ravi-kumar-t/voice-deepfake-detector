"""
FastAPI Backend for Voice Deepfake Detection Suite.

Endpoints:
- GET  /health   -> Health check and model readiness status
- POST /predict  -> Accepts uploaded audio file (.wav) and returns deepfake prediction JSON
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import torch

# Add 'src' directory to Python path to reuse existing ML modules
import sys
API_DIR = Path(__file__).resolve().parent
ML_DIR = API_DIR.parent
SRC_DIR = ML_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import config
from predict import load_model, predict_audio, predict_audio_segments
from model import VoiceDeepfakeCNNLSTM


# Global model reference
model_instance: VoiceDeepfakeCNNLSTM = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to load ML model on startup.
    """
    global model_instance
    print("=" * 60)
    print("Initializing Voice Deepfake Detection FastAPI Service")
    print(f"Device:          {config.DEVICE}")
    print(f"Checkpoint Path: {config.MODEL_SAVE_PATH}")

    try:
        model_instance = load_model(config.MODEL_SAVE_PATH, config.DEVICE)
        print("[Status] Deepfake CNN-LSTM model loaded successfully!")
    except Exception as e:
        print(f"[Warning] Could not load model checkpoint on startup: {e}")
        model_instance = None

    print("=" * 60)
    yield
    print("Shutting down Voice Deepfake Detection Service...")


app = FastAPI(
    title="Voice Deepfake Detection API",
    description="FastAPI backend for spectral-temporal voice deepfake analysis",
    version="1.0.0",
    lifespan=lifespan
)

# -----------------------------------------------------------------------------
# CORS Middleware Configuration (allowing React / Vite frontend origins)
# -----------------------------------------------------------------------------
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health and Model Status")
async def health_check() -> Dict[str, Any]:
    """
    Returns API health status and model readiness.
    """
    global model_instance
    if model_instance is None and config.MODEL_SAVE_PATH.exists():
        try:
            model_instance = load_model(config.MODEL_SAVE_PATH, config.DEVICE)
        except Exception:
            pass

    return {
        "status": "ok",
        "model_loaded": model_instance is not None
    }


@app.post("/predict", summary="Predict Audio Authenticity")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts an uploaded WAV audio file, preprocesses it, and runs inference.
    
    Returns:
        JSON with filename, prediction (REAL / SYNTHETIC), synthetic_probability, and confidence.
    """
    global model_instance

    if model_instance is None:
        # Attempt lazy reload if not already loaded
        try:
            model_instance = load_model(config.MODEL_SAVE_PATH, config.DEVICE)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model checkpoint not available: {e}"
            )

    # Validate file extension
    original_name = file.filename or "unknown.wav"
    file_ext = Path(original_name).suffix.lower()

    if file_ext not in [".wav", ".wave"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Please upload a valid WAV audio file (.wav)."
        )

    # Save uploaded file to a temporary file
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = Path(tmp.name)

        # Run inference using existing pipeline
        prediction, confidence, prob_fake = predict_audio(temp_file, model_instance, config.DEVICE)

        return {
            "filename": original_name,
            "prediction": prediction,
            "synthetic_probability": round(prob_fake, 4),
            "confidence": round(confidence, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process and analyze audio file: {str(e)}"
        )
    finally:
        # Clean up temporary file to prevent disk leaks
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


@app.post("/predict-segments", summary="Predict Audio Authenticity by Time Segments")
async def predict_segments(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts an uploaded WAV audio file and performs model-based segment-level deepfake analysis (0.5s windows).
    
    Returns:
        JSON with filename and a list of segment results (start, end, prediction, synthetic_probability, confidence).
    """
    global model_instance

    if model_instance is None:
        try:
            model_instance = load_model(config.MODEL_SAVE_PATH, config.DEVICE)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model checkpoint not available: {e}"
            )

    original_name = file.filename or "unknown.wav"
    file_ext = Path(original_name).suffix.lower()

    if file_ext not in [".wav", ".wave"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Please upload a valid WAV audio file (.wav)."
        )

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = Path(tmp.name)

        # Run segment-level inference using existing model and preprocessing
        segments = predict_audio_segments(temp_file, model_instance, config.DEVICE)

        return {
            "filename": original_name,
            "segments": segments
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process and analyze audio segments: {str(e)}"
        )
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
