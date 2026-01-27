import os
import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
import urllib.request
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import services
from services.gee_layer import GEEService
from services.model_runner import ModelRunner

app = FastAPI(
    title="PyroCast API",
    description="Wildfire Risk Prediction using Satellite Imagery and ML",
    version="1.0.0"
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (HTML/JS/CSS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- Global State ---
model = None
gee_service = None
model_runner = None

class PredictionRequest(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    date: str
    grid_density: int = 5  # Number of points along the longest edge

@app.on_event("startup")
def load_resources():
    global model, gee_service, model_runner
    
    # 1. Load Model
    model_path = os.path.join(os.path.dirname(__file__), "A-lot-better-post-data-fix_fixed.keras")
    
    # Check if file exists and is valid (not LFS pointer)
    needs_download = True
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        if file_size > 10000000:  # > 10MB means it's the real model
            needs_download = False
            print(f"Model file found ({file_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"Model file is too small ({file_size} bytes) - likely LFS pointer, will download")
    
    # Download model from environment variable URL if needed
    if needs_download:
        model_url = os.getenv('MODEL_URL')
        if model_url:
            print(f"Downloading model from {model_url}...")
            try:
                with urllib.request.urlopen(model_url) as response:
                    with open(model_path, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                print(f"Model downloaded successfully ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")
            except Exception as e:
                print(f"Failed to download model: {e}")
                model_path = None
        else:
            print("WARNING: Model not found and MODEL_URL not set")
            model_path = None
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        try:
            # Try loading normally
            model = tf.keras.models.load_model(model_path)
        except TypeError as e:
            if "quantization_config" in str(e):
                # Keras version mismatch - load with safe_mode=False and compile=False
                print("Detected Keras version mismatch, loading with compatibility mode...")
                model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            else:
                raise e
        except Exception as e:
            print(f"ERROR loading model: {e}")
            model = None
        
        if model:
            print("Model loaded successfully.")
    else:
        print("WARNING: No valid model available")

    # 2. Initialize Services
    gee_service = GEEService()
    model_runner = ModelRunner()

@app.post("/predict_heatmap")
def predict_heatmap(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure best_fire_model.keras exists.")
    
    if gee_service is None:
        raise HTTPException(status_code=500, detail="Google Earth Engine service not initialized.")
    
    try:
        # Validate input bounds
        if req.min_lat >= req.max_lat or req.min_lon >= req.max_lon:
            raise HTTPException(status_code=400, detail="Invalid bounds: min values must be less than max values.")
        
        # Log region size for monitoring (no hard limit)
        lat_range = req.max_lat - req.min_lat
        lon_range = req.max_lon - req.min_lon
        logger.info(f"Region size: {lat_range:.3f}° x {lon_range:.3f}° (~{lat_range*111:.1f}km x {lon_range*111:.1f}km)")
        
        # 1. Get Grid Data from GEE
        logger.info(f"Fetching GEE data for bounds: {req.min_lat:.4f}, {req.min_lon:.4f} to {req.max_lat:.4f}, {req.max_lon:.4f}")
        logger.info(f"Date: {req.date}, Grid Density: {req.grid_density}")
        
        raw_patches, coordinates, actual_date = gee_service.get_data_from_bounds(
            req.min_lat, req.min_lon, req.max_lat, req.max_lon, req.date, req.grid_density
        )

        if not raw_patches:
            return {"status": "error", "message": "No valid land points found in this region. The area may be entirely ocean or have missing data."}

        # 2. Run Inference
        logger.info(f"Running inference on {len(raw_patches)} patches...")
        probabilities = model_runner.predict_batch(model, raw_patches)

        # 3. Format Response
        results = []
        for (lat, lon), prob in zip(coordinates, probabilities):
            results.append({
                "lat": lat,
                "lon": lon,
                "prob": float(prob)
            })
        
        logger.info(f"Analysis complete. Returning {len(results)} predictions.")
            
        return {
            "status": "success", 
            "data": results,
            "actual_date": actual_date
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(static_dir, "index.html"), encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
