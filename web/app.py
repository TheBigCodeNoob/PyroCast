import os
# Set Keras backend before any imports
os.environ['KERAS_BACKEND'] = 'tensorflow'

import uvicorn
import numpy as np
import tensorflow as tf
import keras
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
import urllib.request
import shutil
import gc
import json
from datetime import datetime

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

class FeedbackRequest(BaseModel):
    message: str
    user_agent: str = None
    timestamp: str = None

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
            # Use Keras 3.x API to load the model with safe_mode=False for better compatibility
            model = keras.models.load_model(model_path, safe_mode=False)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print("Attempting to load with compile=False and safe_mode=False...")
            try:
                model = keras.models.load_model(model_path, compile=False, safe_mode=False)
                print("Model loaded successfully (without compilation).")
            except Exception as e2:
                print(f"ERROR loading model even without compilation: {e2}")
                model = None
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
    
    # Max points limit - increased from 450 after memory optimizations
    # (reduced parallel workers 10->4, added gc.collect(), smaller inference batches)
    MAX_POINTS = 1500
    
    try:
        # Validate input bounds
        if req.min_lat >= req.max_lat or req.min_lon >= req.max_lon:
            raise HTTPException(status_code=400, detail="Invalid bounds: min values must be less than max values.")
        
        # Log region size for monitoring
        lat_range = req.max_lat - req.min_lat
        lon_range = req.max_lon - req.min_lon
        logger.info(f"Region size: {lat_range:.3f}° x {lon_range:.3f}° (~{lat_range*111:.1f}km x {lon_range*111:.1f}km)")
        
        # Estimate point count and enforce limit
        if lat_range > lon_range:
            step = lat_range / req.grid_density
        else:
            step = lon_range / req.grid_density
        estimated_points = int((lat_range / step) * (lon_range / step)) if step > 0 else 0
        
        if estimated_points > MAX_POINTS:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many analysis points ({estimated_points}). Maximum is {MAX_POINTS}. Reduce grid density or region size."
            )
        
        # 1. Get Grid Data from GEE
        logger.info(f"Fetching GEE data for bounds: {req.min_lat:.4f}, {req.min_lon:.4f} to {req.max_lat:.4f}, {req.max_lon:.4f}")
        logger.info(f"Date: {req.date}, Grid Density: {req.grid_density}")
        
        raw_patches, coordinates, actual_date = gee_service.get_data_from_bounds(
            req.min_lat, req.min_lon, req.max_lat, req.max_lon, req.date, req.grid_density
        )

        if not raw_patches:
            return {"status": "error", "message": "No valid land points found in this region. The area may be entirely ocean or have missing data."}

        # 2. Run Inference with Chunking (prevent memory overflow on large batches)
        logger.info(f"Running inference on {len(raw_patches)} patches...")
        MAX_BATCH_SIZE = 50  # Process max 50 patches at a time to limit peak memory (~200MB per chunk)
        
        probabilities = []
        num_patches = len(raw_patches)
        
        for i in range(0, num_patches, MAX_BATCH_SIZE):
            chunk = raw_patches[i:i + MAX_BATCH_SIZE]
            chunk_probs = model_runner.predict_batch(model, chunk)
            probabilities.extend(chunk_probs)
            
            # Log progress for large batches
            processed = min(i + MAX_BATCH_SIZE, num_patches)
            if num_patches > MAX_BATCH_SIZE:
                logger.info(f"Processed {processed}/{num_patches} patches...")
            
            # Cleanup chunk memory after each iteration
            del chunk
            if processed < num_patches:  # Don't gc on last iteration, we'll do it at the end
                gc.collect()

        # 3. Format Response
        results = []
        for (lat, lon), prob in zip(coordinates, probabilities):
            results.append({
                "lat": lat,
                "lon": lon,
                "prob": float(prob)
            })
        
        # 4. Memory Cleanup - always run for any batch size
        del raw_patches, probabilities, coordinates
        gc.collect()
        tf.keras.backend.clear_session()
        
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

@app.post("/submit_feedback")
async def submit_feedback(req: FeedbackRequest, request: Request):
    """Submit user feedback via Discord webhook or logging."""
    try:
        # Get client info
        client_ip = request.client.host
        timestamp = datetime.utcnow().isoformat()
        
        # Format message
        feedback_data = {
            "message": req.message,
            "timestamp": timestamp,
            "ip": client_ip,
            "user_agent": req.user_agent or "Unknown"
        }
        
        # Try Discord webhook if configured
        discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if discord_webhook_url:
            try:
                # Truncate message and user agent to Discord's limits
                # Discord limits: description=4096, field value=1024
                safe_message = req.message[:2000] if req.message else "No message"
                safe_user_agent = (req.user_agent or "Unknown")[:1000]
                
                discord_payload = {
                    "embeds": [{
                        "title": "Bug Report / Feedback",  # Removed emoji
                        "description": safe_message,
                        "color": 15158332,  # Red color
                        "fields": [
                            {"name": "Timestamp", "value": timestamp, "inline": True},
                            {"name": "IP", "value": client_ip, "inline": True},
                            {"name": "User Agent", "value": safe_user_agent, "inline": False}
                        ]
                    }]
                }
                
                discord_req = urllib.request.Request(
                    discord_webhook_url,
                    data=json.dumps(discord_payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                
                with urllib.request.urlopen(discord_req, timeout=10) as response:
                    if response.status == 204 or response.status == 200:
                        logger.info(f"Feedback sent to Discord: {safe_message[:50]}...")
                        return {"status": "success", "message": "Feedback submitted successfully!"}
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8') if e.fp else "No error details"
                logger.error(f"Failed to send to Discord: HTTP Error {e.code}: {e.reason}")
                logger.error(f"Discord response: {error_body}")
            except Exception as e:
                logger.error(f"Failed to send to Discord: {e}")
        
        # Fallback: just log it prominently
        logger.warning("=" * 60)
        logger.warning(f"USER FEEDBACK RECEIVED:")
        logger.warning(f"Message: {req.message}")
        logger.warning(f"Time: {timestamp}")
        logger.warning(f"IP: {client_ip}")
        logger.warning(f"User Agent: {req.user_agent or 'Unknown'}")
        logger.warning("=" * 60)
        
        return {"status": "success", "message": "Feedback logged successfully!"}
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(static_dir, "index.html"), encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
