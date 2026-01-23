"""
PyroCast Fire Spread Model - Configuration
==========================================
Central configuration for the entire fire spread simulation pipeline.

Model Goal: Given an ignition point and environmental conditions, predict
how a wildfire would spread over 1-7 days.
"""

import os
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TFRECORD_DIR = os.path.join(DATA_DIR, "tfrecords")

# Raw data subdirectories
ENVIRONMENTAL_DIR = os.path.join(RAW_DATA_DIR, "environmental")
PROGRESSIONS_DIR = os.path.join(RAW_DATA_DIR, "progressions")
PERIMETERS_DIR = os.path.join(RAW_DATA_DIR, "perimeters")

# Model directories
MODEL_DIR = "C:\\Users\\nonna\\Downloads\\PyroCast\\spread_model\\models\\"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TFRECORD_DIR, 
                 MODEL_DIR, CHECKPOINT_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# GOOGLE EARTH ENGINE
# =============================================================================
GEE_PROJECT = "gleaming-glass-426122-k0"  # Your GEE project ID

# =============================================================================
# DATA COLLECTION PARAMETERS
# =============================================================================

# Historical fire data range
# Using 2000-2024 for maximum dataset size (MODIS/VIIRS coverage)
# GeoMAC historic archives available 2000-2019, WFIGS for 2020+
START_YEAR = 2000
END_YEAR = 2024

# Geographic bounds (Continental US focus, but extendable)
CONUS_BOUNDS = {
    "min_lon": -125.0,
    "max_lon": -66.0,
    "min_lat": 24.0,
    "max_lat": 50.0
}

# Minimum fire size to include (acres)
# Lowered to 100 acres to get 50k+ samples for deep learning
# Small fires have decent VIIRS coverage (375m resolution)
MIN_FIRE_SIZE_ACRES = 100

# Maximum fires to process (set to None for all)
MAX_FIRES = None  # Process all fires (47k+ unique fires expected)

# Target number of fires per year for balanced dataset
TARGET_FIRES_PER_YEAR = None  # No per-year limit (use all available fires)

# =============================================================================
# SPATIAL PARAMETERS
# =============================================================================

# Image dimensions (must match model input)
IMG_SIZE = 256

# Spatial resolution in meters
# 30m is a good balance between detail and computational cost
SPATIAL_RESOLUTION = 30

# Patch size in km (determines the area covered by one 256x256 patch)
# 256 pixels * 30m = 7.68 km per side
PATCH_SIZE_KM = IMG_SIZE * SPATIAL_RESOLUTION / 1000

# Buffer around fire perimeter for context (km)
CONTEXT_BUFFER_KM = 5

# =============================================================================
# TEMPORAL PARAMETERS
# =============================================================================

# Number of days to predict
# Reduced to 3 days to increase training dataset size
# (Most fires last 3-5 days, very few last 7+ days)
PREDICTION_DAYS = 3

# Time step in hours (24 = daily predictions)
TIME_STEP_HOURS = 24

# Minimum days of fire progression required
MIN_PROGRESSION_DAYS = 3

# Maximum days to include in a single sample
# Reduced to 10 for faster processing (we only predict 7 days)
MAX_PROGRESSION_DAYS = 10

# =============================================================================
# INPUT CHANNELS (13 total)
# =============================================================================
"""
Simplified channel mapping for lightweight fire spread model.

Fire spread is LOCAL and physics-driven - we don't need complex features.
Focus on the essentials that directly govern spread rate and direction.

FIRE STATE (1 channel):
0: current_fire_mask  - Fire extent at current timestep (for autoregressive prediction)

TOPOGRAPHY (3 channels) - Controls upslope/downslope spread:
1: elevation          - Normalized elevation (0-4000m -> 0-1)
2: slope              - Slope in degrees (0-60° -> 0-1) [fire spreads faster upslope]
3: aspect             - Aspect in radians (0-2π, circular) [sun exposure affects fuel moisture]

FUEL (3 channels) - Determines fuel availability and flammability:
4: fuel_type          - Anderson 13 fuel model (0-13 normalized) [grass vs timber]
5: fuel_moisture      - Combined fuel moisture (dead 1-hr + live) [key limiting factor]
6: fuel_density       - NDVI proxy for fuel load [more vegetation = more fire]

WEATHER (4 channels) - Primary drivers of fire behavior:
7: wind_u             - Wind east-west component (m/s normalized) [main spread driver]
8: wind_v             - Wind north-south component (m/s normalized)
9: temperature        - Air temperature (normalized) [affects fuel drying]
10: humidity          - Relative humidity (normalized) [moisture in air]

FIRE INDICES (2 channels) - Derived fire weather conditions:
11: fwi               - Fire Weather Index (combined metric)
12: spread_potential  - Expected spread rate from environment
"""

INPUT_CHANNELS = 13
OUTPUT_CHANNELS = 1  # Probability of fire at each pixel

CHANNEL_NAMES = [
    "current_fire_mask",
    "elevation", "slope", "aspect",
    "fuel_type", "fuel_moisture", "fuel_density",
    "wind_u", "wind_v", "temperature", "humidity",
    "fwi", "spread_potential"
]

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Architecture: Lightweight CNN Encoder-Decoder (2-3M parameters)
# Philosophy: Match model complexity to problem complexity
# Fire spread is LOCAL (3x3 kernels sufficient) with medium-range spotting (dilated convs)

MODEL_ARCHITECTURE = "cnn_encoder_decoder"

# Encoder filters progression: 32 -> 64 -> 128
ENCODER_FILTERS = [32, 64, 128]

# Bottleneck filters with dilated convolutions for spotting
BOTTLENECK_FILTERS = 256
DILATION_RATES = [2, 4]  # 2=5km range, 4=10km range at 375m resolution

# Decoder filters (mirror encoder): 128 -> 64
DECODER_FILTERS = [128, 64]

# No dropout needed with proper data augmentation and 2-3M params
DROPOUT_RATE = 0.0

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Batch size (reduce if OOM)
# Physical batch size (per forward pass) - kept small for DirectML allocator
MICRO_BATCH_SIZE = 8  # Actual batch size passed to model
# Effective batch size (via gradient accumulation)
BATCH_SIZE = 64  # Simulated via 8 accumulation steps

# Learning rate
INITIAL_LEARNING_RATE = 2e-4  # Higher for faster convergence with shuffled data
MIN_LEARNING_RATE = 1e-7

# Learning rate schedule
USE_COSINE_ANNEALING = True  # Smoother decay than step-based
COSINE_T_MAX = 10  # Restart every 10 epochs
LR_DECAY_FACTOR = 0.5  # Backup ReduceLROnPlateau
LR_DECAY_PATIENCE = 5

# Training epochs
# Increased with shuffled data for better convergence
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 12  # More patience with stable shuffled data

# Gradient clipping (prevent exploding gradients)
GRADIENT_CLIP_VALUE = 1.0

# Loss function weights
# Higher weight on fire pixels since they're sparse
FIRE_PIXEL_WEIGHT = 10.0
NON_FIRE_PIXEL_WEIGHT = 1.0

# Temporal loss weights (later days are harder to predict)
TEMPORAL_WEIGHTS = [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_90": True,
    "random_crop": False,  # Keep False to maintain spatial alignment
    "brightness_shift": 0.1,
    "wind_perturbation": 0.15,  # Add noise to wind direction/speed
}

# Number of augmented samples per original sample
AUGMENTATION_FACTOR = 4

# =============================================================================
# VIIRS/MODIS FIRE DETECTION PARAMETERS
# =============================================================================

# Confidence threshold for fire detections (0-100)
VIIRS_CONFIDENCE_THRESHOLD = 50

# Use nominal or high confidence only
VIIRS_CONFIDENCE_LEVELS = ["nominal", "high"]

# Maximum cloud cover for usable imagery (%)
MAX_CLOUD_COVER = 30

# =============================================================================
# VALIDATION METRICS
# =============================================================================

# IoU thresholds for evaluation
IOU_THRESHOLDS = [0.25, 0.5, 0.75]

# Probability threshold for binary fire mask
FIRE_PROBABILITY_THRESHOLD = 0.5

# =============================================================================
# HARDWARE & PERFORMANCE
# =============================================================================

# Number of parallel workers for data loading
NUM_WORKERS = 4

# Prefetch buffer size
PREFETCH_BUFFER = 2

# Mixed precision training
USE_MIXED_PRECISION = True

# Maximum GEE concurrent requests
GEE_MAX_CONCURRENT = 10

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# TensorBoard logging
TENSORBOARD_LOG_DIR = os.path.join(LOGS_DIR, "tensorboard")
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# =============================================================================
# SIM2REAL-FIRE FINE-TUNING CONFIGURATION
# =============================================================================
# Dataset: NeurIPS 2024 - 1M simulated + 1K real wildfire scenarios
# Source: https://github.com/TJU-IDVLab/Sim2Real-Fire

SIM2REAL_DATA_DIR = os.path.join(DATA_DIR, "sim2real_fire")
os.makedirs(SIM2REAL_DATA_DIR, exist_ok=True)

# Fine-tuning hyperparameters
FINETUNE_LEARNING_RATE = 1e-5  # 10-20x lower than initial training
FINETUNE_EPOCHS = 10  # Short fine-tuning phase
FINETUNE_FREEZE_ENCODER = True  # Preserve encoder features initially
FINETUNE_UNFREEZE_EPOCH = 5  # Unfreeze encoder after N epochs

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_run_name():
    """Generate a unique run name based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_model_path(run_name=None, suffix=''):
    """Get path for saving model."""
    if run_name is None:
        run_name = get_run_name()
    return os.path.join(MODEL_DIR, f"spread_model_{run_name}{suffix}.keras")

def get_checkpoint_path(run_name=None, suffix=''):
    """Get path for saving checkpoints."""
    if run_name is None:
        run_name = get_run_name()
    return os.path.join(CHECKPOINT_DIR, f"spread_model_{run_name}{suffix}_epoch{{epoch:03d}}.keras")

# Print config summary when imported
if __name__ == "__main__":
    print("=" * 60)
    print("PyroCast Fire Spread Model Configuration")
    print("=" * 60)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Input Channels: {INPUT_CHANNELS}")
    print(f"Output: {PREDICTION_DAYS} days of {IMG_SIZE}x{IMG_SIZE} fire probability maps")
    print(f"Architecture: {MODEL_ARCHITECTURE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Year Range: {START_YEAR}-{END_YEAR}")
    print("=" * 60)
