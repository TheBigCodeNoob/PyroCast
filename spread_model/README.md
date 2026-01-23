# PyroCast Fire Spread Simulation Model

## Overview

This model answers the question: **"If a wildfire started right here tomorrow, how bad would it get in a day? What about 2 days? What about a week?"**

The Fire Spread Simulation Model predicts how a wildfire would spread over 1-7 days given:
- An ignition point (where the fire starts)
- Environmental conditions (terrain, fuel, weather)
- Current fire state (for ongoing fires)

Unlike the risk prediction model (which predicts IF a fire will occur), this model predicts HOW a fire spreads once ignited.

---

## Architecture

### Model Type: ConvLSTM U-Net with Attention

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT (256×256×19)                       │
│   Environmental Data + Ignition Mask + Previous Fire State  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   SPATIAL ENCODER (U-Net)                    │
│   Conv Blocks with CBAM Attention + Skip Connections         │
│   64 → 128 → 256 → 512 filters                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 TEMPORAL PROCESSOR (ConvLSTM)                │
│   Models fire spread dynamics over 7 days                    │
│   Captures how fire behavior evolves temporally              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   SPATIAL DECODER (U-Net)                    │
│   Generates daily fire probability maps                      │
│   Uses skip connections for fine-grained detail              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT (7×256×256×1)                       │
│   Fire probability maps for days 1, 2, 3, 4, 5, 6, 7        │
└─────────────────────────────────────────────────────────────┘
```

---

## Input Channels (19 total)

### Fire State (2 channels)
| Channel | Name | Description |
|---------|------|-------------|
| 0 | `ignition_mask` | Binary mask of fire ignition point(s) |
| 1 | `previous_fire_mask` | Fire extent from previous day |

### Topography (4 channels)
| Channel | Name | Description |
|---------|------|-------------|
| 2 | `elevation` | Normalized elevation (0-4000m → 0-1) |
| 3 | `slope` | Slope in degrees (0-45° → 0-1) |
| 4 | `aspect_sin` | Sin of aspect angle (circular encoding) |
| 5 | `aspect_cos` | Cos of aspect angle |

### Fuel & Vegetation (4 channels)
| Channel | Name | Description |
|---------|------|-------------|
| 6 | `fuel_type` | LANDFIRE 40 fuel model (normalized) |
| 7 | `ndvi` | Vegetation Index (density) |
| 8 | `fuel_moisture_live` | Live fuel moisture content |
| 9 | `fuel_moisture_dead` | Dead fuel moisture (1-hr timelag) |

### Weather (6 channels)
| Channel | Name | Description |
|---------|------|-------------|
| 10 | `temperature` | Air temperature (normalized) |
| 11 | `humidity` | Relative humidity (normalized) |
| 12 | `wind_speed` | Wind speed magnitude |
| 13 | `wind_dir_sin` | Sin of wind direction |
| 14 | `wind_dir_cos` | Cos of wind direction |
| 15 | `precipitation` | Precipitation amount |

### Fire Behavior Indices (3 channels)
| Channel | Name | Description |
|---------|------|-------------|
| 16 | `erc` | Energy Release Component |
| 17 | `bi` | Burning Index |
| 18 | `sc` | Spread Component |

---

## Output

- **Shape**: `(7, 256, 256, 1)`
- **Interpretation**: Each pixel contains the probability (0-1) of fire presence on that day
- **Resolution**: ~30m per pixel (7.68km × 7.68km coverage)

---

## Data Pipeline

### Step 1: Fetch Fire Perimeters
```bash
python 01_fetch_fire_perimeters.py
```
- Downloads historical fire perimeter data from NIFC/MTBS (2017-2024)
- Filters fires ≥1,000 acres with valid spatial data
- Outputs: `fire_catalog.csv` + individual GeoJSON perimeters

### Step 2: Fetch Daily Progressions
```bash
python 02_fetch_daily_progressions.py
```
- Gets VIIRS/MODIS active fire detections for each historical fire
- Reconstructs daily fire progression masks
- Outputs: `progression_*.json` files with daily fire extents

### Step 3: Fetch Environmental Data
```bash
python 03_fetch_environmental_data.py
```
- Extracts terrain, fuel, and weather data from Google Earth Engine
- Creates aligned spatial data for each fire event
- Outputs: `env_*.json` files with 17-channel environmental stacks

### Step 4: Build Dataset
```bash
python 04_build_dataset.py
```
- Combines all data into TFRecord format
- Applies data augmentation (flips, rotations, wind perturbation)
- Creates train/val/test splits
- Outputs: `spread_train.tfrecord`, `spread_val.tfrecord`, `spread_test.tfrecord`

### Step 5: Train Model
```bash
python 05_train_model.py
```
- Builds ConvLSTM U-Net architecture
- Trains with weighted loss (fire pixels are sparse)
- Uses early stopping and learning rate scheduling
- Outputs: Trained model + training logs

### Step 6: Evaluate Model
```bash
python 06_evaluate_model.py
```
- Computes IoU, Dice, Precision, Recall at multiple thresholds
- Generates per-day performance analysis
- Creates visualizations of predictions vs ground truth
- Outputs: Evaluation report + visualization images

---

## Training Data

### Target Dataset Size
- **2,000+** unique fire events (2017-2024)
- **3-14 days** of progression data per fire
- **~50,000+** training samples after augmentation

### Data Sources
| Source | Data Type | Resolution |
|--------|-----------|------------|
| NIFC/MTBS | Fire perimeters | Vector |
| NASA FIRMS | VIIRS active fire | 375m |
| NASA FIRMS | MODIS active fire | 1km |
| SRTM | Elevation | 30m |
| LANDFIRE | Fuel models | 30m |
| Sentinel-2 | NDVI | 10-20m |
| GRIDMET | Weather | 4km |
| GRIDMET | Fire indices | 4km |

---

## Requirements

### Python Dependencies
```
tensorflow>=2.12.0
earthengine-api>=0.1.350
pandas>=2.0.0
geopandas>=0.13.0
numpy>=1.24.0
matplotlib>=3.7.0
shapely>=2.0.0
requests>=2.31.0
```

### External Requirements
1. **Google Earth Engine Account**: Required for environmental data
   ```bash
   earthengine authenticate
   ```

2. **NASA FIRMS API Key** (optional, for faster data fetching):
   - Get free key at: https://firms.modaps.eosdis.nasa.gov/api/
   - Update `FIRMS_API_KEY` in `02_fetch_daily_progressions.py`

3. **GPU** (recommended): Training is slow on CPU
   - Minimum: NVIDIA GPU with 8GB VRAM
   - Recommended: 16GB+ VRAM for full batch size

---

## Quick Start

```bash
# 1. Set up environment
cd spread_model
pip install -r requirements.txt

# 2. Authenticate with GEE
earthengine authenticate

# 3. Run full pipeline
python 01_fetch_fire_perimeters.py
python 02_fetch_daily_progressions.py
python 03_fetch_environmental_data.py
python 04_build_dataset.py
python 05_train_model.py
python 06_evaluate_model.py
```

---

## Configuration

All hyperparameters are centralized in `config.py`:

```python
# Key settings
IMG_SIZE = 256              # Spatial resolution
PREDICTION_DAYS = 7         # Days to predict
INPUT_CHANNELS = 19         # Environmental features
BATCH_SIZE = 8              # Training batch size
MAX_EPOCHS = 100            # Training epochs

# Data collection
START_YEAR = 2017
END_YEAR = 2024
MIN_FIRE_SIZE_ACRES = 1000
```

---

## Expected Performance

Based on similar fire spread models in literature:

| Metric | Day 1 | Day 3 | Day 7 |
|--------|-------|-------|-------|
| IoU | 0.65-0.75 | 0.50-0.60 | 0.35-0.45 |
| Dice | 0.75-0.85 | 0.60-0.70 | 0.45-0.55 |

Note: Performance decreases with prediction horizon as uncertainty compounds.

---

## Project Structure

```
spread_model/
├── config.py                    # Central configuration
├── 01_fetch_fire_perimeters.py  # Step 1: Get fire boundaries
├── 02_fetch_daily_progressions.py # Step 2: Get daily spread
├── 03_fetch_environmental_data.py # Step 3: Get terrain/weather
├── 04_build_dataset.py          # Step 4: Create TFRecords
├── 05_train_model.py            # Step 5: Train model
├── 06_evaluate_model.py         # Step 6: Evaluate model
├── README.md                    # This file
├── data/
│   ├── raw/                     # Downloaded raw data
│   ├── processed/               # Processed data
│   └── tfrecords/               # Training data
├── models/                      # Saved models
│   └── checkpoints/             # Training checkpoints
├── logs/                        # Training logs
│   └── tensorboard/             # TensorBoard logs
└── evaluation/                  # Evaluation results
```

---

## License

Part of the PyroCast project. See main repository for license information.
