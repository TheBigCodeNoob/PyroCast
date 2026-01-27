# Florida Wildfire Risk Model

This folder contains scripts specifically designed to train a wildfire risk prediction model using **only** wildfire data from Florida.

## Overview

The Florida model uses the same SE-ResNet-18 architecture and data format as the national model, but is trained exclusively on fire/non-fire data from within Florida's boundaries.

### Key Differences from National Model
- **Geographic Filter**: All data (fires and random points) are constrained to Florida's bounding box: `[-87.6, 24.5, -80.0, 31.0]`
- **Data Source**: Uses MTBS (Monitoring Trends in Burn Severity) fires filtered to Florida
- **Output**: Model is saved as `best_fire_model_florida.keras`

## Workflow

### Step 1: Gather Data
```bash
python Dataget_Florida.py
```
This script:
- Connects to Google Earth Engine
- Finds historical fires in Florida from the MTBS dataset (2018+)
- Generates random non-fire points within Florida
- Exports 12,000 samples (6,000 fire, 6,000 non-fire) as TFRecord files
- Files are exported to your Google Drive folder: `Fire_Prediction_Dataset_Florida_v1`

**Note**: After running, monitor tasks at https://code.earthengine.google.com/tasks

### Step 2: Download Data
1. Go to Google Drive
2. Download all `Export_Florida_Fire_Dataset_Part_*.tfrecord` files
3. Place them in: `Training Data Florida/`

### Step 3: Shuffle Data
```bash
python shuffler_Florida.py
```
This combines all part files into a single shuffled file: `Master_Florida_Fire_Dataset_Shuffled.tfrecord`

### Step 4: Create Train/Validation Split (Optional)
```bash
python Mixer_Florida.py
```
Creates a 95/5 train/validation split:
- `Florida_Training_Data.tfrecord`
- `Florida_Validation_Data.tfrecord`

### Step 5: Train Model
```bash
python Training_Florida.py
```
Or use the GPU batch file:
```bash
train_florida_gpu.bat
```

Training parameters:
- **Architecture**: SE-ResNet-18 (same as national model)
- **Input**: 256x256x15 (15 spectral/environmental bands)
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)
- **Learning Rate**: 1e-4 with Cosine Decay

### Step 6: Validate Model
```bash
python Validate_Florida.py
```
Evaluates the trained model on the validation set and shows detailed metrics.

### Step 7: Visualize Data (Optional)
```bash
python tester_Florida.py        # Show first sample
python tester_Florida.py 10     # Show sample at index 10
python tester_Florida.py --count  # Count samples and label distribution
```

## Data Bands

The model uses 15 input bands:

| Band | Description | Source |
|------|-------------|--------|
| Blue | Blue Light (B2) | Sentinel-2 |
| Green | Green Light (B3) | Sentinel-2 |
| Red | Red Light (B4) | Sentinel-2 |
| NIR | Near Infrared (B8) | Sentinel-2 |
| SWIR1 | Short-Wave IR 1 (B11) | Sentinel-2 |
| SWIR2 | Short-Wave IR 2 (B12) | Sentinel-2 |
| NDVI | Vegetation Index | Derived |
| NDMI | Moisture Index | Derived |
| Temp_Max | Maximum Temperature | GRIDMET |
| Humidity_Min | Minimum Humidity | GRIDMET |
| Wind_Speed | Wind Speed | GRIDMET |
| Precip | Precipitation | GRIDMET |
| Elevation | Terrain Elevation | SRTM |
| Slope | Terrain Slope | Derived from SRTM |
| Pop_Density | Population Density | WorldPop |

## Files

| File | Description |
|------|-------------|
| `Dataget_Florida.py` | Data gathering script (GEE export) |
| `shuffler_Florida.py` | Combines and shuffles TFRecord files |
| `Mixer_Florida.py` | Creates train/validation split |
| `Training_Florida.py` | Model training script |
| `Validate_Florida.py` | Model validation script |
| `tester_Florida.py` | Data visualization tool |
| `train_florida_gpu.bat` | Batch file for GPU training |

## Output

After training, you will have:
- `best_fire_model_florida.keras` - The trained Florida fire model

## Troubleshooting

### "No files found" error
Make sure you've downloaded the TFRecord files from Google Drive to the `Training Data Florida/` folder.

### GEE Authentication Error
Run `earthengine authenticate` in terminal to set up credentials.

### Memory Issues
- Reduce `BATCH_SIZE` in Training_Florida.py
- Reduce `SHUFFLE_BUFFER` in shuffler_Florida.py

### Empty Images
Some samples may have missing data due to cloud coverage. The model handles this with default values.
