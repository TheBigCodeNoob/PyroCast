# Model Loading Fix - Keras 3.x Deserialization Issue

## Problem
The model failed to load with the error:
```
ERROR: <class 'keras.src.models.functional.Functional'> could not be deserialized properly
Exception: functional_from_config.<locals>.get_tensor() takes 3 positional arguments but 11 were given
```

This indicates a version incompatibility in Keras 3.x model deserialization. The model was saved with one version of Keras 3.x but couldn't be loaded with Keras 3.0.5.

## Root Cause
- Model saved with Keras 3.x format (`keras.src.models.functional.Functional`)
- Keras 3.0.5 has stricter deserialization that fails on certain model configs
- The `inbound_nodes` structure in the model config doesn't match what the loader expects

## Solution Applied
1. **Upgraded Keras**: Changed from `keras==3.0.5` to `keras==3.6.0`
   - Keras 3.6.0 has better backward compatibility and more robust deserialization
   
2. **Added `safe_mode=False`**: Modified model loading in `web/app.py`
   ```python
   model = keras.models.load_model(model_path, safe_mode=False)
   ```
   - `safe_mode=False` bypasses strict type/shape checking during deserialization
   - Allows loading models with minor version incompatibilities

## Files Changed
1. **requirements.txt**: Updated `keras==3.0.5` → `keras==3.6.0`
2. **web/app.py**: Added `safe_mode=False` parameter to both model loading attempts

## Deployment Steps
```bash
# Commit the changes
git add requirements.txt web/app.py MODEL_LOADING_FIX.md
git commit -m "Fix model loading: upgrade to Keras 3.6.0 and add safe_mode=False"
git push

# Railway will automatically redeploy with the new dependencies
```

## Testing Locally
```bash
cd web
pip install -r ../requirements.txt
python app.py
# Check logs for "Model loaded successfully"
```

## Alternative Solutions (if this doesn't work)
1. **Try Keras 3.7+**: Even newer versions may have better compatibility
2. **Re-save the model**: Load in training environment and save with explicit format:
   ```python
   model.save('model.keras', save_format='keras_v3')
   ```
3. **Use H5 format**: Save as `.h5` instead of `.keras` for broader compatibility

## Technical Details
- TensorFlow: 2.16.1 (supports Keras 3.x)
- Keras Backend: TensorFlow (set via `KERAS_BACKEND` env var)
- Model Format: Keras 3.x native format (`.keras` file)
- Python: 3.10.13
