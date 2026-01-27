# Model Loading Error Fix

## Problem
The application was failing to load the model with the error:
```
TypeError: Could not deserialize class 'Functional' because its parent module 
tf_keras.src.models.functional cannot be imported.
```

## Root Cause
**Keras Version Mismatch**: The model was saved using **Keras 3.x** (standalone `keras` package) but the deployment environment was trying to load it with **tf_keras** (TensorFlow's bundled Keras 2.x).

The model configuration shows: `'module': 'keras.src.models.functional'` (Keras 3.x format)  
But the environment had: `tf_keras==2.15.0` (Keras 2.x)

## Solution
Updated the application to use Keras 3.x throughout:

### 1. Updated `requirements.txt`
**Before:**
```
tf_keras==2.15.0
```

**After:**
```
keras==3.0.5
```
(Removed `tf_keras`, added standalone `keras`)

### 2. Updated `web/app.py`
**Changes:**
- Set Keras backend environment variable before imports: `os.environ['KERAS_BACKEND'] = 'tensorflow'`
- Added import: `import keras`
- Changed model loading from `tf.keras.models.load_model()` to `keras.models.load_model()`
- Improved error handling with fallback to `compile=False` if needed

## Testing

### Step 1: Install Updated Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Model Loading
```bash
python test_model_loading.py
```

This will verify that:
- Keras 3.x is properly installed
- The model file is present (not a Git LFS pointer)
- The model can be loaded successfully

### Step 3: Run the Application
```bash
cd web
uvicorn app:app --reload
```

Or using the Procfile:
```bash
# From project root
cd web && uvicorn app:app --host 0.0.0.0 --port 8000
```

## Expected Output
When the app starts successfully, you should see:
```
Model file found (119.6 MB)
Loading model from /app/web/A-lot-better-post-data-fix_fixed.keras...
Model loaded successfully.
```

## Important Notes

1. **Git LFS**: Make sure the actual model file is downloaded, not just the LFS pointer:
   ```bash
   git lfs pull
   ```

2. **Model File Size**: The model should be ~120 MB. If it's less than 10 MB, it's likely a Git LFS pointer.

3. **Environment Variables**: If deploying to production (e.g., Heroku, Railway), ensure:
   - Python 3.10+ is used
   - If using a model URL, set `MODEL_URL` environment variable
   - The buildpack supports Git LFS if needed

4. **Keras Backend**: The application now uses TensorFlow as the Keras backend, which is set via `os.environ['KERAS_BACKEND'] = 'tensorflow'` at the top of `app.py`.

## Future Improvements

For consistency, consider updating the training script (`Training_Florida.py`) to also use Keras 3.x:

```python
# Instead of:
from tensorflow.keras import layers, models, Input, optimizers, callbacks

# Use:
import keras
from keras import layers, models, Input, optimizers, callbacks
```

This will ensure the entire pipeline uses the same Keras version.

## Rollback
If you need to rollback to the previous version:
```bash
git checkout HEAD~1 requirements.txt web/app.py
```

## Related Files Modified
- `requirements.txt` - Updated Keras dependency
- `web/app.py` - Updated imports and model loading logic
- `test_model_loading.py` - Created for testing (new file)
- `FIX_SUMMARY.md` - This documentation (new file)
