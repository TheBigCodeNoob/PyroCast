"""
Test script to verify the model can be loaded with Keras 3.x
"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Path to the model
model_path = "web/A-lot-better-post-data-fix_fixed.keras"

if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
    print("Make sure the actual model file (not LFS pointer) is present.")
    exit(1)

file_size = os.path.getsize(model_path)
print(f"\nModel file size: {file_size / 1024 / 1024:.1f} MB")

if file_size < 10_000_000:  # Less than 10MB
    print("WARNING: File is too small - might be a Git LFS pointer!")
    print("Run 'git lfs pull' to download the actual model file.")
    exit(1)

print("\nAttempting to load model with Keras 3.x...")
try:
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
    print(f"\nModel summary:")
    print(f"  - Input shape: {model.input_shape}")
    print(f"  - Output shape: {model.output_shape}")
    print(f"  - Number of layers: {len(model.layers)}")
    print(f"  - Total parameters: {model.count_params():,}")
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    print("\nAttempting to load with compile=False...")
    try:
        model = keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully (without compilation)!")
        print(f"\nModel summary:")
        print(f"  - Input shape: {model.input_shape}")
        print(f"  - Output shape: {model.output_shape}")
        print(f"  - Number of layers: {len(model.layers)}")
        print(f"  - Total parameters: {model.count_params():,}")
    except Exception as e2:
        print(f"✗ ERROR: Could not load model even with compile=False: {e2}")
        exit(1)

print("\n✓ Model loading test PASSED!")
