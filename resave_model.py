"""
Re-save model to fix Keras version compatibility issues.
Loads the model and re-saves it to strip problematic config like quantization_config.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only

import tensorflow as tf
import h5py
import json

MODEL_PATH = "A-lot-better-post-data-fix.keras"
OUTPUT_PATH = "fire_model_compatible.keras"

def fix_keras_config(filepath):
    """Remove quantization_config from saved .keras file"""
    import zipfile
    import tempfile
    import shutil
    
    print(f"Fixing Keras config in {filepath}...")
    
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract .keras file (it's a zip)
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(temp_dir)
        
        # Load and fix config.json
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Recursively remove quantization_config
        def remove_quant_config(obj):
            if isinstance(obj, dict):
                if 'quantization_config' in obj:
                    del obj['quantization_config']
                for key, value in obj.items():
                    remove_quant_config(value)
            elif isinstance(obj, list):
                for item in obj:
                    remove_quant_config(item)
        
        remove_quant_config(config)
        
        # Save fixed config
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Re-create .keras file
        output_path = filepath.replace('.keras', '_fixed.keras')
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zf.write(file_path, arcname)
        
        print(f"Fixed model saved to: {output_path}")
        return output_path
        
    finally:
        shutil.rmtree(temp_dir)

def main():
    print("="*60)
    print("MODEL COMPATIBILITY FIXER")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    # Method 1: Fix the config directly
    try:
        fixed_path = fix_keras_config(MODEL_PATH)
        
        # Test loading the fixed model
        print(f"\nTesting fixed model...")
        model = tf.keras.models.load_model(fixed_path, compile=False)
        print(f"SUCCESS! Model loaded with {len(model.layers)} layers")
        print(f"\nUse this model path in app.py: {fixed_path}")
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
        print("\nTrying Method 2: Save weights only...")
        
        # Method 2: Load with safe_mode=False and re-save
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
            model.save(OUTPUT_PATH)
            print(f"Model re-saved to: {OUTPUT_PATH}")
        except Exception as e2:
            print(f"Method 2 also failed: {e2}")
            print("\nYou may need to re-train the model or update Keras.")

if __name__ == "__main__":
    main()
