import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import keras
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Files
TRAIN_FILE = "Training Data Florida/Florida_Train.tfrecord"
VAL_FILE = "Training Data Florida/Florida_Val.tfrecord"
MODEL_FILE = "68-precision-91-auc-86-acc.keras"
# Same fingerprinting logic
BANDS_TO_CHECK = ['Elevation', 'Slope', 'Pop_Density']

IMG_SIZE = 256
CHANNELS = 15

ALL_BANDS = [
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
    'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
    'Elevation', 'Slope', 'Pop_Density'
]

def get_fingerprint(record_bytes):
    """Exact same logic as before"""
    feature_desc = {
        k: tf.io.VarLenFeature(tf.float32) for k in BANDS_TO_CHECK
    }
    parsed = tf.io.parse_single_example(record_bytes, feature_desc)
    
    vals = []
    for k in BANDS_TO_CHECK:
        x = tf.sparse.to_dense(parsed[k], default_value=0.0)
        val = float(tf.reduce_mean(x).numpy())
        
        if math.isnan(val):
            val = 0.0
            
        vals.append(round(val, 3))
    return tuple(vals)

def get_training_locations():
    """Get all unique locations from training set"""
    print("Scanning training set for locations...")
    train_locs = set()
    
    ds = tf.data.TFRecordDataset(TRAIN_FILE, compression_type=None)
    count = 0
    for record in ds:
        fp = get_fingerprint(record.numpy())
        train_locs.add(fp)
        count += 1
        if count % 1000 == 0:
            print(f"  Scanned {count} training samples...", end='\r')
    
    print(f"\n  Found {len(train_locs)} unique training locations")
    return train_locs

def parse_sample(record_bytes):
    """Parse a sample into image + label"""
    feature_desc = {
        'label': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    }
    for band in ALL_BANDS:
        feature_desc[band] = tf.io.VarLenFeature(tf.float32)
    
    parsed = tf.io.parse_single_example(record_bytes, feature_desc)
    label = parsed['label']
    
    # Stack bands into image with adaptive sizing (eager mode)
    band_arrays = []
    for band in ALL_BANDS:
        x = tf.sparse.to_dense(parsed[band], default_value=0.0).numpy()
        
        # Handle different sizes
        if len(x) == 257 * 257:
            arr = x.reshape(257, 257)[:IMG_SIZE, :IMG_SIZE]
        elif len(x) == 256 * 256:
            arr = x.reshape(256, 256)
        else:
            # Invalid data - use zeros
            arr = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        band_arrays.append(arr)
    
    img = np.stack(band_arrays, axis=-1)
    
    # Safety: replace NaNs
    img = np.nan_to_num(img, nan=0.0)
    
    return img, int(label.numpy())

def main():
    print("="*60)
    print("TRUE VALIDATION PERFORMANCE TEST")
    print("Testing on UNIQUE validation locations only")
    print("="*60)
    
    # Step 1: Get training locations
    train_locs = get_training_locations()
    
    # Step 2: Filter validation set
    print("\nFiltering validation set to unique locations...")
    
    val_ds = tf.data.TFRecordDataset(VAL_FILE, compression_type=None)
    
    unique_samples = []
    unique_count = 0
    leak_count = 0
    
    for record in val_ds:
        rec_bytes = record.numpy()
        fp = get_fingerprint(rec_bytes)
        
        if fp not in train_locs:
            # This is a truly unique location
            unique_samples.append(rec_bytes)
            unique_count += 1
        else:
            leak_count += 1
        
        if (unique_count + leak_count) % 500 == 0:
            print(f"  Processed {unique_count + leak_count} samples...", end='\r')
    
    print(f"\n  Unique validation samples: {unique_count}")
    print(f"  Leaked samples (excluded): {leak_count}")
    
    if unique_count == 0:
        print("\nERROR: No unique validation locations found!")
        print("All validation locations overlap with training.")
        return
    
    # Step 3: Parse samples
    print("\nParsing samples...")
    X_test = []
    y_test = []
    
    for i, rec_bytes in enumerate(unique_samples):
        img, label = parse_sample(rec_bytes)
        X_test.append(img)
        y_test.append(label)
        
        if (i + 1) % 100 == 0:
            print(f"  Parsed {i + 1}/{unique_count}...", end='\r')
    
    print(f"\n  Complete: {len(X_test)} samples ready")
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    
    print(f"\nDataset shape: {X_test.shape}")
    print(f"Label distribution: Fire={np.sum(y_test == 1)}, No-Fire={np.sum(y_test == 0)}")
    
    # Step 4: Load model
    print("\nLoading model...")
    
    # Patch the model file to remove quantization_config
    import zipfile
    import json
    import tempfile
    import shutil
    
    try:
        # Try normal loading first
        model = keras.models.load_model(MODEL_FILE, safe_mode=False)
        print("  Model loaded successfully")
    except (TypeError, ValueError) as e:
        if 'quantization_config' in str(e):
            print("  Patching model to remove incompatible quantization_config...")
            
            # Create temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract the keras file
                with zipfile.ZipFile(MODEL_FILE, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Load and modify config.json
                config_path = os.path.join(tmpdir, 'config.json')
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Recursively remove quantization_config from all layers
                def remove_quantization_config(obj):
                    if isinstance(obj, dict):
                        if 'quantization_config' in obj:
                            del obj['quantization_config']
                        for value in obj.values():
                            remove_quantization_config(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            remove_quantization_config(item)
                
                remove_quantization_config(config)
                
                # Save patched config
                with open(config_path, 'w') as f:
                    json.dump(config, f)
                
                # Create patched model file
                patched_model = MODEL_FILE.replace('.keras', '_patched.keras')
                with zipfile.ZipFile(patched_model, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                    for root, dirs, files in os.walk(tmpdir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, tmpdir)
                            zip_ref.write(file_path, arcname)
                
                print(f"  Created patched model: {patched_model}")
                print("  Loading patched model...")
                model = keras.models.load_model(patched_model, safe_mode=False)
                print("  Model loaded successfully")
        else:
            raise
    
    # Step 5: Run inference
    print("\nRunning inference on unique validation locations...")
    y_pred_probs = model.predict(X_test, verbose=1)
    
    # Get probabilities
    if y_pred_probs.shape[-1] == 1:
        y_pred_probs = y_pred_probs.flatten()
    else:
        y_pred_probs = y_pred_probs[:, 1]
    
    # Threshold at 0.5
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Step 6: Calculate metrics
    print("\n" + "="*60)
    print("TRUE VALIDATION METRICS (Unique Locations Only)")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_pred_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print(f"\nSamples Tested: {unique_count}")
    print(f"Accuracy:       {accuracy*100:.2f}%")
    print(f"Precision:      {precision*100:.2f}%")
    print(f"Recall:         {recall*100:.2f}%")
    print(f"F1 Score:       {f1:.4f}")
    print(f"AUC:            {auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print("Reported Metrics (With Leakage):")
    print("  Accuracy:  86%")
    print("  AUC:       91%")
    print("  Precision: 68%")
    
    print(f"\nTrue Metrics (Unique Locations):")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  AUC:       {auc*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    
    diff_acc = (accuracy - 0.86) * 100
    diff_auc = (auc - 0.91) * 100
    diff_prec = (precision - 0.68) * 100
    
    print(f"\nPerformance Gap:")
    print(f"  Accuracy:  {diff_acc:+.1f}%")
    print(f"  AUC:       {diff_auc:+.1f}%")
    print(f"  Precision: {diff_prec:+.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()
