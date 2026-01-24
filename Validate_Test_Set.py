"""
Validate Best Model on Stress Test Dataset
Loads florida_model_best.keras and evaluates on Top 50 US Fires test set
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras

print("="*70)
print("STRESS TEST VALIDATION")
print("="*70)

# Test data path
TEST_FILE = r"C:\Users\nonna\Downloads\PyroCast\Testing\Stress_Test_Top50_US_Fires.tfrecord"
MODEL_PATH = r"C:\Users\nonna\Downloads\PyroCast\florida_model_best.keras"
BATCH_SIZE = 32

# Florida data format - use defaults for missing features
BANDS = ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
         'Elevation', 'Slope', 'Temp_Max', 'Humidity_Min', 'Precip', 'Wind_Speed']

# Use default values for any missing bands
default_val = [0.0] * (257*257)
feature_desc = {k: tf.io.FixedLenFeature([257*257], tf.float32, default_value=default_val) for k in BANDS}
feature_desc['label'] = tf.io.FixedLenFeature([1], tf.float32, default_value=[0.0])

def parse_fn(serialized):
    """Parse TFRecord - NO augmentation for test set"""
    parsed = tf.io.parse_single_example(serialized, feature_desc)
    bands = tf.stack([parsed[k] for k in BANDS], axis=-1)
    image = tf.reshape(bands, [257, 257, 14])[:256, :256, :]
    # Add 15th channel (zeros) to match model architecture
    image = tf.concat([image, tf.zeros([256, 256, 1])], axis=-1)
    label = parsed['label']
    return image, label

# Load test dataset
print("\n[1/4] Loading test dataset...")
print(f"  File: {TEST_FILE}")

test_ds = tf.data.TFRecordDataset(TEST_FILE)
test_ds = test_ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
test_ds_batched = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Count samples
num_samples = sum(1 for _ in tf.data.TFRecordDataset(TEST_FILE))
print(f"  Total test samples: {num_samples}")

# Load model
print(f"\n[2/4] Loading best model...")
print(f"  Model: {MODEL_PATH}")

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    exit(1)

# Evaluate on full test set
print(f"\n[3/4] Evaluating model on test set...")
results = model.evaluate(test_ds_batched, verbose=1)
print(f"\n  Test Loss:     {results[0]:.4f}")
print(f"  Test Accuracy: {results[1]:.4f}")
print(f"  Test AUC:      {results[2]:.4f}")

# Get individual predictions
print(f"\n[4/4] Getting individual predictions...")
print(f"  Running inference on {num_samples} samples...")

predictions = []
labels_true = []
sample_idx = 0

for images, labels in test_ds_batched:
    preds = model.predict(images, verbose=0)
    predictions.extend(preds.flatten())
    labels_true.extend(labels.numpy().flatten())
    
print(f"  ✓ Predictions complete")

# Convert to numpy arrays
predictions = np.array(predictions)
labels_true = np.array(labels_true)

# Calculate per-sample accuracy (using 0.5 threshold)
pred_binary = (predictions >= 0.5).astype(int)
correct = (pred_binary == labels_true).astype(int)
accuracy_per_sample = correct.mean()

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"Total samples:        {len(predictions)}")
print(f"Overall accuracy:     {accuracy_per_sample:.4f} ({correct.sum()}/{len(predictions)})")
print(f"Mean prediction:      {predictions.mean():.4f}")
print(f"Prediction std dev:   {predictions.std():.4f}")
print(f"Min prediction:       {predictions.min():.4f}")
print(f"Max prediction:       {predictions.max():.4f}")

# Show distribution of predictions
print(f"\nPrediction distribution:")
print(f"  < 0.1:  {(predictions < 0.1).sum():5d} samples ({100*(predictions < 0.1).sum()/len(predictions):.1f}%)")
print(f"  0.1-0.3: {((predictions >= 0.1) & (predictions < 0.3)).sum():5d} samples ({100*((predictions >= 0.1) & (predictions < 0.3)).sum()/len(predictions):.1f}%)")
print(f"  0.3-0.5: {((predictions >= 0.3) & (predictions < 0.5)).sum():5d} samples ({100*((predictions >= 0.3) & (predictions < 0.5)).sum()/len(predictions):.1f}%)")
print(f"  0.5-0.7: {((predictions >= 0.5) & (predictions < 0.7)).sum():5d} samples ({100*((predictions >= 0.5) & (predictions < 0.7)).sum()/len(predictions):.1f}%)")
print(f"  0.7-0.9: {((predictions >= 0.7) & (predictions < 0.9)).sum():5d} samples ({100*((predictions >= 0.7) & (predictions < 0.9)).sum()/len(predictions):.1f}%)")
print(f"  >= 0.9: {(predictions >= 0.9).sum():5d} samples ({100*(predictions >= 0.9).sum()/len(predictions):.1f}%)")

# Show sample predictions (first 20 and last 20)
print("\n" + "="*70)
print("SAMPLE PREDICTIONS (First 20)")
print("="*70)
print(f"{'Index':<8} {'True Label':<12} {'Prediction':<12} {'Correct':<8}")
print("-"*70)
for i in range(min(20, len(predictions))):
    pred_val = predictions[i]
    true_val = int(labels_true[i])
    pred_class = int(pred_val >= 0.5)
    is_correct = "✓" if pred_class == true_val else "✗"
    print(f"{i:<8} {true_val:<12} {pred_val:<12.4f} {is_correct:<8}")

if len(predictions) > 20:
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (Last 20)")
    print("="*70)
    print(f"{'Index':<8} {'True Label':<12} {'Prediction':<12} {'Correct':<8}")
    print("-"*70)
    for i in range(max(0, len(predictions)-20), len(predictions)):
        pred_val = predictions[i]
        true_val = int(labels_true[i])
        pred_class = int(pred_val >= 0.5)
        is_correct = "✓" if pred_class == true_val else "✗"
        print(f"{i:<8} {true_val:<12} {pred_val:<12.4f} {is_correct:<8}")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
