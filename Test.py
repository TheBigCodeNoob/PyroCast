import os
# Force CPU execution to ensure stability
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report

# ================= CONFIGURATION =================
MODEL_PATH = 'Untested-post-data-fix.keras'
# Update this path if you want to test on Validation data instead
DATA_PATTERN = '/workspace/PyroCast/Training Data Florida/Florida_Spatial_Val.tfrecord'

BATCH_SIZE = 64
IMG_SIZE = 256

# Must match the bands used during training
BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
              'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
              'Elevation', 'Slope', 'Pop_Density']

# ================= DATA PARSER =================
def parse_record(example_proto):
    feature_desc = {name: tf.io.VarLenFeature(tf.float32) for name in BAND_NAMES}
    feature_desc['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    
    parsed = tf.io.parse_single_example(example_proto, feature_desc)
    
    bands = []
    for name in BAND_NAMES:
        x = tf.sparse.to_dense(parsed[name], default_value=0.0)
        num_elements = tf.shape(x)[0]
        
        # Robust Resize: Handles valid images and empty (0-size) errors safely
        x = tf.cond(
            tf.greater(num_elements, 0),
            true_fn=lambda: tf.image.resize(
                tf.reshape(x, [tf.cast(tf.sqrt(tf.cast(num_elements, tf.float32)), tf.int32), -1, 1]), 
                [IMG_SIZE, IMG_SIZE]
            )[..., 0],
            false_fn=lambda: tf.zeros([IMG_SIZE, IMG_SIZE], dtype=tf.float32)
        )
        bands.append(x)
        
    image = tf.stack(bands, axis=-1)
    # Ensure shape is explicit for the model
    image = tf.ensure_shape(image, [IMG_SIZE, IMG_SIZE, len(BAND_NAMES)])
    return image, parsed['label']

# ================= EXECUTION =================
def evaluate():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 2. Load Data
    files = glob.glob(DATA_PATTERN)
    if not files:
        print(f"Error: No data found matching {DATA_PATTERN}")
        return
    
    print(f"Found {len(files)} data files. Preparing dataset...")
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    
    # 3. Run Predictions
    print("Running forward pass (Inference)...")
    y_true = []
    y_pred_probs = []
    
    # Iterate through the dataset
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds.flatten())
        
        print(f"  Processed {len(y_true)} samples...", end='\r')
    
    print(f"\nFinished processing {len(y_true)} samples.")

    # 4. Calculate Metrics
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_class = (y_pred_probs > 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_pred_probs)
    acc = accuracy_score(y_true, y_pred_class)
    prec = precision_score(y_true, y_pred_class, zero_division=0)
    rec = recall_score(y_true, y_pred_class, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_class)

    # 5. Output Report
    print("\n" + "="*40)
    print("FINAL EVALUATION REPORT")
    print("="*40)
    print(f"Total Samples: {len(y_true)}")
    print("-" * 40)
    print(f"AUC:        {auc:.4f}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print("-" * 40)
    print("Confusion Matrix:")
    print(f" [ TN: {cm[0][0]:<5}  FP: {cm[0][1]:<5} ]")
    print(f" [ FN: {cm[1][0]:<5}  TP: {cm[1][1]:<5} ]")
    print("="*40)
    
    print("\nFull Classification Report:")
    print(classification_report(y_true, y_pred_class, target_names=['No Fire', 'Fire'], zero_division=0))

if __name__ == "__main__":
    evaluate()