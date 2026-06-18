import os
# Force CPU (Safe Mode)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import glob
import random
import shutil
import math
# ================= CONFIGURATION =================
# Where the Earth Engine exports landed.
# Resolves relative to the repo root so this works on any machine (the original
# hard-coded //workspace//PyroCast// path was specific to one cloud VM).
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parent
RAW_DATA_DIR = str(_REPO_ROOT / "Training Data Florida")
# v3 dataset: matched-temporal-negatives schema. Drive filenames carry the v3 tag.
RAW_PATTERN  = "Export_Florida_Fire_Dataset_v3_Part_*.tfrecord*"

# Final destination for the merged & shuffled dataset.
OUTPUT_TRAIN = str(_REPO_ROOT / "Training Data Florida" / "Florida_Spatial_Train_v3.tfrecord")
OUTPUT_VAL   = str(_REPO_ROOT / "Training Data Florida" / "Florida_Spatial_Val_v3.tfrecord")

# Split Settings
VAL_SPLIT_PCT = 0.10  # 10% of LOCATIONS go to Validation
SEED = 42

# Features for spatial fingerprinting (must all be STATIC bands so the same location
# always produces the same fingerprint regardless of date).
# v2: dropped 'Slope' (no longer in feature set), kept Elevation + Pop_Density,
#     added LC_Forest as a third static signal so fingerprints stay unique.
BANDS_TO_CHECK = ['Elevation', 'Pop_Density', 'LC_Forest']

# ================= FUNCTIONS =================

def get_fingerprint(record_bytes):
    """Parses raw bytes to get location fingerprint (unique spatial ID)"""
    feature_desc = {
        k: tf.io.VarLenFeature(tf.float32) for k in BANDS_TO_CHECK
    }
    parsed = tf.io.parse_single_example(record_bytes, feature_desc)
    
    vals = []
    for k in BANDS_TO_CHECK:
        x = tf.sparse.to_dense(parsed[k], default_value=0.0)
        # Use mean of the image as unique location ID
        val = float(tf.reduce_mean(x).numpy())
        
        # CRITICAL FIX: Replace NaN with 0.0 to prevent KeyError
        if math.isnan(val):
            val = 0.0
            
        vals.append(round(val, 3))
    return tuple(vals)

def process_dataset():
    print("="*60)
    print("DATASET PROCESSOR: MERGE + SPATIAL SPLIT + SHUFFLE")
    print("="*60)
    
    # 1. Gather Files (recursive: Drive dumps each shard in its own
    #    drive-download-*/ subfolder, e.g. Training Data Florida/v3/drive-download-*/)
    files = glob.glob(os.path.join(RAW_DATA_DIR, "**", RAW_PATTERN), recursive=True)
    if not files:
        print(f"CRITICAL ERROR: No files found matching {RAW_PATTERN}")
        return
    print(f"Found {len(files)} raw parts.")
    
    # 2. Scan & Assign Locations (Pass 1)
    print("\n[Phase 1] Scanning for Unique Locations...")
    
    location_map = {} # Fingerprint -> 'train' or 'val'
    unique_locs = set()
    total_records = 0
    
    # We scan all files to build the map
    random.seed(SEED)
    
    for f in files:
        ds = tf.data.TFRecordDataset(f, compression_type=('GZIP' if f.endswith('.gz') else None))
        for record in ds:
            fp = get_fingerprint(record.numpy())
            
            if fp not in unique_locs:
                unique_locs.add(fp)
                # Assign this new location immediately
                if random.random() < VAL_SPLIT_PCT:
                    location_map[fp] = 'val'
                else:
                    location_map[fp] = 'train'
            
            total_records += 1
            if total_records % 2000 == 0:
                print(f"  Scanned {total_records} samples...", end='\r')
                
    print(f"\n  Scan Complete.")
    print(f"  Total Samples:   {total_records}")
    print(f"  Unique Locations: {len(unique_locs)}")
    
    # 3. Write & Shuffle (Pass 2)
    print("\n[Phase 2] Writing & Shuffling...")
    
    # In-memory shuffle buffer. v2 records are ~4 MB each (19 channels at 257x257
    # float32), so 5000 records would be ~20 GB and OOM on a 32 GB machine.
    # 1000 records (~4 GB peak) is still ample for a 12k-sample dataset.
    SHUFFLE_BUFFER_SIZE = 1000
    
    train_buffer = []
    val_buffer = []
    
    train_writer = tf.io.TFRecordWriter(OUTPUT_TRAIN)
    val_writer = tf.io.TFRecordWriter(OUTPUT_VAL)
    
    train_count = 0
    val_count = 0
    processed = 0

    def flush_buffer(buffer, writer):
        random.shuffle(buffer)
        for rec in buffer:
            writer.write(rec)
        return []

    # Read all files again
    for f in files:
        ds = tf.data.TFRecordDataset(f, compression_type=('GZIP' if f.endswith('.gz') else None))
        for record in ds:
            rec_bytes = record.numpy()
            fp = get_fingerprint(rec_bytes)
            assignment = location_map[fp]
            
            if assignment == 'train':
                train_buffer.append(rec_bytes)
                train_count += 1
                if len(train_buffer) >= SHUFFLE_BUFFER_SIZE:
                    train_buffer = flush_buffer(train_buffer, train_writer)
            else:
                val_buffer.append(rec_bytes)
                val_count += 1
                if len(val_buffer) >= SHUFFLE_BUFFER_SIZE:
                    val_buffer = flush_buffer(val_buffer, val_writer)
            
            processed += 1
            if processed % 2000 == 0:
                print(f"  Processed {processed}/{total_records}...", end='\r')

    # Flush remaining buffers
    if train_buffer: flush_buffer(train_buffer, train_writer)
    if val_buffer: flush_buffer(val_buffer, val_writer)
    
    train_writer.close()
    val_writer.close()
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Final Training Samples:   {train_count}")
    print(f"Final Validation Samples: {val_count}")
    print(f"Outputs:\n  {OUTPUT_TRAIN}\n  {OUTPUT_VAL}")
    print("="*60)
    print("Recommendation: Delete the 'Part_*.tfrecord' files now to save space.")

if __name__ == "__main__":
    process_dataset()