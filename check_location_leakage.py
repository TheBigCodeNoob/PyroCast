import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import math

# Same files as Prepare_Florida_Data.py
TRAIN_FILE = "Training Data Florida/Florida_Train.tfrecord"
VAL_FILE = "Training Data Florida/Florida_Val.tfrecord"

# Same fingerprinting logic
BANDS_TO_CHECK = ['Elevation', 'Slope', 'Pop_Density']

def get_fingerprint(record_bytes):
    """Exact same logic as Prepare_Florida_Data.py"""
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

def analyze_dataset(filepath, name):
    """Analyze a single dataset file"""
    print(f"\nAnalyzing {name}...")
    
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return None, 0
    
    location_counts = {}  # fingerprint -> count
    total_samples = 0
    
    ds = tf.data.TFRecordDataset(filepath, compression_type=None)
    for record in ds:
        fp = get_fingerprint(record.numpy())
        location_counts[fp] = location_counts.get(fp, 0) + 1
        total_samples += 1
        
        if total_samples % 1000 == 0:
            print(f"  Processed {total_samples} samples...", end='\r')
    
    print(f"  Complete: {total_samples} samples")
    
    unique_locs = len(location_counts)
    avg_samples_per_loc = total_samples / unique_locs if unique_locs > 0 else 0
    
    print(f"  Unique Locations: {unique_locs}")
    print(f"  Total Samples: {total_samples}")
    print(f"  Avg Samples/Location: {avg_samples_per_loc:.2f}")
    
    # Show distribution
    sample_counts = list(location_counts.values())
    print(f"  Min Samples/Location: {min(sample_counts)}")
    print(f"  Max Samples/Location: {max(sample_counts)}")
    
    return set(location_counts.keys()), total_samples

def main():
    print("="*60)
    print("LOCATION LEAKAGE CHECKER")
    print("="*60)
    print("Using same fingerprinting logic as Prepare_Florida_Data.py")
    print(f"Fingerprint bands: {BANDS_TO_CHECK}")
    
    # Analyze both datasets
    train_locs, train_total = analyze_dataset(TRAIN_FILE, "TRAINING SET")
    val_locs, val_total = analyze_dataset(VAL_FILE, "VALIDATION SET")
    
    if train_locs is None or val_locs is None:
        print("\nERROR: Could not analyze datasets")
        return
    
    # Check for leakage
    print("\n" + "="*60)
    print("LEAKAGE ANALYSIS")
    print("="*60)
    
    overlap = train_locs.intersection(val_locs)
    
    if overlap:
        print(f"🚨 LEAKAGE DETECTED! {len(overlap)} locations appear in BOTH train and val")
        print(f"   This represents {len(overlap)/len(train_locs)*100:.2f}% of train locations")
        print(f"   and {len(overlap)/len(val_locs)*100:.2f}% of val locations")
        print("\n   Example overlapping locations:")
        for i, loc in enumerate(list(overlap)[:5]):
            print(f"     {i+1}. {loc}")
    else:
        print("✓ NO LEAKAGE DETECTED")
        print("  Train and validation sets have completely separate locations")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training Locations:   {len(train_locs)}")
    print(f"Training Samples:     {train_total}")
    print(f"Validation Locations: {len(val_locs)}")
    print(f"Validation Samples:   {val_total}")
    print(f"Total Unique Locations: {len(train_locs) + len(val_locs)}")
    print(f"Overlapping Locations:  {len(overlap)}")
    print(f"Location Isolation:     {(1 - len(overlap)/(len(train_locs) + len(val_locs)))*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()
