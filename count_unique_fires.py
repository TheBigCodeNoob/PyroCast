import os
# ================= CRITICAL =================
# Forces CPU execution to avoid RTX 5080 crash.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# ============================================

import tensorflow as tf
import numpy as np
import glob
from collections import defaultdict

# ================= CONFIGURATION =================
DATA_DIR = "//workspace//PyroCast//Training Data Florida//"
FILE_PATTERN = "*.tfrecord"

# We standardize everything to 256 for analysis
TARGET_IMG_SIZE = 256

# Band indices
BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
         'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
         'Elevation', 'Slope', 'Pop_Density']

# ================= PARSING =================

def parse_tfrecord(proto):
    """Parse a single TFRecord example"""
    
    # Use VarLenFeature to safely read varying sizes
    feature_desc = {k: tf.io.VarLenFeature(tf.float32) for k in BANDS}
    feature_desc['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    
    parsed = tf.io.parse_single_example(proto, feature_desc)
    
    # Helper to decode, reshape, and standardize dynamic tensors
    def decode_band(name):
        x = tf.sparse.to_dense(parsed[name], default_value=0.0)
        
        # 1. Determine size dynamically
        num_elements = tf.shape(x)[0]
        side_len = tf.cast(tf.sqrt(tf.cast(num_elements, tf.float32)), tf.int32)
        
        # 2. Reshape to [H, W, 1] (3D is required for image ops)
        x = tf.reshape(x, [side_len, side_len, 1])
        
        # 3. Standardize to 256x256
        x = tf.image.resize_with_crop_or_pad(x, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
        
        # 4. Squeeze back to [H, W] for analysis
        x = tf.squeeze(x, axis=-1)
        return x

    # Extract static features (don't change over time)
    elevation = decode_band('Elevation')
    slope = decode_band('Slope')
    pop_density = decode_band('Pop_Density')
    
    # Get center pixel values (most representative of location)
    center = TARGET_IMG_SIZE // 2
    elev_center = elevation[center, center]
    slope_center = slope[center, center]
    pop_center = pop_density[center, center]
    
    # Also get mean values for the central 50x50 region
    half_size = 25
    elev_region = tf.reduce_mean(elevation[center-half_size:center+half_size, 
                                            center-half_size:center+half_size])
    slope_region = tf.reduce_mean(slope[center-half_size:center+half_size, 
                                         center-half_size:center+half_size])
    pop_region = tf.reduce_mean(pop_density[center-half_size:center+half_size, 
                                             center-half_size:center+half_size])
    
    label = parsed['label']
    
    return elev_center, slope_center, pop_center, elev_region, slope_region, pop_region, label

# ================= MAIN ANALYSIS =================

def count_unique_fires():
    print("="*70)
    print("FLORIDA FIRE DATASET - UNIQUE FIRE COUNTER")
    print("="*70)
    
    # Find files
    search_path = os.path.join(DATA_DIR, FILE_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f"\nERROR: No files found matching {search_path}")
        print("\nMake sure you have downloaded Florida TFRecord files to:")
        print(f"  {DATA_DIR}")
        return
    
    print(f"\nFound {len(files)} TFRecord file(s):")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    print(f"\nAnalyzing samples...")
    
    # Load all files
    dataset = tf.data.TFRecordDataset(files, compression_type=None)
    dataset = dataset.map(parse_tfrecord)
    
    # Collect fire locations
    fire_locations = []
    total_fires = 0
    total_non_fires = 0
    
    for elev_c, slope_c, pop_c, elev_r, slope_r, pop_r, label in dataset:
        if label.numpy() == 1:
            # This is a fire sample
            # Create a "fingerprint" of the location using static features
            fingerprint = (
                round(elev_r.numpy(), 3),   # Elevation (rounded to ~80m precision)
                round(slope_r.numpy(), 3),  # Slope
                round(pop_r.numpy(), 3)     # Population
            )
            fire_locations.append(fingerprint)
            total_fires += 1
        else:
            total_non_fires += 1
    
    print(f"\n" + "="*70)
    print("SAMPLE COUNTS")
    print("="*70)
    print(f"Total samples:        {total_fires + total_non_fires}")
    print(f"  Fire samples:       {total_fires}")
    print(f"  Non-fire samples:   {total_non_fires}")
    
    # Count unique locations
    unique_locations = set(fire_locations)
    
    print(f"\n" + "="*70)
    print("UNIQUE FIRE ANALYSIS")
    print("="*70)
    print(f"Unique fire locations: {len(unique_locations)}")
    print(f"Total fire samples:    {total_fires}")
    
    if len(unique_locations) > 0:
        repetition_factor = total_fires / len(unique_locations)
        print(f"Avg samples per fire:  {repetition_factor:.1f}x")
        
        # Count frequency distribution
        location_counts = defaultdict(int)
        for loc in fire_locations:
            location_counts[loc] += 1
        
        # Sort by frequency
        sorted_counts = sorted(location_counts.values(), reverse=True)
        
        print(f"\nRepetition Statistics:")
        print(f"  Max samples from one fire: {sorted_counts[0]}")
        print(f"  Min samples from one fire: {sorted_counts[-1]}")
        print(f"  Median:                    {sorted_counts[len(sorted_counts)//2]}")
        
        # Show distribution
        print(f"\nDistribution of samples per unique fire:")
        ranges = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 999)]
        for low, high in ranges:
            count = sum(1 for c in sorted_counts if low <= c <= high)
            if count > 0:
                pct = 100 * count / len(unique_locations)
                if low == high:
                    print(f"  {low:3d} sample:        {count:4d} fires ({pct:5.1f}%)")
                else:
                    print(f"  {low:3d}-{high:3d} samples: {count:4d} fires ({pct:5.1f}%)")
    
    print(f"\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if len(unique_locations) > 0:
        if repetition_factor > 50:
            print("⚠️  WARNING: Very high repetition factor!")
            print("   The model may severely overfit to these specific fire locations.")
            print("   Consider reducing total samples or using transfer learning.")
        elif repetition_factor > 20:
            print("⚠️  CAUTION: High repetition factor.")
            print("   Some overfitting is likely. Monitor validation performance carefully.")
        elif repetition_factor > 10:
            print("✓  Moderate repetition - acceptable with good augmentation.")
        else:
            print("✓  Low repetition - good dataset diversity.")
    
    print("\nNOTE: This analysis uses static geographic features (elevation, slope,")
    print("population) to estimate unique fire locations. The actual number may vary")
    print("slightly if fires occur in areas with very similar terrain characteristics.")
    print("="*70)

if __name__ == "__main__":
    count_unique_fires()