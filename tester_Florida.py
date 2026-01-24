import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIGURATION =================
# FLORIDA-SPECIFIC: Path to your Florida TFRecord data
DATA_PATH = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data Florida\\Master_Florida_Fire_Dataset_Shuffled.tfrecord"

# The Index of a sample to visualize (0-based)
TARGET_INDEX = 0

RAW_IMG_SIZE = 257    
TARGET_IMG_SIZE = 256 
CHANNELS = 15         

def parse(proto):
    """Parse a single TFRecord example from Florida dataset"""
    keys = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
            'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
            'Elevation', 'Slope', 'Pop_Density']
    
    flat = [RAW_IMG_SIZE * RAW_IMG_SIZE]
    zero = [0.0] * (RAW_IMG_SIZE * RAW_IMG_SIZE)
    feats = {k: tf.io.FixedLenFeature(flat, tf.float32, default_value=zero) for k in keys}
    feats['label'] = tf.io.FixedLenFeature([], tf.float32)
    
    parsed = tf.io.parse_single_example(proto, feats)
    img = tf.stack([tf.reshape(parsed[k], [RAW_IMG_SIZE, RAW_IMG_SIZE]) for k in keys], axis=-1)
    img = tf.image.resize_with_crop_or_pad(img, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    label = parsed['label']
    return img, label

def show_sample(index=0):
    """Visualize a sample from the Florida fire dataset"""
    print("="*60)
    print("FLORIDA FIRE DATASET VISUALIZER")
    print("="*60)
    print(f"Data Path: {DATA_PATH}")
    print(f"Sample Index: {index}")
    print("="*60)
    
    try:
        ds = tf.data.TFRecordDataset(DATA_PATH).map(parse).skip(index).take(1)
    except Exception as e:
        print(f"\nERROR: Could not load dataset")
        print(f"  {e}")
        print(f"\nMake sure the file exists at:")
        print(f"  {DATA_PATH}")
        return
    
    for img, label in ds:
        label_val = label.numpy()
        label_str = "FIRE" if label_val == 1 else "NO FIRE"
        
        print(f"\nSample Label: {label_str} ({label_val})")
        
        # Create a False Color Composite (SWIR2, NIR, Red) - Great for seeing burns/vegetation
        # Normalize for display
        swir = img[:,:,5].numpy()  # SWIR2
        nir = img[:,:,3].numpy()   # NIR
        red = img[:,:,2].numpy()   # Red
        
        rgb = np.stack([swir, nir, red], axis=-1)
        
        # Simple Min-Max scaling for display
        if np.max(rgb) - np.min(rgb) > 0:
            rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
        
        # Also create True Color Composite (Red, Green, Blue)
        r = img[:,:,2].numpy()  # Red
        g = img[:,:,1].numpy()  # Green
        b = img[:,:,0].numpy()  # Blue
        
        true_color = np.stack([r, g, b], axis=-1)
        if np.max(true_color) - np.min(true_color) > 0:
            true_color = (true_color - np.min(true_color)) / (np.max(true_color) - np.min(true_color))
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        axes[0].imshow(true_color)
        axes[0].set_title(f"True Color (RGB) - {label_str}")
        axes[0].axis('off')
        
        axes[1].imshow(rgb)
        axes[1].set_title(f"False Color (SWIR2, NIR, Red) - {label_str}")
        axes[1].axis('off')
        
        plt.suptitle(f"Florida Sample #{index+1} - Label: {label_str}", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Check for Zeros (The "Data Hole" Check)
        mean_val = np.mean(img)
        if mean_val == 0:
            print("\n⚠️ DIAGNOSIS: IMAGE IS COMPLETELY EMPTY (ZEROS).")
            print("Reason: Cloud masking or Missing Data in GEE.")
        else:
            print(f"\nImage Statistics:")
            print(f"  Mean Value: {mean_val:.4f} (Normal is ~0.1 - 0.5)")
            print(f"  Min Value:  {np.min(img):.4f}")
            print(f"  Max Value:  {np.max(img):.4f}")

def count_samples():
    """Count total samples in the Florida dataset"""
    print("="*60)
    print("FLORIDA DATASET SAMPLE COUNTER")
    print("="*60)
    
    try:
        ds = tf.data.TFRecordDataset(DATA_PATH)
        total = sum(1 for _ in ds)
        
        # Count labels
        ds_parsed = ds.map(parse)
        fire_count = 0
        no_fire_count = 0
        
        for _, label in ds_parsed:
            if label.numpy() == 1:
                fire_count += 1
            else:
                no_fire_count += 1
        
        print(f"\nTotal Samples:    {total}")
        print(f"  Fire (label=1): {fire_count} ({100*fire_count/total:.1f}%)")
        print(f"  No Fire (0):    {no_fire_count} ({100*no_fire_count/total:.1f}%)")
        
    except Exception as e:
        print(f"\nERROR: Could not load dataset")
        print(f"  {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--count":
            count_samples()
        else:
            try:
                idx = int(sys.argv[1])
                show_sample(idx)
            except ValueError:
                print(f"Usage: python tester_Florida.py [index]")
                print(f"       python tester_Florida.py --count")
    else:
        show_sample(TARGET_INDEX)
