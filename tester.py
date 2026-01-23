import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIGURATION =================
DATA_PATH = "C:\\Users\\nonna\\Downloads\\PyroCast\\Testing\\Stress_Test_Top50_US_Fires.tfrecord"
# The Index of a failed fire (0-based). 
# Fire #43 in your list corresponds to index 126, 127, 128 (since 42 * 3 = 126)
TARGET_INDEX = 126 # Fire #43 (30 Days out)

RAW_IMG_SIZE = 257    
TARGET_IMG_SIZE = 256 
CHANNELS = 15         

def parse(proto):
    # (Same parsing logic as your tester - abbreviated for brevity)
    keys = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
            'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
            'Elevation', 'Slope', 'Pop_Density']
    
    flat = [RAW_IMG_SIZE*RAW_IMG_SIZE]
    zero = [0.0]* (RAW_IMG_SIZE*RAW_IMG_SIZE)
    feats = {k: tf.io.FixedLenFeature(flat, tf.float32, default_value=zero) for k in keys}
    feats['label'] = tf.io.FixedLenFeature([], tf.float32)
    
    parsed = tf.io.parse_single_example(proto, feats)
    img = tf.stack([tf.reshape(parsed[k], [RAW_IMG_SIZE, RAW_IMG_SIZE]) for k in keys], axis=-1)
    img = tf.image.resize_with_crop_or_pad(img, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    return img

def show_fire():
    ds = tf.data.TFRecordDataset(DATA_PATH).map(parse).skip(TARGET_INDEX).take(1)
    
    for img in ds:
        # Create a False Color Composite (SWIR2, NIR, Red) - Great for seeing burns/vegetation
        # Normalize for display
        swir = img[:,:,5]  # SWIR2
        nir = img[:,:,3]   # NIR
        red = img[:,:,2]   # Red
        
        rgb = tf.stack([swir, nir, red], axis=-1).numpy()
        
        # Simple Min-Max scaling for display
        rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title(f"Visualizing Sample #{TARGET_INDEX+1} (False Color)")
        plt.axis('off')
        plt.show()
        
        # Check for Zeros (The "Data Hole" Check)
        if np.mean(img) == 0:
            print("⚠️ DIAGNOSIS: IMAGE IS COMPLETELY EMPTY (ZEROS).")
            print("Reason: Cloud masking or Missing Data in GEE.")
        else:
            print(f"Image Mean Value: {np.mean(img):.4f} (Normal is ~0.1 - 0.5)")

if __name__ == "__main__":
    show_fire()