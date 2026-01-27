import tensorflow as tf
import glob
import os

# ================= CONFIGURATION =================
SOURCE_DIR = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data Florida\\"
INPUT_PATTERN = "Export_Florida_Fire_Dataset_Part_*.tfrecord"

# Output file (training only)
TRAIN_OUTPUT = "Florida_Train.tfrecord"

# Assume 15% was used for validation
VAL_SPLIT = 0.15

# Data dimensions
RAW_IMG_SIZE = 257
TARGET_IMG_SIZE = 256
CHANNELS = 15

# Band names
BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
         'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
         'Elevation', 'Slope', 'Pop_Density']

# ================= FUNCTIONS =================

def parse_example(serialized):
    """Parse TFRecord example"""
    flat_shape = [RAW_IMG_SIZE * RAW_IMG_SIZE]
    zero_default = [0.0] * (RAW_IMG_SIZE * RAW_IMG_SIZE)
    
    feature_desc = {k: tf.io.FixedLenFeature(flat_shape, tf.float32, default_value=zero_default) 
                    for k in BANDS}
    feature_desc['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    
    parsed = tf.io.parse_single_example(serialized, feature_desc)
    
    # Reshape and crop
    band_tensors = [tf.reshape(parsed[k], [RAW_IMG_SIZE, RAW_IMG_SIZE]) for k in BANDS]
    image = tf.stack(band_tensors, axis=-1)
    image = tf.image.resize_with_crop_or_pad(image, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    
    label = parsed['label']
    return image, label

def serialize_example(image, label):
    """Convert back to TFRecord"""
    feature = {}
    for i, band_name in enumerate(BANDS):
        band_data = image[:, :, i].numpy().flatten().tolist()
        feature[band_name] = tf.train.Feature(float_list=tf.train.FloatList(value=band_data))
    
    feature['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[label.numpy()]))
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# ================= MAIN =================

def quick_prepare_train():
    print("="*70)
    print("QUICK FLORIDA TRAINING SET PREPARATION")
    print("="*70)
    print("Creating training set only (assumes validation already exists)")
    print("="*70)
    
    # Find files
    search_path = os.path.join(SOURCE_DIR, INPUT_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f"\n❌ ERROR: No files found matching {search_path}")
        return
    
    print(f"\n[1/3] Found {len(files)} input file(s)")
    
    # Count total
    print(f"\n[2/3] Counting samples...")
    dataset = tf.data.TFRecordDataset(files, compression_type=None)
    total_count = sum(1 for _ in dataset)
    
    val_count = int(total_count * VAL_SPLIT)
    train_count = total_count - val_count
    
    print(f"  Total samples: {total_count}")
    print(f"  Skipping first {val_count} for validation")
    print(f"  Processing {train_count} for training")
    print(f"  Will write {train_count * 4} augmented samples")
    
    # Process training set only
    print(f"\n[3/3] Streaming, augmenting, and writing TRAINING set...")
    
    dataset = tf.data.TFRecordDataset(files, compression_type=None)
    dataset = dataset.shuffle(buffer_size=total_count, seed=42, reshuffle_each_iteration=False)
    dataset = dataset.skip(val_count)  # Skip validation portion
    dataset_parsed = dataset.map(parse_example)
    
    train_path = os.path.join(SOURCE_DIR, TRAIN_OUTPUT)
    
    with tf.io.TFRecordWriter(train_path) as writer:
        total_written = 0
        processed = 0
        
        for image, label in dataset_parsed:
            # Write 4 rotated versions
            for k in range(4):
                rotated = tf.image.rot90(image, k=k)
                serialized = serialize_example(rotated, label)
                writer.write(serialized)
                total_written += 1
                
                if total_written % 2000 == 0:
                    print(f"  Written: {total_written}/{train_count * 4} samples ({100*total_written/(train_count*4):.1f}%)...", end='\r')
            
            processed += 1
    
    print(f"  Written: {total_written}/{train_count * 4} samples (100.0%)... Done!   ")
    
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nOutput: {train_path}")
    print(f"Training samples: {total_written}")
    print("\nYou can now run Training_Florida.py")
    print("="*70)

if __name__ == "__main__":
    quick_prepare_train()
