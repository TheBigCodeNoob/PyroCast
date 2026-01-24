import tensorflow as tf
import numpy as np
import glob
import os

# ================= CONFIGURATION =================
SOURCE_DIR = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data Florida\\"
INPUT_PATTERN = "Export_Florida_Fire_Dataset_Part_*.tfrecord"

# Output files
TRAIN_OUTPUT = "Florida_Train.tfrecord"
VAL_OUTPUT = "Florida_Val.tfrecord"

# Split ratio
VAL_SPLIT = 0.15  # 15% validation, 85% training

# Data dimensions
RAW_IMG_SIZE = 257
TARGET_IMG_SIZE = 256
CHANNELS = 15

# Band names
BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
         'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
         'Elevation', 'Slope', 'Pop_Density']

# ================= PARSING & AUGMENTATION =================

def parse_example(serialized):
    """Parse a single TFRecord example"""
    flat_shape = [RAW_IMG_SIZE * RAW_IMG_SIZE]
    zero_default = [0.0] * (RAW_IMG_SIZE * RAW_IMG_SIZE)
    
    feature_desc = {k: tf.io.FixedLenFeature(flat_shape, tf.float32, default_value=zero_default) 
                    for k in BANDS}
    feature_desc['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    
    parsed = tf.io.parse_single_example(serialized, feature_desc)
    
    # Reshape bands to images
    band_tensors = [tf.reshape(parsed[k], [RAW_IMG_SIZE, RAW_IMG_SIZE]) for k in BANDS]
    image = tf.stack(band_tensors, axis=-1)
    
    # Crop to target size
    image = tf.image.resize_with_crop_or_pad(image, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    
    label = parsed['label']
    return image, label

def augment_4x(image, label):
    """
    4x Augmentation: Generate 4 versions with 0°, 90°, 180°, 270° rotations
    Returns a list of 4 (image, label) tuples
    """
    augmented = []
    for k in range(4):
        rotated = tf.image.rot90(image, k=k)
        augmented.append((rotated, label))
    return augmented

def serialize_example(image, label):
    """Convert image and label back to TFRecord format"""
    # Flatten each band
    feature = {}
    for i, band_name in enumerate(BANDS):
        band_data = image[:, :, i].numpy().flatten().tolist()
        feature[band_name] = tf.train.Feature(float_list=tf.train.FloatList(value=band_data))
    
    feature['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[label.numpy()]))
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# ================= MAIN PROCESSING =================

def prepare_florida_data():
    print("="*70)
    print("FLORIDA FIRE DATASET PREPARATION")
    print("="*70)
    print("Operations: Combine → Shuffle → Augment (4x) → Split (Train/Val)")
    print("="*70)
    
    # 1. Find input files
    search_path = os.path.join(SOURCE_DIR, INPUT_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f"\n❌ ERROR: No files found matching {search_path}")
        print("\nMake sure you have downloaded Florida TFRecord files to:")
        print(f"  {SOURCE_DIR}")
        return
    
    print(f"\n[1/5] Found {len(files)} input file(s):")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # 2. Load and count
    print(f"\n[2/5] Loading and counting samples...")
    dataset = tf.data.TFRecordDataset(files, compression_type=None)
    total_count = sum(1 for _ in dataset)
    print(f"  Total original samples: {total_count}")
    
    # 3. Shuffle entire dataset
    print(f"\n[3/5] Shuffling all samples...")
    dataset = tf.data.TFRecordDataset(files, compression_type=None)
    dataset = dataset.shuffle(buffer_size=total_count, seed=42, reshuffle_each_iteration=False)
    
    # Parse into images
    dataset_parsed = dataset.map(parse_example)
    
    # 4. Apply 4x augmentation
    print(f"\n[4/5] Applying 4x rotation augmentation...")
    print(f"  Original samples: {total_count}")
    print(f"  After 4x augment: {total_count * 4}")
    
    augmented_samples = []
    count = 0
    for image, label in dataset_parsed:
        # Generate 4 rotated versions
        versions = augment_4x(image, label)
        augmented_samples.extend(versions)
        
        count += 1
        if count % 500 == 0:
            print(f"  Processed {count}/{total_count} samples...", end='\r')
    
    print(f"  Processed {total_count}/{total_count} samples... Done!")
    
    # Shuffle augmented data
    print(f"  Shuffling augmented samples...")
    np.random.seed(42)
    np.random.shuffle(augmented_samples)
    
    # 5. Split into train/val and write
    total_augmented = len(augmented_samples)
    val_size = int(total_augmented * VAL_SPLIT)
    train_size = total_augmented - val_size
    
    print(f"\n[5/5] Splitting and writing files...")
    print(f"  Training:   {train_size} samples ({100*(1-VAL_SPLIT):.0f}%)")
    print(f"  Validation: {val_size} samples ({100*VAL_SPLIT:.0f}%)")
    
    # Write validation set
    val_path = os.path.join(SOURCE_DIR, VAL_OUTPUT)
    print(f"\n  Writing validation set to: {VAL_OUTPUT}")
    with tf.io.TFRecordWriter(val_path) as writer:
        for i in range(val_size):
            image, label = augmented_samples[i]
            serialized = serialize_example(image, label)
            writer.write(serialized)
            if (i + 1) % 1000 == 0:
                print(f"    Validation: {i+1}/{val_size} samples...", end='\r')
    print(f"    Validation: {val_size}/{val_size} samples... Done!")
    
    # Write training set
    train_path = os.path.join(SOURCE_DIR, TRAIN_OUTPUT)
    print(f"\n  Writing training set to: {TRAIN_OUTPUT}")
    with tf.io.TFRecordWriter(train_path) as writer:
        for i in range(val_size, total_augmented):
            image, label = augmented_samples[i]
            serialized = serialize_example(image, label)
            writer.write(serialized)
            if (i - val_size + 1) % 1000 == 0:
                print(f"    Training: {i-val_size+1}/{train_size} samples...", end='\r')
    print(f"    Training: {train_size}/{train_size} samples... Done!")
    
    print("\n" + "="*70)
    print("✓ SUCCESS! Florida dataset prepared.")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  Training:   {train_path}")
    print(f"  Validation: {val_path}")
    print(f"\nDataset statistics:")
    print(f"  Original samples:     {total_count}")
    print(f"  After augmentation:   {total_augmented} (4x rotation)")
    print(f"  Training samples:     {train_size}")
    print(f"  Validation samples:   {val_size}")
    print(f"\nYou can now run Training_Florida.py")
    print("="*70)

if __name__ == "__main__":
    prepare_florida_data()
