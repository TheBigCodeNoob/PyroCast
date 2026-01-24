import tensorflow as tf
import os

# ================= CONFIGURATION =================
# FLORIDA-SPECIFIC: Paths for Florida fire data
INPUT_FILE = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data Florida\\Master_Florida_Fire_Dataset_Shuffled.tfrecord"
TRAIN_FILE = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data Florida\\Florida_Training_Data.tfrecord"
VAL_FILE = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data Florida\\Florida_Validation_Data.tfrecord"

VAL_SPLIT = 0.05  # 5% for validation

print("="*60)
print("FLORIDA FIRE DATA SPLITTER")
print("="*60)
print(f"Reading: {INPUT_FILE}")

# 1. Count Total
try:
    raw_dataset = tf.data.TFRecordDataset(INPUT_FILE)
    total_records = sum(1 for _ in raw_dataset)
    print(f"Total Records: {total_records}")
except Exception as e:
    print(f"\nERROR: Could not read input file")
    print(f"  {e}")
    print(f"\nMake sure you have run shuffler_Florida.py first to create:")
    print(f"  {INPUT_FILE}")
    exit(1)

val_size = int(total_records * VAL_SPLIT)
train_size = total_records - val_size

print(f"Splitting: {train_size} Training | {val_size} Validation")

# 2. Write Files
raw_dataset = tf.data.TFRecordDataset(INPUT_FILE)
# We take the first 5% for Validation (Since it's already mixed, this is random)
val_data = raw_dataset.take(val_size)
train_data = raw_dataset.skip(val_size)

print("\nWriting Validation File...")
with tf.io.TFRecordWriter(VAL_FILE) as writer:
    count = 0
    for record in val_data:
        writer.write(record.numpy())
        count += 1
        if count % 100 == 0:
            print(f"  Validation: {count}/{val_size} records...", end='\r')
print(f"  Validation: {count} records written to:")
print(f"    {VAL_FILE}")

print("\nWriting Training File...")
with tf.io.TFRecordWriter(TRAIN_FILE) as writer:
    count = 0
    for record in train_data:
        writer.write(record.numpy())
        count += 1
        if count % 500 == 0:
            print(f"  Training: {count}/{train_size} records...", end='\r')
print(f"  Training: {count} records written to:")
print(f"    {TRAIN_FILE}")

print("\n" + "="*60)
print("SUCCESS! You now have clean Florida training and validation sets.")
print("="*60)
print(f"\nTraining Set:   {TRAIN_FILE}")
print(f"Validation Set: {VAL_FILE}")
print("\nYou can now run Training_Florida.py to train the model.")
