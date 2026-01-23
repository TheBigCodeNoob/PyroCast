import tensorflow as tf
import os

# --- CONFIGURATION ---
INPUT_FILE = "C:/Users/nonna/Downloads/Mixed_Data.tfrecord"
TRAIN_FILE = "C:/Users/nonna/Downloads/Training_Data.tfrecord"
VAL_FILE = "C:/Users/nonna/Downloads/Validation_Data.tfrecord"

VAL_SPLIT = 0.05  # 5% as requested

print("--- DATA SPLITTER ---")
print(f"Reading: {INPUT_FILE}")

# 1. Count Total
raw_dataset = tf.data.TFRecordDataset(INPUT_FILE)
total_records = sum(1 for _ in raw_dataset)
print(f"Total Records: {total_records}")

val_size = int(total_records * VAL_SPLIT)
train_size = total_records - val_size

print(f"Splitting: {train_size} Training | {val_size} Validation")

# 2. Write Files
raw_dataset = tf.data.TFRecordDataset(INPUT_FILE)
# We take the first 5% for Validation (Since it's already mixed, this is random)
val_data = raw_dataset.take(val_size)
train_data = raw_dataset.skip(val_size)

print("Writing Validation File...")
with tf.io.TFRecordWriter(VAL_FILE) as writer:
    for record in val_data:
        writer.write(record.numpy())

print("Writing Training File...")
with tf.io.TFRecordWriter(TRAIN_FILE) as writer:
    for record in train_data:
        writer.write(record.numpy())

print("\nSUCCESS. You now have clean training and validation sets.")