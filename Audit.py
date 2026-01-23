import tensorflow as tf
import numpy as np
import os

FILE_PATH = "C:/Users/nonna/Downloads/Mixed_Data.tfrecord"
IMAGE_SIZE = 65 
BANDS = [
    'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
    'elevation', 'slope', 'aspect',
    'temperature_2m', 'total_precipitation',
    'u_component_of_wind_10m', 'v_component_of_wind_10m',
    'distance_to_road', 'population_density'
]

def parse_tfrecord(example_proto):
    feature_description = {band: tf.io.VarLenFeature(tf.float32) for band in BANDS}
    feature_description['label'] = tf.io.FixedLenFeature([], tf.float32)
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Check just the first band (B2 - Blue Light)
    # If B2 is all zeros, the whole image is likely a Dummy
    b2 = tf.sparse.to_dense(parsed['B2'], default_value=0.0)
    label = parsed['label']
    return b2, label

print(f"--- DATA VALUE INSPECTION ---")
print(f"Scanning: {FILE_PATH}")

dataset = tf.data.TFRecordDataset(FILE_PATH)
dataset = dataset.map(parse_tfrecord)

total_count = 0
zero_count = 0
fire_zeros = 0

print("Checking for 'Ghost' (All-Zero) Images...")

for b2_pixels, label in dataset:
    total_count += 1
    
    # Check if the sum of pixels is 0 (Image is completely empty)
    if tf.reduce_sum(b2_pixels) == 0:
        zero_count += 1
        if label.numpy() == 1:
            fire_zeros += 1
            
    if total_count % 5000 == 0:
        print(f"Scanned {total_count}...", end='\r')

print(f"\n\n--- DIAGNOSIS ---")
print(f"Total Images: {total_count}")
print(f"Empty 'Ghost' Images: {zero_count}")
print(f"Ghosts labeled as Fire: {fire_zeros}")

percentage = (zero_count / total_count) * 100
print(f"Ghost Data: {percentage:.2f}%")

if percentage > 5:
    print("CRITICAL ISSUE: Your dataset is polluted with empty images.")
    print("The model is trying to learn on blank data.")
else:
    print("Data Integrity looks good. The issue is likely normalization.")