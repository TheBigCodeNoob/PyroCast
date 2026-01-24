import tensorflow as tf
import glob
import os
import random

# ================= CONFIGURATION =================
# FLORIDA-SPECIFIC: Folder containing your downloaded Florida Part_*.tfrecord files
SOURCE_DIR = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data Florida\\"
# The name of the file pattern to look for
FILE_PATTERN = "Export_Florida_Fire_Dataset_Part_*.tfrecord"

# Output filename
OUTPUT_FILE = "Master_Florida_Fire_Dataset_Shuffled.tfrecord"

# Buffer size for shuffling (24000 fits easily in RAM)
# If you have < 50k samples, 50000 is safe.
SHUFFLE_BUFFER = 50000 

# ================= EXECUTION =================
def combine_and_shuffle():
    print("="*60)
    print("FLORIDA FIRE DATASET SHUFFLER")
    print("="*60)
    
    # 1. Find Files
    search_path = os.path.join(SOURCE_DIR, FILE_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f"CRITICAL ERROR: No files found matching {search_path}")
        print(f"\nMake sure you have downloaded Florida TFRecord files to:")
        print(f"  {SOURCE_DIR}")
        print(f"\nExpected file pattern: {FILE_PATTERN}")
        return

    print(f"Found {len(files)} TFRecord files. Combining...")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # 2. Read RAW Bytes (Much faster than parsing features)
    # We read the raw serialized strings so we don't care about columns/types here.
    dataset = tf.data.TFRecordDataset(files, compression_type=None)
    
    # 3. Shuffle in Memory
    # This ensures the "Hard Negatives" (Part X) are mixed with "Positives" (Part Y)
    print(f"\nShuffling with buffer size: {SHUFFLE_BUFFER}")
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER)

    # 4. Write to New File
    output_path = os.path.join(SOURCE_DIR, OUTPUT_FILE)
    print(f"Writing to: {output_path}...")
    
    writer = tf.io.TFRecordWriter(output_path)
    
    count = 0
    # Iterate through the shuffled raw records and write them
    for raw_record in dataset:
        writer.write(raw_record.numpy())
        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count} records...", end='\r')
            
    writer.close()
    
    print(f"\n\n" + "="*60)
    print(f"SUCCESS! Combined {count} Florida samples into:")
    print(f"  '{OUTPUT_FILE}'")
    print("="*60)
    print("\nYou can now run Training_Florida.py to train the model.")

if __name__ == "__main__":
    combine_and_shuffle()
