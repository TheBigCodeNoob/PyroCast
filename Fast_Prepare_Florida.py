import tensorflow as tf
import glob
import os

# ================= CONFIGURATION =================
SOURCE_DIR = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data Florida\\"
INPUT_PATTERN = "Export_Florida_Fire_Dataset_Part_*.tfrecord"

TRAIN_OUTPUT = "Florida_Train.tfrecord"
VAL_OUTPUT = "Florida_Val.tfrecord"
VAL_SPLIT = 0.15

# ================= MAIN =================

def fast_prepare():
    """
    NO PARSING, NO AUGMENTATION - Just shuffle and split raw bytes.
    Augmentation will happen during training (much faster).
    """
    print("="*70)
    print("FAST FLORIDA DATASET PREPARATION")
    print("="*70)
    print("Strategy: Shuffle → Split → Copy raw bytes (NO parsing!)")
    print("Augmentation will be done during training (4x faster)")
    print("="*70)
    
    # Find files
    search_path = os.path.join(SOURCE_DIR, INPUT_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f"\n❌ ERROR: No files found matching {search_path}")
        return
    
    print(f"\n[1/4] Found {len(files)} input file(s)")
    
    # Count
    print(f"\n[2/4] Counting samples...")
    dataset = tf.data.TFRecordDataset(files, compression_type=None)
    total_count = sum(1 for _ in dataset)
    
    val_count = int(total_count * VAL_SPLIT)
    train_count = total_count - val_count
    
    print(f"  Total: {total_count}")
    print(f"  Train: {train_count} (85%)")
    print(f"  Val:   {val_count} (15%)")
    
    # Shuffle
    print(f"\n[3/4] Shuffling all samples...")
    dataset = tf.data.TFRecordDataset(files, compression_type=None)
    dataset = dataset.shuffle(buffer_size=total_count, seed=42, reshuffle_each_iteration=False)
    
    # Split
    val_dataset = dataset.take(val_count)
    train_dataset = dataset.skip(val_count)
    
    # Write validation
    print(f"\n[4/4] Writing files (copying raw bytes - FAST)...")
    val_path = os.path.join(SOURCE_DIR, VAL_OUTPUT)
    print(f"  Writing {VAL_OUTPUT}...")
    
    with tf.io.TFRecordWriter(val_path) as writer:
        count = 0
        for raw_record in val_dataset:
            writer.write(raw_record.numpy())
            count += 1
            if count % 500 == 0:
                print(f"    Val: {count}/{val_count}...", end='\r')
    print(f"    Val: {val_count}/{val_count}... Done!   ")
    
    # Write training
    train_path = os.path.join(SOURCE_DIR, TRAIN_OUTPUT)
    print(f"  Writing {TRAIN_OUTPUT}...")
    
    with tf.io.TFRecordWriter(train_path) as writer:
        count = 0
        for raw_record in train_dataset:
            writer.write(raw_record.numpy())
            count += 1
            if count % 500 == 0:
                print(f"    Train: {count}/{train_count}...", end='\r')
    print(f"    Train: {train_count}/{train_count}... Done!   ")
    
    print("\n" + "="*70)
    print("✓ SUCCESS! (Completed in ~2-3 minutes)")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  Training:   {train_path} ({train_count} samples)")
    print(f"  Validation: {val_path} ({val_count} samples)")
    print(f"\nNOTE: No pre-augmentation applied.")
    print("Training_Florida.py will do 4x rotation augmentation on-the-fly.")
    print("="*70)

if __name__ == "__main__":
    fast_prepare()
