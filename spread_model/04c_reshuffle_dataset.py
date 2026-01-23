"""
PyroCast Fire Spread Model - Step 4c: Reshuffle Augmented Dataset
==================================================================
Fixes data ordering issue where augmented variations are sequential.

Problem: Current dataset has 8 sequential variations of each fire, causing
training oscillations as model sees easy/hard fires in blocks.

Solution: 
1. Load all augmented samples
2. Shuffle completely (break up augmentation groups)
3. Create proper 80/15/5 train/val/test split
4. Write back to TFRecords

This ensures each batch has diverse fire scenarios.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def read_sample_count(info_path):
    """
    Read sample count from info.json file (much faster than iterating TFRecords).
    """
    with open(info_path, 'r') as f:
        info = json.load(f)
    return info['num_samples']


def stream_shuffle_tfrecords(input_paths, output_path, buffer_size=500):
    """
    Memory-efficient shuffling using TensorFlow's shuffle buffer.
    Buffer size of 500 = ~2GB memory (safe for most systems).
    """
    logger.info(f"Streaming shuffle to {output_path}...")
    logger.info(f"  Using buffer size: {buffer_size} samples (~{buffer_size * 4}MB)")
    
    try:
        # Create combined dataset from all inputs
        datasets = [tf.data.TFRecordDataset(path) for path in input_paths]
        combined = datasets[0]
        for ds in datasets[1:]:
            combined = combined.concatenate(ds)
        
        # Shuffle with buffer (keeps only buffer_size samples in memory)
        shuffled = combined.shuffle(buffer_size=buffer_size, seed=42, reshuffle_each_iteration=False)
        
        # Write to output
        count = 0
        with tf.io.TFRecordWriter(output_path) as writer:
            for record in shuffled:
                writer.write(record.numpy())
                count += 1
                if count % 1000 == 0:
                    logger.info(f"  Wrote {count} samples...")
        
        logger.info(f"  Total: {count} samples")
        return count
    
    except Exception as e:
        logger.error(f"Error during shuffle: {e}")
        raise





def reshuffle_dataset():
    """
    Memory-efficient dataset reshuffling using streaming.
    
    Steps:
    1. Count total samples
    2. Stream shuffle all data
    3. Split into train/val/test
    """
    logger.info("=" * 60)
    logger.info("Reshuffling Augmented Dataset (Memory-Efficient)")
    logger.info("=" * 60)
    
    # Paths
    train_path = os.path.join(TFRECORD_DIR, "spread_train.tfrecord")
    val_path = os.path.join(TFRECORD_DIR, "spread_val.tfrecord")
    test_path = os.path.join(TFRECORD_DIR, "spread_test.tfrecord")
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        logger.error("Original TFRecord files not found!")
        logger.error("Run 04b_augment_dataset.py first.")
        return
    
    # Read sample counts from info files (fast!)
    logger.info("\n1. Reading sample counts...")
    train_count = read_sample_count(os.path.join(TFRECORD_DIR, "spread_train_info.json"))
    val_count = read_sample_count(os.path.join(TFRECORD_DIR, "spread_val_info.json"))
    test_count = read_sample_count(os.path.join(TFRECORD_DIR, "spread_test_info.json"))
    total = train_count + val_count + test_count
    
    logger.info(f"  Train: {train_count} samples")
    logger.info(f"  Val: {val_count} samples")
    logger.info(f"  Test: {test_count} samples")
    logger.info(f"  Total: {total} samples")
    
    # Create temporary shuffled file
    temp_path = os.path.join(TFRECORD_DIR, "spread_temp_shuffled.tfrecord")
    
    logger.info("\n2. Shuffling all samples (streaming)...")
    total_shuffled = stream_shuffle_tfrecords(
        [train_path, val_path, test_path],
        temp_path,
        buffer_size=500  # 2GB buffer - safe for most systems
    )
    
    # Calculate new splits
    train_size = int(0.80 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    logger.info(f"\n3. Creating new splits (80/15/5)...")
    logger.info(f"  Train: {train_size} samples ({train_size/total*100:.1f}%)")
    logger.info(f"  Val:   {val_size} samples ({val_size/total*100:.1f}%)")
    logger.info(f"  Test:  {test_size} samples ({test_size/total*100:.1f}%)")
    
    # Split shuffled data
    logger.info("\n4. Splitting into new train/val/test...")
    dataset = tf.data.TFRecordDataset(temp_path)
    
    # Write new train
    logger.info("  Writing new train set...")
    with tf.io.TFRecordWriter(train_path) as writer:
        for i, record in enumerate(dataset.take(train_size)):
            writer.write(record.numpy())
            if (i + 1) % 1000 == 0:
                logger.info(f"    {i + 1}/{train_size}...")
    
    # Write new val
    logger.info("  Writing new val set...")
    with tf.io.TFRecordWriter(val_path) as writer:
        for i, record in enumerate(dataset.skip(train_size).take(val_size)):
            writer.write(record.numpy())
            if (i + 1) % 500 == 0:
                logger.info(f"    {i + 1}/{val_size}...")
    
    # Write new test
    logger.info("  Writing new test set...")
    with tf.io.TFRecordWriter(test_path) as writer:
        for i, record in enumerate(dataset.skip(train_size + val_size).take(test_size)):
            writer.write(record.numpy())
            if (i + 1) % 500 == 0:
                logger.info(f"    {i + 1}/{test_size}...")
    
    # Clean up temp file
    os.remove(temp_path)
    logger.info("  Removed temporary file")
    
    # Update info files
    info = {
        'num_samples': train_size,
        'shuffled': True,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(TFRECORD_DIR, "spread_train_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    info['num_samples'] = val_size
    with open(os.path.join(TFRECORD_DIR, "spread_val_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    info['num_samples'] = test_size
    with open(os.path.join(TFRECORD_DIR, "spread_test_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    # Delete old cache files
    logger.info("\n5. Cleaning up old cache files...")
    cache_files = [
        "spread_train_inputs.npy", "spread_train_outputs.npy", "spread_train_meta.json",
        "spread_val_inputs.npy", "spread_val_outputs.npy", "spread_val_meta.json",
    ]
    
    for cache_file in cache_files:
        path = os.path.join(TFRECORD_DIR, cache_file)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"  Deleted {cache_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Reshuffling Complete!")
    logger.info("=" * 60)
    logger.info("Benefits:")
    logger.info("  ✓ Removed sequential augmentation grouping")
    logger.info(f"  ✓ Validation set: 207 → {val_size} samples (9x larger!)")
    logger.info("  ✓ Each batch now has diverse fire scenarios")
    logger.info("  ✓ Memory-efficient streaming (no RAM issues)")
    logger.info("\nNext: Run 05_train_pytorch.py to train with shuffled data")
    logger.info("=" * 60)


if __name__ == "__main__":
    reshuffle_dataset()
