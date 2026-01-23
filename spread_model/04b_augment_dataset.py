"""
PyroCast Fire Spread Model - Augmented Dataset Builder
======================================================
Generates an augmented dataset with 8 variations per sample (D4 Symmetry Group).
Designed for speed using parallel processing and sharded writing to avoid MemoryError.
"""

import sys
import os
import json
import glob
import random
import time
import shutil
import numpy as np
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import config
try:
    from config import *
except ImportError:
    sys.path.append(os.path.join(current_dir, 'spread_model'))
    from config import *

# Import DatasetBuilder from 04_build_dataset.py
spec = importlib.util.spec_from_file_location("build_dataset", os.path.join(current_dir, "04_build_dataset.py"))
build_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset_module)
DatasetBuilder = build_dataset_module.DatasetBuilder

# Indices
ASPECT_IDX = 3
WIND_U_IDX = 7
WIND_V_IDX = 8

def get_d4_augmentations(input_tensor, output_tensor):
    """Generate all 8 symmetries of the square (D4 group)."""
    augmented = []
    
    def rotate_sample(img_in, img_out, k):
        in_rot = np.rot90(img_in, k=k, axes=(0, 1)).copy()
        out_rot = np.rot90(img_out, k=k, axes=(1, 2)).copy()
        if k == 0: return in_rot, out_rot
        
        u, v = in_rot[:, :, WIND_U_IDX], in_rot[:, :, WIND_V_IDX]
        if k == 1: new_u, new_v = -v, u
        elif k == 2: new_u, new_v = -u, -v
        elif k == 3: new_u, new_v = v, -u
        else: new_u, new_v = u, v
        in_rot[:, :, WIND_U_IDX] = new_u
        in_rot[:, :, WIND_V_IDX] = new_v
        
        aspect_shift = k * 0.25
        in_rot[:, :, ASPECT_IDX] = (in_rot[:, :, ASPECT_IDX] - aspect_shift) % 1.0
        return in_rot, out_rot

    def flip_h_sample(img_in, img_out):
        in_flip = np.flip(img_in, axis=1).copy()
        out_flip = np.flip(img_out, axis=2).copy()
        in_flip[:, :, WIND_U_IDX] = -in_flip[:, :, WIND_U_IDX]
        in_flip[:, :, ASPECT_IDX] = (1.0 - in_flip[:, :, ASPECT_IDX]) % 1.0
        return in_flip, out_flip

    augmented.append(rotate_sample(input_tensor, output_tensor, 0))
    augmented.append(rotate_sample(input_tensor, output_tensor, 1))
    augmented.append(rotate_sample(input_tensor, output_tensor, 2))
    augmented.append(rotate_sample(input_tensor, output_tensor, 3))
    
    flip_in, flip_out = flip_h_sample(input_tensor, output_tensor)
    augmented.append((flip_in, flip_out))
    
    augmented.append(rotate_sample(flip_in, flip_out, 1))
    augmented.append(rotate_sample(flip_in, flip_out, 2))
    augmented.append(rotate_sample(flip_in, flip_out, 3))
    
    return augmented

def process_chunk_task(fire_ids, output_path, augment=False):
    """
    Process a list of fires and write directly to a TFRecord file.
    Returns the number of samples written.
    """
    try:
        builder = DatasetBuilder()
        writer = tf.io.TFRecordWriter(output_path)
        count = 0
        
        for fire_id in fire_ids:
            try:
                fire_data = builder.load_fire_data(fire_id)
                if fire_data is None: continue
                
                dates = sorted(fire_data['progression'].get('daily_masks', {}).keys())
                max_start = len(dates) - builder.prediction_days - 1
                
                for start_day in range(min(max_start, 5)):
                    result = builder.build_sample(fire_data, start_day)
                    if result is None: continue
                    
                    input_tensor, output_tensor = result
                    aug_samples = get_d4_augmentations(input_tensor, output_tensor) if augment else [(input_tensor, output_tensor)]
                    
                    for aug_in, aug_out in aug_samples:
                        metadata = {
                            'fire_id': fire_id,
                            'start_date': dates[start_day] if start_day < len(dates) else ''
                        }
                        serialized = builder.serialize_sample(aug_in, aug_out, metadata)
                        writer.write(serialized)
                        count += 1
            except Exception as e:
                # Log but continue
                print(f"Error processing fire {fire_id}: {e}")
                continue
                
        writer.close()
        return count
    except Exception as e:
        print(f"Critical error in worker: {e}")
        return 0

class FastDatasetBuilder:
    def __init__(self):
        self.builder_ref = DatasetBuilder()
        self.env_dir = self.builder_ref.env_dir

    def build_dataset(self):
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("FAST DATASET BUILDER (Sharded Writing)")
        logger.info("=" * 60)
        
        env_files = glob.glob(os.path.join(self.env_dir, "env_*.json"))
        fire_ids = [os.path.basename(f).replace('env_', '').replace('.json', '') for f in env_files]
        
        if not fire_ids:
            logger.error("No data found.")
            return

        logger.info(f"Found {len(fire_ids)} fires.")
        random.shuffle(fire_ids)
        
        n_train = int(len(fire_ids) * TRAIN_RATIO)
        n_val = int(len(fire_ids) * VAL_RATIO)
        
        splits = [
            ('train', fire_ids[:n_train], True),
            ('val', fire_ids[n_train:n_train + n_val], False),
            ('test', fire_ids[n_train + n_val:], False)
        ]
        
        max_workers = min(6, os.cpu_count() or 4)
        logger.info(f"Using {max_workers} worker processes.")
        
        total_samples = 0
        
        for split_name, split_ids, augment in splits:
            final_output_path = os.path.join(TFRECORD_DIR, f"spread_{split_name}.tfrecord")
            logger.info(f"\nBuilding {split_name} split ({len(split_ids)} fires). Augment={augment}")
            
            # Split fires into chunks
            chunk_size = int(np.ceil(len(split_ids) / max_workers))
            chunks = [split_ids[i:i + chunk_size] for i in range(0, len(split_ids), chunk_size)]
            
            temp_files = []
            futures = []
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i, chunk in enumerate(chunks):
                    temp_file = os.path.join(TFRECORD_DIR, f"temp_{split_name}_{i}.tfrecord")
                    temp_files.append(temp_file)
                    futures.append(executor.submit(process_chunk_task, chunk, temp_file, augment))
                
                split_count = 0
                for i, future in enumerate(as_completed(futures)):
                    try:
                        count = future.result()
                        split_count += count
                        logger.info(f"  Worker finished chunk. Samples: {count}")
                    except Exception as e:
                        logger.error(f"  Worker failed: {e}")
            
            # Concatenate files
            logger.info(f"  Concatenating {len(temp_files)} chunks into {final_output_path}...")
            with open(final_output_path, 'wb') as outfile:
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        with open(temp_file, 'rb') as infile:
                            shutil.copyfileobj(infile, outfile)
                        os.remove(temp_file)
            
            logger.info(f"  Saved {split_count} samples to {final_output_path}")
            total_samples += split_count
            
            # Save info
            info_path = os.path.join(TFRECORD_DIR, f"spread_{split_name}_info.json")
            with open(info_path, 'w') as f:
                json.dump({
                    'split': split_name,
                    'num_fires': len(split_ids),
                    'num_samples': split_count,
                    'augmented': augment,
                    'augmentation_type': 'D4_8x' if augment else 'None'
                }, f, indent=2)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Dataset generation complete in {elapsed:.1f}s")
        logger.info(f"Total samples: {total_samples}")
        logger.info("=" * 60)

if __name__ == "__main__":
    builder = FastDatasetBuilder()
    builder.build_dataset()
