"""
PyroCast Fire Spread Model - Step 4: Build Training Dataset
============================================================
Combines all collected data into TFRecord format for efficient training:
1. Environmental data (terrain, fuel, weather)
2. Daily fire progression masks
3. Ignition points and previous fire state

Output: Train/Val/Test TFRecord files with data augmentation
"""

import os
import sys
import json
import glob
import random
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from shapely.geometry import shape, Point, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# TFRECORD FEATURE SPECIFICATION
# =============================================================================

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_array_feature(value):
    """Returns a float_list from a numpy array."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


class DatasetBuilder:
    """
    Builds TFRecord dataset for fire spread model training.
    
    Each sample contains:
    - Input: (T_in, H, W, C) environmental + fire state tensor
    - Output: (T_out, H, W, 1) future fire progression masks
    
    Where T_in = 1 (current state) and T_out = PREDICTION_DAYS (7)
    """
    
    def __init__(self):
        self.img_size = IMG_SIZE
        self.channels = INPUT_CHANNELS
        self.prediction_days = PREDICTION_DAYS
        
        # Paths
        self.env_dir = ENVIRONMENTAL_DIR
        self.prog_dir = PROGRESSIONS_DIR
        self.perim_dir = os.path.join(RAW_DATA_DIR, "perimeters")
        
        # Statistics for normalization (will be computed)
        self.stats = {}
        
    def load_fire_data(self, fire_id):
        """
        Load all data for a single fire event.
        
        Returns:
            dict with 'env_data', 'progression', 'perimeter' or None if missing
        """
        # Load environmental data
        env_path = os.path.join(self.env_dir, f"env_{fire_id}.json")
        if not os.path.exists(env_path):
            return None
        
        with open(env_path, 'r') as f:
            env_data = json.load(f)
        
        # Load progression data
        prog_path = os.path.join(self.prog_dir, f"progression_{fire_id}.json")
        if not os.path.exists(prog_path):
            return None
        
        with open(prog_path, 'r') as f:
            progression = json.load(f)
        
        # Load perimeter (optional, for final boundary)
        perim_path = os.path.join(self.perim_dir, f"perimeter_{fire_id}.geojson")
        perimeter = None
        if os.path.exists(perim_path):
            with open(perim_path, 'r') as f:
                perimeter = json.load(f)
        
        return {
            'fire_id': fire_id,
            'env_data': env_data,
            'progression': progression,
            'perimeter': perimeter
        }
    
    def create_fire_mask(self, geometry, bbox, img_size):
        """
        Rasterize a fire geometry to a binary mask.
        
        Args:
            geometry: GeoJSON geometry dict
            bbox: (min_lon, min_lat, max_lon, max_lat)
            img_size: output image size
        
        Returns:
            numpy array of shape (img_size, img_size) with 1s for fire, 0s elsewhere
        """
        try:
            fire_shape = shape(geometry)
            
            if fire_shape.is_empty:
                return np.zeros((img_size, img_size), dtype=np.float32)
            
            # Create pixel grid
            min_lon, min_lat, max_lon, max_lat = bbox
            lon_step = (max_lon - min_lon) / img_size
            lat_step = (max_lat - min_lat) / img_size
            
            mask = np.zeros((img_size, img_size), dtype=np.float32)
            
            for i in range(img_size):
                for j in range(img_size):
                    # Pixel center coordinates
                    px_lon = min_lon + (j + 0.5) * lon_step
                    px_lat = max_lat - (i + 0.5) * lat_step  # Note: lat decreases downward
                    
                    point = Point(px_lon, px_lat)
                    
                    if fire_shape.contains(point):
                        mask[i, j] = 1.0
            
            return mask
            
        except Exception as e:
            logger.warning(f"Error creating fire mask: {e}")
            return np.zeros((img_size, img_size), dtype=np.float32)
    
    def create_ignition_mask(self, first_day_geometry, bbox, img_size):
        """
        Create ignition point mask from first day's fire extent.
        Uses the centroid of the first detection as the ignition point.
        """
        try:
            fire_shape = shape(first_day_geometry)
            centroid = fire_shape.centroid
            
            min_lon, min_lat, max_lon, max_lat = bbox
            
            # Convert centroid to pixel coordinates
            px_x = int((centroid.x - min_lon) / (max_lon - min_lon) * img_size)
            px_y = int((max_lat - centroid.y) / (max_lat - min_lat) * img_size)
            
            # Create ignition mask with small Gaussian blob
            mask = np.zeros((img_size, img_size), dtype=np.float32)
            
            # Create small circular ignition point (radius ~5 pixels)
            for i in range(max(0, px_y - 5), min(img_size, px_y + 6)):
                for j in range(max(0, px_x - 5), min(img_size, px_x + 6)):
                    dist = np.sqrt((i - px_y)**2 + (j - px_x)**2)
                    if dist <= 5:
                        mask[i, j] = np.exp(-dist**2 / 8)  # Gaussian falloff
            
            # Ensure at least one pixel is marked
            if mask.sum() == 0:
                px_x = max(0, min(img_size - 1, px_x))
                px_y = max(0, min(img_size - 1, px_y))
                mask[px_y, px_x] = 1.0
            
            return mask
            
        except Exception as e:
            logger.warning(f"Error creating ignition mask: {e}")
            # Return center point as fallback
            mask = np.zeros((img_size, img_size), dtype=np.float32)
            mask[img_size // 2, img_size // 2] = 1.0
            return mask
    
    def build_sample(self, fire_data, start_day_idx):
        """
        Build a single training sample from fire data.
        
        Args:
            fire_data: dict from load_fire_data()
            start_day_idx: which day to use as ignition (0 = first day)
        
        Returns:
            (input_tensor, output_tensor) or None if invalid
        """
        env_data = fire_data['env_data']
        progression = fire_data['progression']
        
        # Get sorted dates
        dates = sorted(env_data.get('env_data', {}).keys())
        daily_masks = progression.get('daily_masks', {})
        mask_dates = sorted(daily_masks.keys())
        
        # Find common dates
        common_dates = sorted(set(dates) & set(mask_dates))
        
        if len(common_dates) < start_day_idx + self.prediction_days + 1:
            return None
        
        # Select date range
        ignition_date = common_dates[start_day_idx]
        future_dates = common_dates[start_day_idx + 1 : start_day_idx + 1 + self.prediction_days]
        
        if len(future_dates) < self.prediction_days:
            # Pad with last available date
            while len(future_dates) < self.prediction_days:
                future_dates.append(future_dates[-1] if future_dates else common_dates[-1])
        
        bbox = tuple(env_data['bbox'])
        
        # === BUILD INPUT TENSOR ===
        # Shape: (H, W, C) where C = 13 channels
        # Channels: fire_mask, elevation, slope, aspect, fuel_type, fuel_moisture,
        #           fuel_density, wind_u, wind_v, temperature, humidity, fwi, spread_potential
        
        input_tensor = np.zeros((self.img_size, self.img_size, self.channels), dtype=np.float32)
        
        # Channel 0: Current fire mask (fire extent at start of prediction)
        if ignition_date in daily_masks:
            fire_mask = self.create_fire_mask(
                daily_masks[ignition_date]['geometry'],
                bbox,
                self.img_size
            )
            input_tensor[:, :, 0] = fire_mask
        
        # Channels 1-12: Environmental data (12 channels)
        # Order: elevation, slope, aspect, fuel_type, fuel_moisture, fuel_density,
        #        wind_u, wind_v, temperature, humidity, fwi, spread_potential
        if ignition_date in env_data.get('env_data', {}):
            env_array = np.array(env_data['env_data'][ignition_date]['data'], dtype=np.float32)
            # Environmental data fills channels 1-12 (12 channels from Step 03)
            if len(env_array.shape) == 3 and env_array.shape[2] == 12:
                input_tensor[:, :, 1:13] = env_array
            elif len(env_array.shape) == 3 and env_array.shape[2] >= 12:
                # Take first 12 channels if more are provided
                input_tensor[:, :, 1:13] = env_array[:, :, :12]
        
        # === BUILD OUTPUT TENSOR ===
        # Shape: (PREDICTION_DAYS, H, W, 1)
        
        output_tensor = np.zeros((self.prediction_days, self.img_size, self.img_size, 1), dtype=np.float32)
        
        for day_idx, future_date in enumerate(future_dates):
            if future_date in daily_masks:
                fire_mask = self.create_fire_mask(
                    daily_masks[future_date]['geometry'],
                    bbox,
                    self.img_size
                )
                output_tensor[day_idx, :, :, 0] = fire_mask
        
        # Validate: output should have some fire pixels
        if output_tensor.sum() == 0:
            return None
        
        return input_tensor, output_tensor
    
    def augment_sample(self, input_tensor, output_tensor):
        """
        Apply data augmentation to a sample.
        
        Returns list of (augmented_input, augmented_output) tuples
        """
        augmented = [(input_tensor, output_tensor)]  # Original
        
        config = AUGMENTATION_CONFIG
        
        # Channel indices for new 13-channel structure:
        # 0: fire_mask, 1: elevation, 2: slope, 3: aspect
        # 4: fuel_type, 5: fuel_moisture, 6: fuel_density
        # 7: wind_u, 8: wind_v, 9: temperature, 10: humidity
        # 11: fwi, 12: spread_potential
        
        ASPECT_IDX = 3
        WIND_U_IDX = 7
        WIND_V_IDX = 8
        
        # Horizontal flip (flip along W axis)
        if config.get('horizontal_flip', True):
            aug_in = np.flip(input_tensor, axis=1).copy()
            aug_out = np.flip(output_tensor, axis=2).copy()
            
            # Flip wind_u (east-west component) and aspect
            aug_in[:, :, WIND_U_IDX] = -aug_in[:, :, WIND_U_IDX]
            aug_in[:, :, ASPECT_IDX] = np.pi - aug_in[:, :, ASPECT_IDX]  # Mirror aspect
            
            augmented.append((aug_in, aug_out))
        
        # Vertical flip (flip along H axis)
        if config.get('vertical_flip', True):
            aug_in = np.flip(input_tensor, axis=0).copy()
            aug_out = np.flip(output_tensor, axis=1).copy()
            
            # Flip wind_v (north-south component) and aspect
            aug_in[:, :, WIND_V_IDX] = -aug_in[:, :, WIND_V_IDX]
            aug_in[:, :, ASPECT_IDX] = -aug_in[:, :, ASPECT_IDX]  # Flip aspect
            
            augmented.append((aug_in, aug_out))
        
        # 90-degree rotations
        if config.get('rotation_90', True):
            for k in [1, 2, 3]:  # 90, 180, 270 degrees
                aug_in = np.rot90(input_tensor, k=k, axes=(0, 1)).copy()
                aug_out = np.rot90(output_tensor, k=k, axes=(1, 2)).copy()
                
                # Rotate wind vector (u, v) by k*90 degrees
                angle = k * np.pi / 2
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                wind_u = aug_in[:, :, WIND_U_IDX].copy()
                wind_v = aug_in[:, :, WIND_V_IDX].copy()
                aug_in[:, :, WIND_U_IDX] = wind_u * cos_a - wind_v * sin_a
                aug_in[:, :, WIND_V_IDX] = wind_u * sin_a + wind_v * cos_a
                
                # Rotate aspect by same angle
                aug_in[:, :, ASPECT_IDX] = aug_in[:, :, ASPECT_IDX] + angle
                
                augmented.append((aug_in, aug_out))
        
        # Wind perturbation (add noise to wind vector components)
        if config.get('wind_perturbation', 0) > 0:
            noise_level = config['wind_perturbation']
            aug_in = input_tensor.copy()
            
            # Add noise to wind_u and wind_v (channels 7, 8)
            aug_in[:, :, WIND_U_IDX] += np.random.normal(0, noise_level, aug_in[:, :, WIND_U_IDX].shape)
            aug_in[:, :, WIND_V_IDX] += np.random.normal(0, noise_level, aug_in[:, :, WIND_V_IDX].shape)
            
            augmented.append((aug_in, output_tensor))
        
        return augmented[:AUGMENTATION_FACTOR]  # Limit to config factor
    
    def serialize_sample(self, input_tensor, output_tensor, metadata):
        """
        Serialize a sample to TFRecord format.
        """
        feature = {
            'input': _float_array_feature(input_tensor),
            'output': _float_array_feature(output_tensor),
            'input_shape': _bytes_feature(tf.io.serialize_tensor(tf.constant(input_tensor.shape))),
            'output_shape': _bytes_feature(tf.io.serialize_tensor(tf.constant(output_tensor.shape))),
            'fire_id': _bytes_feature(metadata['fire_id'].encode()),
            'start_date': _bytes_feature(metadata.get('start_date', '').encode()),
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()
    
    def build_dataset(self):
        """
        Build the complete TFRecord dataset.
        """
        logger.info("=" * 60)
        logger.info("PyroCast Fire Spread Model - Dataset Builder")
        logger.info("=" * 60)
        
        # Get list of fires with environmental data
        env_files = glob.glob(os.path.join(self.env_dir, "env_*.json"))
        fire_ids = [os.path.basename(f).replace('env_', '').replace('.json', '') 
                   for f in env_files]
        
        if not fire_ids:
            logger.error("No environmental data found. Run previous steps first.")
            return
        
        logger.info(f"Found {len(fire_ids)} fires with environmental data")
        
        # Shuffle and split
        random.shuffle(fire_ids)
        
        n_train = int(len(fire_ids) * TRAIN_RATIO)
        n_val = int(len(fire_ids) * VAL_RATIO)
        
        train_ids = fire_ids[:n_train]
        val_ids = fire_ids[n_train:n_train + n_val]
        test_ids = fire_ids[n_train + n_val:]
        
        logger.info(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        
        # Build each split
        splits = [
            ('train', train_ids, True),   # Augment training data
            ('val', val_ids, False),
            ('test', test_ids, False)
        ]
        
        for split_name, split_ids, augment in splits:
            self._build_split(split_name, split_ids, augment)
        
        logger.info("=" * 60)
        logger.info("Dataset building complete!")
        logger.info(f"Output directory: {TFRECORD_DIR}")
        logger.info("=" * 60)
    
    def _build_split(self, split_name, fire_ids, augment):
        """Build a single data split (train/val/test)."""
        
        output_path = os.path.join(TFRECORD_DIR, f"spread_{split_name}.tfrecord")
        
        logger.info(f"\nBuilding {split_name} split ({len(fire_ids)} fires)...")
        
        writer = tf.io.TFRecordWriter(output_path)
        sample_count = 0
        
        for idx, fire_id in enumerate(fire_ids):
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processing fire {idx + 1}/{len(fire_ids)}...")
            
            # Load fire data
            fire_data = self.load_fire_data(fire_id)
            if fire_data is None:
                continue
            
            # Generate samples from different starting days
            dates = sorted(fire_data['progression'].get('daily_masks', {}).keys())
            max_start = len(dates) - self.prediction_days - 1
            
            for start_day in range(min(max_start, 5)):  # Up to 5 samples per fire
                result = self.build_sample(fire_data, start_day)
                
                if result is None:
                    continue
                
                input_tensor, output_tensor = result
                
                # Augment if training
                if augment:
                    samples = self.augment_sample(input_tensor, output_tensor)
                else:
                    samples = [(input_tensor, output_tensor)]
                
                # Write samples
                for aug_in, aug_out in samples:
                    metadata = {
                        'fire_id': fire_id,
                        'start_date': dates[start_day] if start_day < len(dates) else ''
                    }
                    
                    serialized = self.serialize_sample(aug_in, aug_out, metadata)
                    writer.write(serialized)
                    sample_count += 1
        
        writer.close()
        
        logger.info(f"  {split_name}: {sample_count} samples written to {output_path}")
        
        # Save split info
        info_path = os.path.join(TFRECORD_DIR, f"spread_{split_name}_info.json")
        with open(info_path, 'w') as f:
            json.dump({
                'split': split_name,
                'num_fires': len(fire_ids),
                'num_samples': sample_count,
                'fire_ids': fire_ids,
                'input_shape': [self.img_size, self.img_size, self.channels],
                'output_shape': [self.prediction_days, self.img_size, self.img_size, 1],
                'augmented': augment
            }, f, indent=2)


def create_tf_dataset(tfrecord_path, batch_size=BATCH_SIZE, shuffle=True):
    """
    Create a tf.data.Dataset from TFRecord file for training.
    
    Usage:
        train_ds = create_tf_dataset('spread_train.tfrecord', batch_size=8)
        for inputs, outputs in train_ds:
            # inputs: (batch, 256, 256, 13)
            # outputs: (batch, 7, 256, 256, 1)
            ...
    """
    
    # Feature description for parsing
    feature_description = {
        'input': tf.io.FixedLenFeature([IMG_SIZE * IMG_SIZE * INPUT_CHANNELS], tf.float32),
        'output': tf.io.FixedLenFeature([PREDICTION_DAYS * IMG_SIZE * IMG_SIZE * 1], tf.float32),
        'input_shape': tf.io.FixedLenFeature([], tf.string),
        'output_shape': tf.io.FixedLenFeature([], tf.string),
        'fire_id': tf.io.FixedLenFeature([], tf.string),
        'start_date': tf.io.FixedLenFeature([], tf.string),
    }
    
    def parse_fn(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Reshape tensors
        input_tensor = tf.reshape(parsed['input'], [IMG_SIZE, IMG_SIZE, INPUT_CHANNELS])
        output_tensor = tf.reshape(parsed['output'], [PREDICTION_DAYS, IMG_SIZE, IMG_SIZE, 1])
        
        return input_tensor, output_tensor
    
    # Create dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def main():
    """Main entry point."""
    builder = DatasetBuilder()
    builder.build_dataset()
    
    # Print dataset statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        info_path = os.path.join(TFRECORD_DIR, f"spread_{split}_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            print(f"\n{split.upper()}:")
            print(f"  Fires: {info['num_fires']}")
            print(f"  Samples: {info['num_samples']}")
            print(f"  Augmented: {info['augmented']}")


if __name__ == "__main__":
    main()
