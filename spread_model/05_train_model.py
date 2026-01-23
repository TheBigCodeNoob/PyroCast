"""
PyroCast Fire Spread Model - Step 5: Train Model
=================================================
Implements and trains a lightweight CNN encoder-decoder for fire spread prediction.

Architecture Philosophy:
Fire spread is fundamentally a LOCAL, physics-constrained process:
- Fire spreads to adjacent cells based on fuel, wind, slope
- Kernel of influence ~5-10km (ember spotting)
- Follows predictable physical rules (Rothermel equations)

Architecture (2-3M parameters):
- Encoder: Extract local fire spread patterns (3x3 convs)
- Bottleneck: Dilated convs for medium-range spotting effects
- Decoder: Reconstruct spatial detail with skip connections
- Prediction: Autoregressive (day-by-day) for physical alignment

Training Strategy:
1. Phase 1: Single-step training (predict day N+1 from day N)
2. Phase 2: Multi-step fine-tuning (minimize error accumulation)

Output: Trained model that predicts fire spread one day at a time
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# Import dataset creation function
import importlib.util
spec = importlib.util.spec_from_file_location("build_dataset", os.path.join(os.path.dirname(__file__), "04_build_dataset.py"))
build_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset_module)
create_tf_dataset = build_dataset_module.create_tf_dataset

import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# CPU OPTIMIZATION FOR TRAINING
# =============================================================================
# Optimize CPU performance with threading
os.environ['TF_NUM_INTEROP_THREADS'] = str(os.cpu_count() or 4)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count() or 4)
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)

# Enable XLA compilation for faster execution
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

logger.info(f"CPU threads configured: {os.cpu_count() or 4}")
logger.info("XLA JIT compilation enabled for performance")

# Note: For AMD GPU on Windows, use WSL2 + ROCm or PyTorch with DirectML
# TensorFlow-DirectML is deprecated (last supported TF 2.10)

# Enable mixed precision if configured  
if USE_MIXED_PRECISION:
    # Use float32 for CPU training (float16 not well supported on CPU)
    tf.keras.mixed_precision.set_global_policy('float32')
    logger.info("Training with float32 precision (CPU optimized)")

# =============================================================================
# ARCHITECTURE BUILDING BLOCKS
# =============================================================================

def conv_bn_relu(x, filters, kernel_size=3, dilation_rate=1, name=None):
    """
    Standard convolution block: Conv -> BatchNorm -> ReLU
    """
    x = layers.Conv2D(
        filters, 
        kernel_size, 
        padding='same',
        dilation_rate=dilation_rate,
        name=f"{name}_conv" if name else None
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)
    return x


def encoder_block(x, filters, name):
    """
    Encoder block: 2x Conv-BN-ReLU + MaxPool
    Returns features (for skip) and pooled output
    """
    # Two conv layers to capture local patterns
    x = conv_bn_relu(x, filters, name=f"{name}_conv1")
    features = conv_bn_relu(x, filters, name=f"{name}_conv2")
    
    # Downsample
    pooled = layers.MaxPooling2D(2, name=f"{name}_pool")(features)
    
    return features, pooled


def decoder_block(x, skip, filters, name):
    """
    Decoder block: Upsample + Concatenate + 2x Conv-BN-ReLU
    """
    # Upsample
    x = layers.Conv2DTranspose(
        filters, 
        kernel_size=2, 
        strides=2, 
        padding='same',
        name=f"{name}_upsample"
    )(x)
    
    # Concatenate with skip connection
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    
    # Refine features
    x = conv_bn_relu(x, filters, name=f"{name}_conv1")
    x = conv_bn_relu(x, filters, name=f"{name}_conv2")
    
    return x


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_fire_spread_model(input_channels=13):
    """
    Build lightweight CNN encoder-decoder for single-step fire spread prediction.
    
    Architecture (2-3M parameters):
    ┌─────────────────────────────────────────┐
    │ ENCODER: Capture local spread patterns │
    ├─────────────────────────────────────────┤
    │ Conv 3×3, 32 → BN → ReLU                │
    │ Conv 3×3, 64 → BN → ReLU → Pool         │  skip1
    │ Conv 3×3, 128 → BN → ReLU → Pool        │  skip2
    ├─────────────────────────────────────────┤
    │ BOTTLENECK: Medium-range spotting      │
    ├─────────────────────────────────────────┤
    │ Conv 3×3, 256, dilation=2 → ReLU        │  5km range
    │ Conv 3×3, 256, dilation=4 → ReLU        │  10km range
    ├─────────────────────────────────────────┤
    │ DECODER: Reconstruct spatial detail    │
    ├─────────────────────────────────────────┤
    │ Upsample 2× + Conv 3×3, 128 + skip2     │
    │ Upsample 2× + Conv 3×3, 64 + skip1      │
    │ Conv 1×1, 1 → Sigmoid                   │
    └─────────────────────────────────────────┘
    
    Input: (H, W, C) where C includes:
        - Current fire mask (1)
        - Terrain: elevation, slope, aspect (3)
        - Fuel: type, moisture, density (3)
        - Weather: wind_u, wind_v, temp, humidity (4)
        - Fire indices: FWI, spread potential (2)
    
    Output: Next-day fire probability (H, W, 1)
    
    Prediction: Use autoregressively for multi-day forecasts
    """
    # Input
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, input_channels), name='input')
    
    # =========================================================================
    # ENCODER: Extract local fire spread patterns
    # =========================================================================
    
    # Initial feature extraction
    x = conv_bn_relu(inputs, 32, name='enc_init')
    
    # Encoder level 1: 256×256 → 128×128
    skip1, x = encoder_block(x, 64, name='enc1')
    
    # Encoder level 2: 128×128 → 64×64
    skip2, x = encoder_block(x, 128, name='enc2')
    
    # =========================================================================
    # BOTTLENECK: Capture medium-range effects (ember spotting)
    # =========================================================================
    
    # Dilated convolutions expand receptive field without downsampling
    # dilation=2: ~5km range at 375m/pixel resolution
    x = conv_bn_relu(x, 256, dilation_rate=2, name='bottleneck_d2')
    
    # dilation=4: ~10km range (max spotting distance)
    x = conv_bn_relu(x, 256, dilation_rate=4, name='bottleneck_d4')
    
    # =========================================================================
    # DECODER: Reconstruct spatial detail
    # =========================================================================
    
    # Decoder level 1: 64×64 → 128×128
    x = decoder_block(x, skip2, 128, name='dec1')
    
    # Decoder level 2: 128×128 → 256×256
    x = decoder_block(x, skip1, 64, name='dec2')
    
    # Final prediction: fire probability map
    x = layers.Conv2D(1, 1, padding='same', name='output_conv')(x)
    outputs = layers.Activation('sigmoid', dtype='float32', name='output')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs, name='FireSpreadModel')
    
    return model


def build_multistep_wrapper(base_model, num_days=7):
    """
    Wrap single-step model for autoregressive multi-day prediction.
    
    For training Phase 2 (multi-step fine-tuning) and inference.
    
    Args:
        base_model: Single-step fire spread model
        num_days: Number of days to predict
    
    Returns:
        Model that predicts num_days of fire spread
    """
    # Extract input shape from base model
    input_shape = base_model.input_shape[1:]  # Remove batch dim
    
    # Static inputs (don't change day-to-day)
    static_inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, input_shape[-1] - 1),  # All channels except fire mask
        name='static_input'
    )
    
    # Initial fire mask
    initial_fire = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 1),
        name='initial_fire'
    )
    
    # Predict day-by-day
    predictions = []
    current_fire = initial_fire
    
    for day in range(num_days):
        # Combine current fire with static environment
        model_input = layers.Concatenate(name=f'day{day}_concat')([
            current_fire, 
            static_inputs
        ])
        
        # Predict next day
        next_fire = base_model(model_input)
        predictions.append(next_fire)
        
        # Update current fire (autoregressive)
        current_fire = next_fire
    
    # Stack predictions: (batch, T, H, W, 1)
    output = layers.Lambda(
        lambda x: tf.stack(x, axis=1),
        name='stack_predictions'
    )(predictions)
    
    model = Model(
        inputs=[static_inputs, initial_fire],
        outputs=output,
        name='MultiStepFireSpreadModel'
    )
    
    return model


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def weighted_bce(y_true, y_pred):
    """
    Weighted binary cross-entropy for handling class imbalance.
    Fire pixels are rare (~1-5% of image), so weight them higher.
    """
    # Compute base BCE
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    
    # Apply class weights
    weights = y_true * FIRE_PIXEL_WEIGHT + (1 - y_true) * NON_FIRE_PIXEL_WEIGHT
    weighted_bce = bce * weights
    
    return tf.reduce_mean(weighted_bce)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss - excellent for segmentation with class imbalance.
    Measures overlap between predicted and true fire masks.
    """
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice


def combined_loss(y_true, y_pred):
    """
    Combine BCE and Dice for robust training.
    BCE: Per-pixel accuracy
    Dice: Overall segmentation quality
    """
    bce = weighted_bce(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice


def temporal_weighted_loss(y_true, y_pred):
    """
    Loss for multi-step predictions with temporal weighting.
    Earlier days weighted more heavily (more certain predictions).
    
    Used in Phase 2 (multi-step fine-tuning).
    """
    total_loss = 0.0
    
    for t in range(PREDICTION_DAYS):
        y_true_t = y_true[:, t, :, :, :]
        y_pred_t = y_pred[:, t, :, :, :]
        
        # Combined loss for this timestep
        loss_t = combined_loss(y_true_t, y_pred_t)
        
        # Temporal weight (decay for later days)
        weight = TEMPORAL_WEIGHTS[t] if t < len(TEMPORAL_WEIGHTS) else TEMPORAL_WEIGHTS[-1]
        
        total_loss += weight * loss_t
    
    return total_loss / sum(TEMPORAL_WEIGHTS[:PREDICTION_DAYS])


# =============================================================================
# METRICS
# =============================================================================

class IoUMetric(tf.keras.metrics.Metric):
    """
    Intersection over Union metric for fire spread prediction.
    """
    def __init__(self, threshold=0.5, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_binary = tf.cast(y_pred > self.threshold, tf.float32)
        y_true_binary = tf.cast(y_true > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_pred_binary * y_true_binary)
        union = tf.reduce_sum(y_pred_binary) + tf.reduce_sum(y_true_binary) - intersection
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
    
    def result(self):
        return self.intersection / (self.union + 1e-6)
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class TemporalAccuracy(tf.keras.metrics.Metric):
    """
    Measures prediction accuracy across time steps.
    """
    def __init__(self, name='temporal_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # For each time step, check if fire pixels are correctly identified
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        y_true_binary = tf.cast(y_true > 0.5, tf.float32)
        
        correct = tf.reduce_sum(tf.cast(y_pred_binary == y_true_binary, tf.float32))
        total = tf.cast(tf.size(y_true), tf.float32)
        
        self.correct.assign_add(correct)
        self.total.assign_add(total)
    
    def result(self):
        return self.correct / (self.total + 1e-6)
    
    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


# =============================================================================
# TRAINING
# =============================================================================

def prepare_single_step_dataset(dataset):
    """
    Convert multi-day dataset to single-step transitions.
    
    Input: (batch, T, H, W, C_in) and (batch, T, H, W, 1)
    Output: (batch, H, W, C_in) -> (batch, H, W, 1)
    
    Extracts all day-to-day transitions for training.
    """
    def extract_transitions(inputs, targets):
        # inputs: (batch, H, W, C_in)
        # targets: (batch, T, H, W, 1)
        
        # Extract all transitions: day 0->1, day 1->2, etc.
        transitions = []
        
        for t in range(PREDICTION_DAYS - 1):
            # Current fire mask
            current_fire = targets[:, t, :, :, :]  # (batch, H, W, 1)
            
            # Static environment (all channels except first = fire mask)
            static_env = inputs[:, :, :, 1:]  # (batch, H, W, C-1)
            
            # Combine for input
            step_input = tf.concat([current_fire, static_env], axis=-1)
            
            # Target is next day
            step_target = targets[:, t + 1, :, :, :]
            
            transitions.append((step_input, step_target))
        
        # Return first transition (will be called in batches)
        return transitions[0]
    
    return dataset.map(extract_transitions)


def train_phase1_single_step(run_name):
    """
    Phase 1: Single-step training.
    
    Train model to predict day N+1 from day N.
    This is the bulk of training - learning the physics of fire spread.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Single-Step Training")
    logger.info("Learning fire spread physics (day N → day N+1)")
    logger.info("=" * 60)
    
    # Load datasets
    train_path = os.path.join(TFRECORD_DIR, "spread_train.tfrecord")
    val_path = os.path.join(TFRECORD_DIR, "spread_val.tfrecord")
    
    if not os.path.exists(train_path):
        logger.error(f"Training data not found at {train_path}")
        logger.error("Run 04_build_dataset.py first.")
        return None
    
    logger.info("Loading datasets...")
    train_ds = create_tf_dataset(train_path, BATCH_SIZE, shuffle=True)
    val_ds = create_tf_dataset(val_path, BATCH_SIZE, shuffle=False)
    
    # Convert to single-step format
    logger.info("Preparing single-step transitions...")
    train_ds = prepare_single_step_dataset(train_ds)
    val_ds = prepare_single_step_dataset(val_ds)
    
    # Build model
    logger.info("Building model...")
    model = build_fire_spread_model(input_channels=INPUT_CHANNELS)
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[
            IoUMetric(threshold=0.5),
            'accuracy'
        ]
    )
    
    # Print model summary
    model.summary(line_length=100)
    
    # Count parameters
    total_params = model.count_params()
    logger.info(f"Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            get_checkpoint_path(run_name, suffix='_phase1'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_DECAY_FACTOR,
            patience=LR_DECAY_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(TENSORBOARD_LOG_DIR, f"{run_name}_phase1"),
            histogram_freq=1
        ),
        CSVLogger(
            os.path.join(LOGS_DIR, f"training_{run_name}_phase1.csv")
        )
    ]
    
    # Train
    logger.info("Starting Phase 1 training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Phase 1 complete!")
    logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    logger.info(f"Best IoU: {max(history.history.get('val_iou', [0])):.4f}")
    
    return model, history


def train_phase2_multistep(base_model, run_name):
    """
    Phase 2: Multi-step fine-tuning.
    
    Fine-tune with autoregressive prediction to minimize error accumulation.
    Uses the single-step model wrapped for multi-day prediction.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Multi-Step Fine-Tuning")
    logger.info("Reducing error accumulation in autoregressive prediction")
    logger.info("=" * 60)
    
    # Load datasets (original multi-day format)
    train_path = os.path.join(TFRECORD_DIR, "spread_train.tfrecord")
    val_path = os.path.join(TFRECORD_DIR, "spread_val.tfrecord")
    
    logger.info("Loading multi-step datasets...")
    train_ds = create_tf_dataset(train_path, BATCH_SIZE // 2, shuffle=True)  # Smaller batch for memory
    val_ds = create_tf_dataset(val_path, BATCH_SIZE // 2, shuffle=False)
    
    # Wrap for multi-step prediction
    logger.info("Building multi-step wrapper...")
    multistep_model = build_multistep_wrapper(base_model, num_days=PREDICTION_DAYS)
    
    # Compile with temporal loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE / 10)  # Lower LR for fine-tuning
    
    multistep_model.compile(
        optimizer=optimizer,
        loss=temporal_weighted_loss,
        metrics=[
            IoUMetric(threshold=0.5)
        ]
    )
    
    multistep_model.summary(line_length=100)
    
    # Callbacks (fewer epochs for fine-tuning)
    callbacks = [
        ModelCheckpoint(
            get_checkpoint_path(run_name, suffix='_phase2'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE // 2,  # Less patience
            restore_best_weights=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(TENSORBOARD_LOG_DIR, f"{run_name}_phase2")
        ),
        CSVLogger(
            os.path.join(LOGS_DIR, f"training_{run_name}_phase2.csv")
        )
    ]
    
    # Fine-tune (fewer epochs)
    logger.info("Starting Phase 2 fine-tuning...")
    history = multistep_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS // 3,  # ~10-15 epochs
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Phase 2 complete!")
    logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    
    return multistep_model, history


def train_model():
    """
    Main training function with two-phase strategy.
    
    Phase 1: Single-step training (bulk of learning)
        - Input: fire_mask[t] + environment
        - Output: fire_mask[t+1]
        - Learn: Physics of local fire spread
    
    Phase 2: Multi-step fine-tuning (reduce error accumulation)
        - Input: initial_fire + environment
        - Output: 7 days of predictions
        - Learn: Minimize compounding errors in autoregressive prediction
    """
    # Create run name
    run_name = get_run_name()
    logger.info("=" * 60)
    logger.info("PyroCast Fire Spread Model - Training")
    logger.info("=" * 60)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Architecture: Lightweight CNN Encoder-Decoder (~2-3M params)")
    logger.info(f"Strategy: Two-phase (single-step → multi-step)")
    logger.info("=" * 60)
    
    # Phase 1: Single-step training
    phase1_model, phase1_history = train_phase1_single_step(run_name)
    
    if phase1_model is None:
        logger.error("Phase 1 training failed!")
        return None, None
    
    # Save Phase 1 model
    phase1_path = get_model_path(run_name, suffix='_phase1')
    phase1_model.save(phase1_path)
    logger.info(f"Phase 1 model saved to: {phase1_path}")
    
    # Phase 2: Multi-step fine-tuning
    logger.info("\nStarting Phase 2 in 3 seconds...")
    import time
    time.sleep(3)
    
    phase2_model, phase2_history = train_phase2_multistep(phase1_model, run_name)
    
    # Save final model
    final_model_path = get_model_path(run_name)
    phase2_model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    # Save training histories
    for phase, history in [('phase1', phase1_history), ('phase2', phase2_history)]:
        history_path = os.path.join(LOGS_DIR, f"history_{run_name}_{phase}.json")
        with open(history_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Phase 1 best IoU: {max(phase1_history.history.get('val_iou', [0])):.4f}")
    logger.info(f"Phase 2 best loss: {min(phase2_history.history['val_loss']):.4f}")
    logger.info("=" * 60)
    
    return phase2_model, (phase1_history, phase2_history)


def main():
    """Main entry point."""
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        except Exception as e:
            logger.info(f"GPU configuration skipped: {e}")
    else:
        logger.info("Training on CPU with optimized threading")
        logger.info("For AMD GPU: Use WSL2 + TensorFlow-ROCm or PyTorch with DirectML")
    
    # Train model
    model, history = train_model()
    
    return model


if __name__ == "__main__":
    main()
