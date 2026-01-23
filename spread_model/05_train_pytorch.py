"""
PyroCast Fire Spread Model - PyTorch Training with DirectML (AMD GPU)
======================================================================
PyTorch implementation with DirectML backend for AMD GPU acceleration on Windows.

Architecture: Same CNN encoder-decoder as TensorFlow version (~1.9M params)
"""

import os
import sys
import json
import glob
import time
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# DirectML for AMD GPU (requires Python 3.10 or lower)
DML_AVAILABLE = False
DML_ERROR = None
try:
    import torch_directml
    DML_AVAILABLE = True
    print(f"✓ torch_directml imported successfully")
    print(f"  DirectML version: {torch_directml.__version__ if hasattr(torch_directml, '__version__') else 'unknown'}")
except ImportError as e:
    DML_ERROR = str(e)
    print(f"✗ Failed to import torch_directml: {e}")
    print(f"  Current Python version: {sys.version}")
    print(f"  torch_directml requires Python 3.10 or lower")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# DEVICE SETUP
# =============================================================================

def get_device():
    """Get the best available device (DirectML > CUDA > CPU)."""
    logger.info("=" * 60)
    logger.info("DEVICE DETECTION")
    logger.info("=" * 60)
    
    if DML_AVAILABLE:
        try:
            # Force DirectML to not pre-allocate large memory pools
            os.environ['PYTORCH_DIRECTML_DISABLE_MEMORY_POOL'] = '1'
            
            device = torch_directml.device()
            logger.info(f"✓ Using DirectML (AMD GPU)")
            logger.info(f"  Device: {device}")
            logger.info(f"  Memory pool caching: DISABLED (prevents allocator issues)")
            # Test if device actually works
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
            logger.info(f"  Device test: PASSED (tensor created successfully)")
            return device
        except Exception as e:
            logger.error(f"✗ DirectML available but failed to initialize: {e}")
            logger.error("  Falling back to CPU...")
    elif DML_ERROR:
        logger.warning(f"✗ DirectML NOT available: {DML_ERROR}")
        logger.warning(f"  Python version: {sys.version.split()[0]}")
        logger.warning(f"  Required: Python 3.10 or lower + torch-directml package")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
        return device
    
    # Fallback to CPU
    device = torch.device('cpu')
    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(os.cpu_count() or 4)
    logger.warning(f"✗ Using CPU with {os.cpu_count()} threads")
    logger.warning("  For AMD GPU acceleration:")
    logger.warning("    1. Create conda env with Python 3.10: conda create -n pyrocast_gpu python=3.10")
    logger.warning("    2. Install torch-directml: pip install torch-directml")
    logger.info("=" * 60)
    return device

# =============================================================================
# DATASET - Load from TFRecord
# =============================================================================

class FireSpreadDataset(Dataset):
    """
    PyTorch Dataset using memory-mapped files for efficient large dataset handling.
    Converts TFRecord to memmap format (disk-backed, OS manages memory).
    """
    def __init__(self, tfrecord_path, single_step=True):
        """
        Args:
            tfrecord_path: Path to .tfrecord file
            single_step: If True, extract day-to-day transitions for Phase 1
        """
        self.single_step = single_step
        
        # Memory-mapped file paths
        cache_dir = os.path.dirname(tfrecord_path)
        base_name = os.path.basename(tfrecord_path).replace('.tfrecord', '')
        self.input_mmap_path = os.path.join(cache_dir, f"{base_name}_inputs.npy")
        self.output_mmap_path = os.path.join(cache_dir, f"{base_name}_outputs.npy")
        self.meta_path = os.path.join(cache_dir, f"{base_name}_meta.json")
        
        # Load or create memory-mapped cache
        if os.path.exists(self.meta_path):
            logger.info(f"Loading memory-mapped dataset...")
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
            self.record_count = meta['record_count']
            
            # Load as memory-mapped (disk-backed, minimal RAM usage)
            self.inputs = np.load(self.input_mmap_path, mmap_mode='r')
            self.outputs = np.load(self.output_mmap_path, mmap_mode='r')
            logger.info(f"Loaded {self.record_count} records (memory-mapped)")
        else:
            logger.info(f"Creating memory-mapped cache from {tfrecord_path}...")
            self._create_memmap_cache(tfrecord_path)
        
        # For single_step, we get (PREDICTION_DAYS - 1) transitions per record
        if single_step:
            self.transitions_per_record = PREDICTION_DAYS - 1
            self.total_samples = self.record_count * self.transitions_per_record
        else:
            self.transitions_per_record = 1
            self.total_samples = self.record_count
        
        logger.info(f"Dataset ready: {self.record_count} records, {self.total_samples} samples")
    
    def _create_memmap_cache(self, tfrecord_path):
        """Convert TFRecord to memory-mapped numpy arrays (streaming, low memory)."""
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        
        feature_description = {
            'input': tf.io.FixedLenFeature([IMG_SIZE * IMG_SIZE * INPUT_CHANNELS], tf.float32),
            'output': tf.io.FixedLenFeature([PREDICTION_DAYS * IMG_SIZE * IMG_SIZE * 1], tf.float32),
        }
        
        # First pass: count records
        logger.info("  Counting records...")
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        record_count = sum(1 for _ in dataset)
        logger.info(f"  Found {record_count} records")
        
        # Create memory-mapped files
        input_shape = (record_count, IMG_SIZE, IMG_SIZE, INPUT_CHANNELS)
        output_shape = (record_count, PREDICTION_DAYS, IMG_SIZE, IMG_SIZE, 1)
        
        # Use float16 to reduce disk/memory usage by 50%
        inputs_mmap = np.lib.format.open_memmap(
            self.input_mmap_path, mode='w+', dtype=np.float16, shape=input_shape
        )
        outputs_mmap = np.lib.format.open_memmap(
            self.output_mmap_path, mode='w+', dtype=np.float16, shape=output_shape
        )
        
        # Second pass: write data
        logger.info("  Writing data to memory-mapped files...")
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        
        for i, raw_record in enumerate(dataset):
            if (i + 1) % 2000 == 0:
                logger.info(f"    Processed {i + 1}/{record_count} records...")
            
            parsed = tf.io.parse_single_example(raw_record, feature_description)
            
            inputs_mmap[i] = parsed['input'].numpy().reshape(IMG_SIZE, IMG_SIZE, INPUT_CHANNELS).astype(np.float16)
            outputs_mmap[i] = parsed['output'].numpy().reshape(PREDICTION_DAYS, IMG_SIZE, IMG_SIZE, 1).astype(np.float16)
        
        # Flush to disk
        del inputs_mmap
        del outputs_mmap
        
        # Save metadata
        with open(self.meta_path, 'w') as f:
            json.dump({'record_count': record_count}, f)
        
        # Reload as read-only memmap
        self.record_count = record_count
        self.inputs = np.load(self.input_mmap_path, mmap_mode='r')
        self.outputs = np.load(self.output_mmap_path, mmap_mode='r')
        
        logger.info(f"  Cache created successfully")
        
        # CRITICAL: Clear TensorFlow memory before GPU training
        import gc
        gc.collect()
        tf.keras.backend.clear_session()
        logger.info(f"  Cleared TensorFlow memory")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if self.single_step:
            # Map sample index to record index and transition index
            record_idx = idx // self.transitions_per_record
            transition_idx = idx % self.transitions_per_record
            
            # Load from memmap (OS handles caching efficiently)
            input_tensor = np.array(self.inputs[record_idx], dtype=np.float32)
            output_tensor = np.array(self.outputs[record_idx], dtype=np.float32)
            
            # Extract transition t -> t+1
            t = transition_idx
            current_fire = output_tensor[t, :, :, :]  # (H, W, 1)
            static_env = input_tensor[:, :, 1:]  # (H, W, 12)
            
            step_input = np.concatenate([current_fire, static_env], axis=-1)  # (H, W, 13)
            step_target = output_tensor[t + 1, :, :, :]  # (H, W, 1)
            
            # Convert to PyTorch format: (C, H, W)
            input_pt = torch.from_numpy(step_input).permute(2, 0, 1)
            target_pt = torch.from_numpy(step_target).permute(2, 0, 1)
            
            return input_pt, target_pt
        else:
            input_tensor = np.array(self.inputs[idx], dtype=np.float32)
            output_tensor = np.array(self.outputs[idx], dtype=np.float32)
            
            input_pt = torch.from_numpy(input_tensor).permute(2, 0, 1)
            target_pt = torch.from_numpy(output_tensor).permute(0, 3, 1, 2)
            
            return input_pt, target_pt

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ConvBNReLU(nn.Module):
    """Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """2x Conv-BN-ReLU + MaxPool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        features = self.conv2(self.conv1(x))
        pooled = self.pool(features)
        return features, pooled


class SpatialAttention(nn.Module):
    """Spatial attention to focus on fire-prone areas"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class DecoderBlock(nn.Module):
    """Upsample + Concat + Attention + 2x Conv-BN-ReLU"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = SpatialAttention(out_channels + skip_channels) if use_attention else None
        self.conv1 = ConvBNReLU(out_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        if self.attention:
            x = self.attention(x)
        return self.conv2(self.conv1(x))


class FireSpreadModel(nn.Module):
    """
    CNN Encoder-Decoder for single-step fire spread prediction.
    
    Architecture (~1.9M params):
    - Encoder: 32 -> 64 -> 128 channels
    - Bottleneck: 256 channels with dilated convolutions
    - Decoder: 128 -> 64 channels with skip connections
    - Output: Sigmoid fire probability
    """
    def __init__(self, input_channels=13):
        super().__init__()
        
        # Initial convolution
        self.init_conv = ConvBNReLU(input_channels, 32)
        
        # Encoder
        self.enc1 = EncoderBlock(32, 64)    # 256 -> 128
        self.enc2 = EncoderBlock(64, 128)   # 128 -> 64
        
        # Bottleneck with dilated convolutions
        self.bottleneck_d2 = ConvBNReLU(128, 256, dilation=2)
        self.bottleneck_d4 = ConvBNReLU(256, 256, dilation=4)
        
        # Decoder with attention
        self.dec1 = DecoderBlock(256, 128, 128, use_attention=True)  # 64 -> 128
        self.dec2 = DecoderBlock(128, 64, 64, use_attention=False)    # 128 -> 256
        
        # Output
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        # Initial
        x = self.init_conv(x)
        
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        
        # Bottleneck
        x = self.bottleneck_d2(x)
        x = self.bottleneck_d4(x)
        
        # Decoder
        x = self.dec1(x, skip2)
        x = self.dec2(x, skip1)
        
        # Output
        x = torch.sigmoid(self.output_conv(x))
        
        return x

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class CombinedLoss(nn.Module):
    """Weighted BCE + Dice Loss"""
    def __init__(self, fire_weight=10.0, non_fire_weight=1.0):
        super().__init__()
        self.fire_weight = fire_weight
        self.non_fire_weight = non_fire_weight
    
    def forward(self, pred, target):
        # Weighted BCE
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        weights = target * self.fire_weight + (1 - target) * self.non_fire_weight
        weighted_bce = (bce * weights).mean()
        
        # Dice loss
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = 1 - (2 * intersection + smooth) / (union + smooth)
        
        return 0.5 * weighted_bce + 0.5 * dice

# =============================================================================
# METRICS
# =============================================================================

def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    return (intersection / (union + 1e-6)).item()

def compute_accuracy(pred, target, threshold=0.5):
    """Compute pixel accuracy."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    correct = (pred_binary == target_binary).sum()
    total = target.numel()
    
    return (correct / total).item()

# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0
    
    # Gradient accumulation: simulate larger batch size
    accumulation_steps = BATCH_SIZE // MICRO_BATCH_SIZE
    
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Scale loss by accumulation steps for correct gradient magnitude
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Unscale for logging
        total_iou += compute_iou(outputs, targets)
        total_acc += compute_accuracy(outputs, targets)
        num_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)} - "
                       f"Loss: {loss.item():.4f}, IoU: {compute_iou(outputs, targets):.4f}")
    
    return total_loss / num_batches, total_iou / num_batches, total_acc / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            total_iou += compute_iou(outputs, targets)
            total_acc += compute_accuracy(outputs, targets)
            num_batches += 1
    
    return total_loss / num_batches, total_iou / num_batches, total_acc / num_batches


def train_model():
    """Main training function."""
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 60)
    logger.info("PyroCast Fire Spread Model - PyTorch Training")
    logger.info("=" * 60)
    logger.info(f"Run name: {run_name}")
    
    # Get device
    device = get_device()
    
    # Load datasets
    train_path = os.path.join(TFRECORD_DIR, "spread_train.tfrecord")
    val_path = os.path.join(TFRECORD_DIR, "spread_val.tfrecord")
    
    if not os.path.exists(train_path):
        logger.error(f"Training data not found at {train_path}")
        logger.error("Run 04_build_dataset.py or 04b_augment_dataset.py first.")
        return None
    
    logger.info("Loading datasets...")
    train_dataset = FireSpreadDataset(train_path, single_step=True)
    val_dataset = FireSpreadDataset(val_path, single_step=True)
    
    # Use micro-batch size for DataLoader (gradient accumulation simulates full batch)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Build model
    logger.info("Building model...")
    model = FireSpreadModel(input_channels=INPUT_CHANNELS).to(device)
    
    # torch.compile is not compatible with DirectML
    if not DML_AVAILABLE:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compiled with torch.compile for faster execution")
        except Exception as e:
            logger.info(f"torch.compile not available: {e}")
    else:
        logger.info("Skipping torch.compile (not compatible with DirectML)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(fire_weight=FIRE_PIXEL_WEIGHT, non_fire_weight=NON_FIRE_PIXEL_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    
    # Schedulers - use cosine annealing with warm restarts + ReduceLROnPlateau backup
    if USE_COSINE_ANNEALING:
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=COSINE_T_MAX, T_mult=1, eta_min=MIN_LEARNING_RATE
        )
        logger.info(f"Using CosineAnnealingWarmRestarts (T_0={COSINE_T_MAX})")
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_DECAY_FACTOR, 
                                          patience=LR_DECAY_PATIENCE, min_lr=MIN_LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_iou = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    
    logger.info("Starting training...")
    logger.info(f"Epochs: {MAX_EPOCHS}, Effective batch size: {BATCH_SIZE} (micro-batch: {MICRO_BATCH_SIZE})")
    logger.info(f"  Gradient accumulation: {BATCH_SIZE // MICRO_BATCH_SIZE} steps")
    logger.info("=" * 60)
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_iou, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_iou, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        if USE_COSINE_ANNEALING:
            cosine_scheduler.step()  # Called every epoch
        plateau_scheduler.step(val_loss)  # Backup if cosine not working
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Log progress
        logger.info(f"Epoch {epoch + 1}/{MAX_EPOCHS} ({epoch_time:.1f}s) - "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}, LR: {current_lr:.2e}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iou = val_iou
            patience_counter = 0
            
            model_path = os.path.join(MODEL_DIR, f"spread_model_pytorch_{run_name}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
            }, model_path)
            logger.info(f"  → Saved best model (Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    final_path = os.path.join(MODEL_DIR, f"spread_model_pytorch_{run_name}_final.pt")
    torch.save(model.state_dict(), final_path)
    
    # Save history
    history_path = os.path.join(LOGS_DIR, f"history_pytorch_{run_name}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best Val IoU: {best_val_iou:.4f}")
    logger.info(f"Model saved to: {final_path}")
    logger.info("=" * 60)
    
    return model


if __name__ == "__main__":
    model = train_model()
