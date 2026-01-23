"""
PyroCast Fire Spread Model - Step 7: Fine-tune on Sim2Real-Fire Dataset
========================================================================
Fine-tunes the pre-trained model on physics-based simulation data to remove
suppression bias from historical fire data.

Sim2Real-Fire Dataset (NeurIPS 2024):
- 1M simulated wildfire scenarios (FARSITE, WFDS, WRF-SFIRE)
- Multi-modal: topography, vegetation, fuel, weather, satellite imagery
- Uncontrolled spread (no firefighting intervention)

Strategy:
- Load pre-trained weights from historical data training
- Freeze encoder (preserves learned spatial features)
- Fine-tune decoder with low learning rate
- Use simulation data to correct suppression bias
"""

import os
import sys
import json
import cv2
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

try:
    import tifffile as tif
except ImportError:
    print("Installing tifffile for GeoTIFF support...")
    os.system("pip install tifffile")
    import tifffile as tif

# DirectML for AMD GPU
DML_AVAILABLE = False
try:
    import torch_directml
    DML_AVAILABLE = True
    print("✓ torch_directml imported successfully")
except ImportError as e:
    print(f"✗ DirectML not available: {e}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Sim2Real-Fire Fine-tuning
# =============================================================================

# Paths to Sim2Real-Fire dataset (update after download)
SIM2REAL_ROOT = os.path.join(BASE_DIR, "data", "sim2real_fire")
SIM2REAL_SCENES = []  # Will be populated from available scene directories

# Fine-tuning hyperparameters
FINETUNE_LR = 1e-5  # 10x lower than initial training
FINETUNE_EPOCHS = 10  # Short fine-tuning phase
FINETUNE_BATCH_SIZE = 8  # Same micro-batch as main training
FREEZE_ENCODER = True  # Freeze encoder to preserve learned features
UNFREEZE_AFTER_EPOCH = 5  # Optionally unfreeze encoder after N epochs

# Data mapping from Sim2Real-Fire to PyroCast format
# Sim2Real uses: topography (3), vegetation (4), fuel (4), weather (10)
# PyroCast uses: 13 channels (fire_state + env features)

# =============================================================================
# DEVICE SETUP
# =============================================================================

def get_device():
    """Get the best available device."""
    if DML_AVAILABLE:
        try:
            os.environ['PYTORCH_DIRECTML_DISABLE_MEMORY_POOL'] = '1'
            device = torch_directml.device()
            logger.info(f"✓ Using DirectML (AMD GPU): {device}")
            return device
        except Exception as e:
            logger.error(f"DirectML failed: {e}")
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    return torch.device('cpu')

# =============================================================================
# SIM2REAL-FIRE DATASET ADAPTER
# =============================================================================

class Sim2RealFireDataset(Dataset):
    """
    Adapter for Sim2Real-Fire dataset to PyroCast format.
    
    Sim2Real-Fire structure per scene:
    - 1.1_Topography_Map/: DEM, slope, aspect (.tif)
    - 1.2_Vegetation_Map/: vegetation indices (.tif)
    - 1.3_Fuel_Map/: fuel types (.tif)
    - 1.4_Weather_Data/: weather parameters (.wxs files)
    - 1.5_Satellite_Images/: fire progression images (.jpg)
    - fire_xxx/: fire event directories with out1.jpg, out2.jpg, etc.
    """
    
    def __init__(self, scene_dirs, img_size=(256, 256), max_samples_per_scene=500):
        """
        Args:
            scene_dirs: List of paths to scene directories (e.g., 0001_02614)
            img_size: Output image size (H, W)
            max_samples_per_scene: Limit samples per scene to balance dataset
        """
        self.img_size = img_size
        self.samples = []
        
        logger.info(f"Loading Sim2Real-Fire dataset from {len(scene_dirs)} scenes...")
        
        for scene_dir in scene_dirs:
            if not os.path.exists(scene_dir):
                logger.warning(f"Scene directory not found: {scene_dir}")
                continue
            
            scene_samples = self._load_scene(scene_dir, max_samples_per_scene)
            self.samples.extend(scene_samples)
            logger.info(f"  {os.path.basename(scene_dir)}: {len(scene_samples)} samples")
        
        logger.info(f"Total samples loaded: {len(self.samples)}")
    
    def _load_scene(self, scene_dir, max_samples):
        """Load samples from a single scene directory."""
        samples = []
        
        # Load static environmental data (same for all fires in scene)
        topo_dir = os.path.join(scene_dir, "1.1_Topography_Map")
        vege_dir = os.path.join(scene_dir, "1.2_Vegetation_Map")
        fuel_dir = os.path.join(scene_dir, "1.3_Fuel_Map")
        
        try:
            topo_data = self._load_topo(topo_dir)
            vege_data = self._load_vege(vege_dir)
            fuel_data = self._load_fuel(fuel_dir)
        except Exception as e:
            logger.warning(f"Failed to load env data for {scene_dir}: {e}")
            return []
        
        # Find fire event directories
        fire_dirs = glob.glob(os.path.join(scene_dir, "fire_*"))
        if not fire_dirs:
            # Alternative: look for out*.jpg directly in scene
            fire_dirs = [scene_dir]
        
        sample_count = 0
        for fire_dir in fire_dirs:
            if sample_count >= max_samples:
                break
            
            # Get fire progression images (out1.jpg, out2.jpg, ...)
            fire_images = sorted(glob.glob(os.path.join(fire_dir, "out*.jpg")))
            
            if len(fire_images) < 2:
                continue
            
            # Create input-output pairs for consecutive timesteps
            for i in range(len(fire_images) - 1):
                if sample_count >= max_samples:
                    break
                
                samples.append({
                    'input_fire': fire_images[i],
                    'target_fire': fire_images[i + 1],
                    'topo': topo_data,
                    'vege': vege_data,
                    'fuel': fuel_data,
                    'scene': os.path.basename(scene_dir),
                    'timestep': i
                })
                sample_count += 1
        
        return samples
    
    def _load_topo(self, topo_dir):
        """Load topography data: DEM, slope, aspect."""
        if not os.path.exists(topo_dir):
            return np.zeros((3, *self.img_size), dtype=np.float32)
        
        topo_files = sorted(glob.glob(os.path.join(topo_dir, "*.tif")))
        topo_data = []
        
        for f in topo_files[:3]:  # DEM, slope, aspect
            try:
                img = tif.imread(f)
                img = cv2.resize(img.astype(np.float32), self.img_size, 
                               interpolation=cv2.INTER_NEAREST)
                topo_data.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                topo_data.append(np.zeros(self.img_size, dtype=np.float32))
        
        # Pad if fewer than 3 channels
        while len(topo_data) < 3:
            topo_data.append(np.zeros(self.img_size, dtype=np.float32))
        
        topo_array = np.stack(topo_data[:3], axis=0)
        
        # Normalize (simple z-score per channel)
        for i in range(topo_array.shape[0]):
            if topo_array[i].std() > 0:
                topo_array[i] = (topo_array[i] - topo_array[i].mean()) / topo_array[i].std()
        
        return topo_array
    
    def _load_vege(self, vege_dir):
        """Load vegetation data."""
        if not os.path.exists(vege_dir):
            return np.zeros((4, *self.img_size), dtype=np.float32)
        
        vege_files = sorted(glob.glob(os.path.join(vege_dir, "*.tif")))
        vege_data = []
        
        for f in vege_files[:4]:
            try:
                img = tif.imread(f)
                img = cv2.resize(img.astype(np.float32), self.img_size,
                               interpolation=cv2.INTER_NEAREST)
                vege_data.append(img)
            except Exception as e:
                vege_data.append(np.zeros(self.img_size, dtype=np.float32))
        
        while len(vege_data) < 4:
            vege_data.append(np.zeros(self.img_size, dtype=np.float32))
        
        vege_array = np.stack(vege_data[:4], axis=0)
        
        for i in range(vege_array.shape[0]):
            if vege_array[i].std() > 0:
                vege_array[i] = (vege_array[i] - vege_array[i].mean()) / vege_array[i].std()
        
        return vege_array
    
    def _load_fuel(self, fuel_dir):
        """Load fuel map data."""
        if not os.path.exists(fuel_dir):
            return np.zeros((4, *self.img_size), dtype=np.float32)
        
        fuel_files = sorted(glob.glob(os.path.join(fuel_dir, "*.tif")))
        fuel_data = []
        
        for f in fuel_files[:4]:
            try:
                img = tif.imread(f)
                img = cv2.resize(img.astype(np.float32), self.img_size,
                               interpolation=cv2.INTER_NEAREST)
                fuel_data.append(img)
            except Exception as e:
                fuel_data.append(np.zeros(self.img_size, dtype=np.float32))
        
        while len(fuel_data) < 4:
            fuel_data.append(np.zeros(self.img_size, dtype=np.float32))
        
        fuel_array = np.stack(fuel_data[:4], axis=0)
        
        for i in range(fuel_array.shape[0]):
            if fuel_array[i].std() > 0:
                fuel_array[i] = (fuel_array[i] - fuel_array[i].mean()) / fuel_array[i].std()
        
        return fuel_array
    
    def _load_fire_mask(self, img_path):
        """Load fire mask from JPG and binarize."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(self.img_size, dtype=np.float32)
        
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)
        # Binarize: any non-zero pixel is fire
        mask = (img > 0).astype(np.float32)
        return mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load fire masks
        input_fire = self._load_fire_mask(sample['input_fire'])
        target_fire = self._load_fire_mask(sample['target_fire'])
        
        # Get environmental data (already loaded)
        topo = sample['topo']  # (3, H, W)
        vege = sample['vege']  # (4, H, W)
        fuel = sample['fuel']  # (4, H, W)
        
        # Combine into PyroCast input format: (13, H, W)
        # Channel 0: current fire state
        # Channels 1-3: topography (elevation, slope, aspect)
        # Channels 4-7: vegetation (4 indices)
        # Channels 8-11: fuel (4 types)
        # Channel 12: padding/weather proxy
        
        input_tensor = np.concatenate([
            input_fire[np.newaxis, ...],  # (1, H, W)
            topo,                          # (3, H, W)
            vege,                          # (4, H, W)
            fuel,                          # (4, H, W)
            np.zeros((1, *self.img_size), dtype=np.float32)  # (1, H, W) padding
        ], axis=0)  # Total: 13 channels
        
        target_tensor = target_fire[np.newaxis, ...]  # (1, H, W)
        
        return (
            torch.from_numpy(input_tensor).float(),
            torch.from_numpy(target_tensor).float()
        )


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pretrained_model(checkpoint_path, device):
    """Load pre-trained PyroCast model from checkpoint."""
    from config import INPUT_CHANNELS
    
    # Import model architecture from training script
    # We need to define it here or import from 05_train_pytorch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Try to import model from training script
    try:
        from importlib import import_module
        train_module = import_module('05_train_pytorch')
        FireSpreadModel = train_module.FireSpreadModel
    except ImportError:
        logger.error("Could not import FireSpreadModel from 05_train_pytorch.py")
        logger.error("Make sure 05_train_pytorch.py is in the same directory")
        return None
    
    model = FireSpreadModel(input_channels=INPUT_CHANNELS)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        logger.info("✓ Checkpoint loaded successfully")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Starting with randomly initialized weights")
    
    model = model.to(device)
    return model


def freeze_encoder(model):
    """Freeze encoder layers to preserve learned features."""
    frozen_count = 0
    
    for name, param in model.named_parameters():
        # Freeze encoder blocks (enc1, enc2, enc3) and bottleneck
        if any(x in name for x in ['enc1', 'enc2', 'enc3', 'bottleneck']):
            param.requires_grad = False
            frozen_count += 1
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Encoder frozen: {frozen_count} parameter groups")
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return model


def unfreeze_encoder(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"All parameters unfrozen: {trainable:,} trainable")
    
    return model


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + Focal loss (same as Sim2Real-Fire paper)."""
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + \
               self.focal_weight * self.focal(pred, target)


# =============================================================================
# METRICS
# =============================================================================

def compute_iou(pred, target, threshold=0.5):
    """Compute IoU for fire pixels."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    if union < 1e-6:
        return 1.0 if target.sum() < 1e-6 else 0.0
    
    return (intersection / union).item()


# =============================================================================
# TRAINING LOOPS
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=8):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets) / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total_iou += compute_iou(outputs, targets)
        num_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)} - "
                       f"Loss: {loss.item() * accumulation_steps:.4f}, "
                       f"IoU: {compute_iou(outputs, targets):.4f}")
    
    return total_loss / num_batches, total_iou / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            total_iou += compute_iou(outputs, targets)
            num_batches += 1
    
    return total_loss / num_batches, total_iou / num_batches


# =============================================================================
# MAIN FINE-TUNING FUNCTION
# =============================================================================

def finetune_on_sim2real():
    """Main fine-tuning pipeline."""
    logger.info("=" * 60)
    logger.info("PyroCast Fire Spread Model - Sim2Real-Fire Fine-tuning")
    logger.info("=" * 60)
    
    device = get_device()
    
    # Check for Sim2Real-Fire data
    if not os.path.exists(SIM2REAL_ROOT):
        logger.error(f"Sim2Real-Fire data not found at: {SIM2REAL_ROOT}")
        logger.error("")
        logger.error("Please download the dataset:")
        logger.error("1. Mini version (testing): https://1drv.ms/f/s!AhX2uIQNmngrafE5KFjNyZym_7o")
        logger.error("2. Full dataset: https://github.com/TJU-IDVLab/Sim2Real-Fire")
        logger.error("")
        logger.error(f"Extract to: {SIM2REAL_ROOT}")
        return None
    
    # Find available scene directories
    scene_dirs = []
    for item in os.listdir(SIM2REAL_ROOT):
        item_path = os.path.join(SIM2REAL_ROOT, item)
        if os.path.isdir(item_path):
            # Check if it has expected structure
            if os.path.exists(os.path.join(item_path, "1.1_Topography_Map")) or \
               any(f.endswith('.tif') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                scene_dirs.append(item_path)
    
    if not scene_dirs:
        logger.error("No valid scene directories found in Sim2Real-Fire data")
        logger.error("Expected structure: {scene_id}/1.1_Topography_Map/, 1.2_Vegetation_Map/, etc.")
        return None
    
    logger.info(f"Found {len(scene_dirs)} scene directories")
    
    # Load dataset
    logger.info("Loading Sim2Real-Fire dataset...")
    
    # Split scenes for train/val
    np.random.seed(42)
    np.random.shuffle(scene_dirs)
    split_idx = int(0.9 * len(scene_dirs))
    train_scenes = scene_dirs[:split_idx]
    val_scenes = scene_dirs[split_idx:]
    
    train_dataset = Sim2RealFireDataset(train_scenes, img_size=(IMG_SIZE, IMG_SIZE))
    val_dataset = Sim2RealFireDataset(val_scenes, img_size=(IMG_SIZE, IMG_SIZE))
    
    if len(train_dataset) == 0:
        logger.error("No training samples loaded. Check data format.")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE, 
                             shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=FINETUNE_BATCH_SIZE,
                           shuffle=False, num_workers=0, pin_memory=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Load pre-trained model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    model = load_pretrained_model(checkpoint_path, device)
    
    if model is None:
        return None
    
    # Freeze encoder if configured
    if FREEZE_ENCODER:
        model = freeze_encoder(model)
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=FINETUNE_LR / 10)
    
    # Training loop
    best_val_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    
    logger.info("")
    logger.info("Starting fine-tuning...")
    logger.info(f"Epochs: {FINETUNE_EPOCHS}, LR: {FINETUNE_LR}")
    logger.info(f"Encoder frozen: {FREEZE_ENCODER}")
    logger.info("=" * 60)
    
    for epoch in range(FINETUNE_EPOCHS):
        epoch_start = time.time()
        
        # Optionally unfreeze encoder after N epochs
        if FREEZE_ENCODER and epoch == UNFREEZE_AFTER_EPOCH:
            logger.info("Unfreezing encoder for full fine-tuning...")
            model = unfreeze_encoder(model)
            # Reset optimizer to include all parameters
            optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR / 10)
        
        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device,
            accumulation_steps=BATCH_SIZE // FINETUNE_BATCH_SIZE
        )
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Log
        logger.info(f"Epoch {epoch + 1}/{FINETUNE_EPOCHS} ({epoch_time:.1f}s) - "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}, LR: {current_lr:.2e}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_path = os.path.join(CHECKPOINT_DIR, "best_model_finetuned.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  → Saved best fine-tuned model (Val IoU: {val_iou:.4f})")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
    
    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "model_finetuned_final.pt")
    torch.save(model.state_dict(), final_path)
    
    # Save training history
    history_path = os.path.join(LOGS_DIR, f"finetune_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Fine-tuning Complete!")
    logger.info("=" * 60)
    logger.info(f"Best Val IoU: {best_val_iou:.4f}")
    logger.info(f"Model saved: {final_path}")
    logger.info(f"History saved: {history_path}")
    
    return model


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    finetune_on_sim2real()
