import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers, callbacks
import glob
import os
import sys

# ================= CONFIGURATION =================
# FLORIDA-SPECIFIC DATA DIRECTORY
DATA_DIR = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data Florida\\"
BATCH_SIZE = 64       # Optimized for CPU/Memory safety
EPOCHS = 100           
LEARNING_RATE = 1e-4

# Dimensions
RAW_IMG_SIZE = 257    # Size exported from GEE
TARGET_IMG_SIZE = 256 # Size required by the AI
CHANNELS = 15         # Number of bands

# Output model name (Florida-specific)
MODEL_OUTPUT = 'best_fire_model_florida.keras'

# ================= DATA PIPELINE =================

def parse_tfrecord_fn(example_proto):
    band_names = [
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
        'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
        'Elevation', 'Slope', 'Pop_Density'
    ]
    
    # 1. Define Features with Robust Defaults
    # This creates a "Safety Net": if a band is missing in the file, 
    # it fills the data with Zeros instead of crashing.
    flat_shape = [RAW_IMG_SIZE * RAW_IMG_SIZE]
    zero_default = [0.0] * (RAW_IMG_SIZE * RAW_IMG_SIZE)
    
    feature_description = {
        name: tf.io.FixedLenFeature(flat_shape, tf.float32, default_value=zero_default) 
        for name in band_names
    }
    
    # Label: Read as Float32 to prevent "Type Mismatch" errors
    feature_description['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)

    # 2. Parse
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # 3. Reshape Flat Arrays to Images
    band_tensors = [tf.reshape(parsed[name], [RAW_IMG_SIZE, RAW_IMG_SIZE]) for name in band_names]
    image = tf.stack(band_tensors, axis=-1)
    
    # 4. Crop to Exact Size (Fixes 257 vs 256 error)
    image = tf.image.resize_with_crop_or_pad(image, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    image = tf.ensure_shape(image, [TARGET_IMG_SIZE, TARGET_IMG_SIZE, CHANNELS])
    
    label = parsed['label']
    return image, label

def augment(image, label):
    """
    AGGRESSIVE Data Augmentation:
    Multiplies effective dataset size significantly to combat overfitting.
    """
    # Geometric augmentation (8x multiplier)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Random brightness/contrast (spectral invariance)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Random crop and resize (spatial invariance)
    if tf.random.uniform(shape=[]) > 0.5:
        crop_size = tf.random.uniform(shape=[], minval=240, maxval=256, dtype=tf.int32)
        image = tf.image.random_crop(image, [crop_size, crop_size, CHANNELS])
        image = tf.image.resize(image, [TARGET_IMG_SIZE, TARGET_IMG_SIZE])
    
    # Gaussian noise (sensor noise simulation)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
    image = image + noise
    
    # Clamp values to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

def get_dataset(file_pattern, is_training=True, validation_split=0.15):
    files = glob.glob(file_pattern)
    if not files:
        raise ValueError(f"No files found matching {file_pattern} in {os.getcwd()}")
    
    print(f"Found {len(files)} TFRecord file(s) for Florida training data")
    
    # compression_type=None (Fixes 'incorrect header check' error)
    dataset = tf.data.TFRecordDataset(files, compression_type=None, num_parallel_reads=tf.data.AUTOTUNE)
    
    # Count total samples
    total_count = sum(1 for _ in dataset)
    print(f"Total samples in dataset: {total_count}")
    
    # Reload dataset after counting and SHUFFLE ENTIRE DATASET FIRST
    dataset = tf.data.TFRecordDataset(files, compression_type=None, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=total_count, seed=42)  # Shuffle ALL data with fixed seed
    
    if is_training == 'train':
        # Training set: skip validation samples (which are now shuffled)
        val_size = int(total_count * validation_split)
        train_size = total_count - val_size
        dataset = dataset.skip(val_size)
        print(f"Training set: {train_size} samples")
        
        # Parse and augment
        dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=2000)  # Shuffle again for batching
        dataset = dataset.repeat()  # Repeat for multiple epochs
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, train_size
        
    elif is_training == 'val':
        # Validation set: take first validation_split samples (already shuffled)
        val_size = int(total_count * validation_split)
        dataset = dataset.take(val_size)
        print(f"Validation set: {val_size} samples")
        
        # Parse only (no augmentation for validation)
        dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, val_size
    else:
        # Legacy support (not used)
        dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

# ================= MODEL: SE-ResNet-18 =================
# State-of-the-Art architecture for multispectral data

def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation: Teaches AI to pay attention to specific bands (e.g. Moisture)"""
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([input_tensor, se])

def res_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x) # Apply Attention
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Initial Convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # ResNet Stages
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = res_block(x, 128, stride=2)
    x = res_block(x, 128)
    x = res_block(x, 256, stride=2)
    x = res_block(x, 256)
    x = res_block(x, 512, stride=2)
    x = res_block(x, 512)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs, outputs, name="Fire_Prediction_AI_Florida")

# ================= EXECUTION =================

if __name__ == "__main__":
    print("="*60)
    print("FLORIDA FIRE RISK MODEL TRAINING (WITH VALIDATION)")
    print("="*60)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Output:   {MODEL_OUTPUT}")
    print(f"Augmentation:   AGGRESSIVE (geometric + spectral + spatial + noise)")
    print("="*60)
    
    # 1. Load Data with Train/Val Split
    try:
        print("\nLoading datasets...")
        train_ds, train_size = get_dataset(os.path.join(DATA_DIR, "*.tfrecord"), is_training='train', validation_split=0.15)
        val_ds, val_size = get_dataset(os.path.join(DATA_DIR, "*.tfrecord"), is_training='val', validation_split=0.15)
        
        # Calculate steps per epoch
        steps_per_epoch = train_size // BATCH_SIZE
        validation_steps = val_size // BATCH_SIZE
        
        print(f"\nSteps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Validation Step: Ensure we can read a batch without crashing
        print("\nValidating data stream...")
        for img, lbl in train_ds.take(1):
            print(f"Success! Input Shape: {img.shape}, Label Shape: {lbl.shape}")
            break
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print("\nMake sure you have downloaded Florida TFRecord files to:")
        print(f"  {DATA_DIR}")
        exit()

    # 2. Build & Compile Model
    model = build_model((TARGET_IMG_SIZE, TARGET_IMG_SIZE, CHANNELS))
    
    # Use plain learning rate (not schedule) so ReduceLROnPlateau can work
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')
        ]
    )

    # 3. Train with Validation
    print("\n" + "="*60)
    print("Starting Training (Florida Dataset with Validation)")
    print("="*60)
    
    # Class Weights: Weight 1.2 on fires ensures we don't miss ignitions
    class_weight = {0: 1.0, 1: 1.2} 

    callbacks_list = [
        callbacks.ModelCheckpoint(
            MODEL_OUTPUT, 
            save_best_only=True, 
            monitor='val_loss',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        class_weight=class_weight,
        verbose=1
    )
    
    print("\n" + "="*60)
    print(f"Training Complete. Model saved as '{MODEL_OUTPUT}'")
    print("="*60)
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    print(f"  Loss:      {history.history['loss'][-1]:.4f}")
    print(f"  Accuracy:  {history.history['accuracy'][-1]:.4f}")
    print(f"  AUC:       {history.history['auc'][-1]:.4f}")
    
    print("\nFinal Validation Metrics:")
    print(f"  Loss:      {history.history['val_loss'][-1]:.4f}")
    print(f"  Accuracy:  {history.history['val_accuracy'][-1]:.4f}")
    print(f"  AUC:       {history.history['val_auc'][-1]:.4f}")
    
    # Check for overfitting
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    gap = train_acc - val_acc
    
    print(f"\nOverfitting Analysis:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy:   {val_acc:.4f}")
    print(f"  Gap:            {gap:.4f} ({gap*100:.1f}%)")
    
    if gap > 0.15:
        print("  ⚠️  WARNING: Significant overfitting detected!")
    elif gap > 0.10:
        print("  ⚠️  CAUTION: Moderate overfitting.")
    else:
        print("  ✓  Good generalization!")
    
    print("="*60)
