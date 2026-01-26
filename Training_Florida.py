import os
# ================= CRITICAL =================
# Forces CPU execution to avoid RTX 5080 crash.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# ============================================

import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers, callbacks
import glob

# Enable XLA for speed
tf.config.optimizer.set_jit(True)

# ================= CONFIGURATION =================
TRAIN_DATA_PATH = "//workspace//PyroCast//Training Data Florida//Florida_Spatial_Trai*.tfrecord"
VAL_DATA_PATH   = "//workspace//PyroCast//Training Data Florida//Florida_Spatial_Va*.tfrecord"

BATCH_SIZE = 64         
EPOCHS = 30             
LEARNING_RATE = 1e-4
LABEL_SMOOTHING = 0.05  

# Dimensions
TARGET_DIM = 256       # Input size for the model
CHANNELS = 15         

MODEL_OUTPUT = 'best_robust_fire_model.keras'

BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
              'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
              'Elevation', 'Slope', 'Pop_Density']

# ================= UNIVERSAL PARSER =================

def parse_tfrecord_fn(example_proto):
    """
    UNIVERSAL PARSER:
    Handles both Raw GEE export (257x257) AND Pre-processed (256x256) data.
    Prevents the 'Black Square' validation bug.
    """
    feature_desc = {
        name: tf.io.VarLenFeature(tf.float32) for name in BAND_NAMES
    }
    feature_desc['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)

    parsed = tf.io.parse_single_example(example_proto, feature_desc)
    
    band_tensors = []
    
    # Pre-calculate sizes for the graph
    size_257 = 257 * 257
    size_256 = 256 * 256
    
    for name in BAND_NAMES:
        x = tf.sparse.to_dense(parsed[name], default_value=0.0)
        num_elements = tf.shape(x)[0]
        
        # --- ADAPTIVE SHAPE LOGIC ---
        # Logic:
        # 1. If 257x257 -> Reshape & Crop to 256
        # 2. If 256x256 -> Reshape (Keep as is)
        # 3. Else       -> Return Zeros (Broken data)
        
        x = tf.cond(
            tf.equal(num_elements, size_257),
            # Case A: Raw GEE Data (257) - Crop it
            true_fn=lambda: tf.reshape(x, [257, 257])[:TARGET_DIM, :TARGET_DIM],
            # Case B: Check if it's 256
            false_fn=lambda: tf.cond(
                tf.equal(num_elements, size_256),
                true_fn=lambda: tf.reshape(x, [256, 256]),
                # Case C: Invalid
                false_fn=lambda: tf.zeros([TARGET_DIM, TARGET_DIM], dtype=tf.float32)
            )
        )
        
        band_tensors.append(x)
    
    image = tf.stack(band_tensors, axis=-1)
    image = tf.ensure_shape(image, [TARGET_DIM, TARGET_DIM, len(BAND_NAMES)])
    label = parsed['label']
    return image, label

def augment_safe(image, label):
    """
    Robust Augmentation:
    1. Geometric Flips/Rotations.
    2. Spectral Noise.
    REMOVED: 'Input Channel Dropout' (It corrupts Batch Normalization stats).
    """
    # 1. Geometric
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # 2. Spectral Noise (Safe Regularization)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

def get_dataset(file_path, is_training=True):
    files = glob.glob(file_path)
    if not files:
        # Fallback search if exact name differs
        print(f"Warning: Exact match not found for {file_path}. Trying generic search...")
        files = glob.glob(file_path.replace("Florida_Spatial_Trai", "*").replace("Florida_Spatial_Va", "*"))
        if not files:
             raise ValueError(f"CRITICAL: No TFRecords found at {file_path}")
    
    print(f"Loading data from: {len(files)} files.")
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    # Quick count
    dataset_for_count = tf.data.TFRecordDataset(files, compression_type=None)
    if is_training:
        print("  Counting training records...")
        total_count = sum(1 for _ in dataset_for_count)
        print(f"  Total samples: {total_count}")
    else:
        # Estimate validation size for speed
        total_count = sum(1 for _ in dataset_for_count)
        print(f"  Validation samples: {total_count}")

    dataset = tf.data.TFRecordDataset(files, compression_type=None, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.with_options(options)
    
    # Apply Universal Parser
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1024, seed=42)
        dataset = dataset.map(augment_safe, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
    return dataset, total_count

# ================= MODEL: SE-ResNet-18 =================

def se_block(input_tensor, ratio=16):
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
    x = se_block(x) 
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = res_block(x, 128, stride=2)
    x = res_block(x, 128)
    x = res_block(x, 256, stride=2)
    x = res_block(x, 256)
    x = res_block(x, 512, stride=2)
    x = res_block(x, 512)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs, outputs, name="Fire_Risk_Robust_AI")

# ================= EXECUTION =================

if __name__ == "__main__":
    print("="*60)
    print("ROBUST FIRE MODEL TRAINING (Universal Parser Edition)")
    print("="*60)
    
    try:
        print("\nLoading Training Data...")
        train_ds, train_size = get_dataset(TRAIN_DATA_PATH, is_training=True)
        
        print("\nLoading Validation Data...")
        val_ds, val_size = get_dataset(VAL_DATA_PATH, is_training=False)
        
        steps_per_epoch = train_size // BATCH_SIZE
        validation_steps = val_size // BATCH_SIZE
        # Safety for small datasets
        if validation_steps == 0: validation_steps = 1
        
        print(f"\nSteps per epoch: {steps_per_epoch}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit()

    model = build_model((TARGET_DIM, TARGET_DIM, CHANNELS))
    
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')
        ]
    )

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    class_weight = {0: 1.0, 1: 1.2} 

    callbacks_list = [
        callbacks.ModelCheckpoint(
            MODEL_OUTPUT, 
            save_best_only=True, 
            monitor='val_auc', # Switched to AUC (safer for imbalanced)
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_auc', 
            patience=8,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=3,
            mode='max',
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