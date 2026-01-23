import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers, callbacks
import glob
import os
import sys

# ================= CONFIGURATION =================
DATA_DIR = "C:\\Users\\nonna\\Downloads\\PyroCast\\Training Data\\"   # Path to your downloaded .tfrecord files
BATCH_SIZE = 64       # Optimized for CPU/Memory safety
EPOCHS = 50           
LEARNING_RATE = 1e-4

# Dimensions
RAW_IMG_SIZE = 257    # Size exported from GEE
TARGET_IMG_SIZE = 256 # Size required by the AI
CHANNELS = 15         # Number of bands

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
    Data Augmentation:
    Rotates and flips images. Since fire physics are the same regardless of 
    map orientation, this multiplies your effective dataset size by 8x.
    """
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label

def get_dataset(file_pattern, is_training=True):
    files = glob.glob(file_pattern)
    if not files:
        raise ValueError(f"No files found matching {file_pattern} in {os.getcwd()}")
    
    # compression_type=None (Fixes 'incorrect header check' error)
    dataset = tf.data.TFRecordDataset(files, compression_type=None, num_parallel_reads=tf.data.AUTOTUNE)
    
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
    
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
    
    return models.Model(inputs, outputs, name="Fire_Prediction_AI")

# ================= EXECUTION =================

if __name__ == "__main__":
    # 1. Load Data
    try:
        train_ds = get_dataset(os.path.join(DATA_DIR, "*.tfrecord"), is_training=True)
        
        # Validation Step: Ensure we can read a batch without crashing
        print("Validating data stream...")
        for img, lbl in train_ds.take(1):
            print(f"Success! Input Shape: {img.shape}, Label Shape: {lbl.shape}")
            break
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit()

    # 2. Build & Compile Model
    model = build_model((TARGET_IMG_SIZE, TARGET_IMG_SIZE, CHANNELS))
    
    # Learning Rate Schedule (Cosine Decay for smooth convergence)
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * 100
    )
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # 3. Train
    print("\nStarting Training...")
    
    # Class Weights: Weight 1.2 on fires ensures we don't miss ignitions
    class_weight = {0: 1.0, 1: 1.2} 

    callbacks_list = [
        callbacks.ModelCheckpoint('best_fire_model.keras', save_best_only=True, monitor='loss'),
        callbacks.EarlyStopping(monitor='loss', patience=8)
    ]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        class_weight=class_weight
    )
    
    print("Training Complete. Model saved as 'best_fire_model.keras'")