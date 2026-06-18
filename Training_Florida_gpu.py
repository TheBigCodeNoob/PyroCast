import os
# ================= GPU PATH (Keras 3 on the PyTorch/CUDA backend) =================
# TensorFlow has no native-Windows GPU support, so we run the *model* on the
# PyTorch backend (CUDA) via Keras 3's multi-backend support, and keep tf.data
# (CPU) only for the input pipeline. The RTX 5070 (Blackwell, sm_120) needs a
# torch wheel built for cu128. This is a sibling of Training_Florida.py, which
# stays as the CPU fallback (do NOT delete it).
os.environ["KERAS_BACKEND"] = "torch"
# NOTE: intentionally NOT setting CUDA_VISIBLE_DEVICES=-1 here — we want the GPU.

import tensorflow as tf            # input pipeline only (forced to CPU below)
import torch                       # backend compute (GPU)
import keras
from keras import layers, models, Input, optimizers, callbacks
import glob

# Keep TensorFlow off the GPU (it can't use it on Windows anyway); torch owns it.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# ================= DEVICE REPORT (fail fast if no GPU) =================
_gpu_ok = torch.cuda.is_available()
print("=" * 60)
print(f"Keras backend : {keras.backend.backend()}")
print(f"torch         : {torch.__version__} | CUDA available: {_gpu_ok}")
if _gpu_ok:
    print(f"GPU           : {torch.cuda.get_device_name(0)} "
          f"| capability sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
print("=" * 60)
if not _gpu_ok:
    raise SystemExit(
        "CUDA GPU not visible to PyTorch. Aborting rather than silently running on CPU.\n"
        "Fix the torch install (needs a cu128 Blackwell build) and retry, or use "
        "Training_Florida.py for the CPU path."
    )

# ================= CONFIGURATION =================
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parent
TRAIN_DATA_PATH = str(_REPO_ROOT / "Training Data Florida" / "Florida_Spatial_Train_v3*.tfrecord")
VAL_DATA_PATH   = str(_REPO_ROOT / "Training Data Florida" / "Florida_Spatial_Val_v3*.tfrecord")

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
LABEL_SMOOTHING = 0.05

# Dimensions
TARGET_DIM = 256       # Input size for the model
# v3: 20 channels (v2 was 19). Added PDSI (Palmer Drought Severity Index)
# between FM100 and Elevation.
CHANNELS = 20

MODEL_OUTPUT = 'best_robust_fire_model_v3.keras'

# Sample counts come from Prepare_Florida_Data.py (see logs/prepare_v3.log):
# 9660 train / 1000 val. Hardcoded so we don't read the 48GB train file just to
# count before training. Update these if the data is regenerated.
TRAIN_SIZE = 9660
VAL_SIZE = 1000

BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
              'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip', 'ERC', 'FM100', 'PDSI',
              'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open', 'Pop_Density']

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
        # 1. If 257x257 -> Reshape & Crop to 256
        # 2. If 256x256 -> Reshape (Keep as is)
        # 3. Else       -> Return Zeros (Broken data)
        x = tf.cond(
            tf.equal(num_elements, size_257),
            true_fn=lambda: tf.reshape(x, [257, 257])[:TARGET_DIM, :TARGET_DIM],
            false_fn=lambda: tf.cond(
                tf.equal(num_elements, size_256),
                true_fn=lambda: tf.reshape(x, [256, 256]),
                false_fn=lambda: tf.zeros([TARGET_DIM, TARGET_DIM], dtype=tf.float32)
            )
        )

        band_tensors.append(x)

    image = tf.stack(band_tensors, axis=-1)
    image = tf.ensure_shape(image, [TARGET_DIM, TARGET_DIM, len(BAND_NAMES)])
    label = parsed['label']
    return image, label

def augment_safe(image, label):
    """Geometric flips/rotations + spectral noise (runs on CPU in tf.data)."""
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def get_dataset(file_path, is_training=True):
    files = glob.glob(file_path)
    if not files:
        raise ValueError(f"CRITICAL: No TFRecords found at {file_path}")
    print(f"Loading data from: {len(files)} files -> {files}")

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    dataset = tf.data.TFRecordDataset(files, compression_type=None, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.with_options(options)
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

    return dataset

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
    print("=" * 60)
    print("ROBUST FIRE MODEL TRAINING — GPU (Keras 3 / PyTorch CUDA)")
    print("=" * 60)

    print("\nLoading Training Data...")
    train_ds = get_dataset(TRAIN_DATA_PATH, is_training=True)
    print("\nLoading Validation Data...")
    val_ds = get_dataset(VAL_DATA_PATH, is_training=False)

    steps_per_epoch = TRAIN_SIZE // BATCH_SIZE
    validation_steps = max(1, VAL_SIZE // BATCH_SIZE)
    print(f"\nTrain samples: {TRAIN_SIZE} | Val samples: {VAL_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch} | Validation steps: {validation_steps}")

    model = build_model((TARGET_DIM, TARGET_DIM, CHANNELS))

    loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision')
        ]
    )

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    class_weight = {0: 1.0, 1: 1.2}

    callbacks_list = [
        callbacks.ModelCheckpoint(
            MODEL_OUTPUT,
            save_best_only=True,
            monitor='val_auc',
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

    print("\n" + "=" * 60)
    print(f"Training Complete. Model saved as '{MODEL_OUTPUT}'")
    print("=" * 60)
