import os
# GPU via Keras 3 on the PyTorch/CUDA backend (run in pyrocast_cuda env).
os.environ["KERAS_BACKEND"] = "torch"

import tensorflow as tf
import torch
import keras
from keras import layers, models, Input, optimizers, callbacks
import glob

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

assert torch.cuda.is_available(), "CUDA not available to torch"
print(f"GPU: {torch.cuda.get_device_name(0)} | keras backend: {keras.backend.backend()}")

import pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent
TRAIN_GLOB = str(_REPO / "Training Data Florida" / "Florida_Spatial_Train_v4.tfrecord.gz")
VAL_GLOB = str(_REPO / "Training Data Florida" / "Florida_Spatial_Val_v4.tfrecord.gz")

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-4
LABEL_SMOOTHING = 0.05
TARGET_DIM = 256
CHANNELS = 26
MODEL_OUTPUT = 'best_robust_fire_model_v4.keras'

BAND_NAMES = [
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
    'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip', 'ERC', 'FM100', 'PDSI',
    'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open',
    'Precip_30d', 'Precip_90d', 'ERC_30d', 'FM100_30d', 'VPD', 'PDSI_90dago', 'NDVI_90d',
]


def parse_tfrecord_fn(example_proto):
    fd = {name: tf.io.VarLenFeature(tf.float32) for name in BAND_NAMES}
    fd['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    parsed = tf.io.parse_single_example(example_proto, fd)
    size_257, size_256 = 257 * 257, 256 * 256
    bands = []
    for name in BAND_NAMES:
        x = tf.sparse.to_dense(parsed[name], default_value=0.0)
        n = tf.shape(x)[0]
        x = tf.cond(tf.equal(n, size_257),
                    lambda: tf.reshape(x, [257, 257])[:TARGET_DIM, :TARGET_DIM],
                    lambda: tf.cond(tf.equal(n, size_256),
                                    lambda: tf.reshape(x, [256, 256]),
                                    lambda: tf.zeros([TARGET_DIM, TARGET_DIM], tf.float32)))
        bands.append(x)
    img = tf.ensure_shape(tf.stack(bands, axis=-1), [TARGET_DIM, TARGET_DIM, len(BAND_NAMES)])
    return img, parsed['label']


def augment(image, label):
    image = tf.image.rot90(image, tf.random.uniform([], 0, 4, tf.int32))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = image + tf.random.normal(tf.shape(image), 0.0, 0.02)
    return tf.clip_by_value(image, 0.0, 1.0), label


def count(path):
    return sum(1 for _ in tf.data.TFRecordDataset(glob.glob(path), compression_type='GZIP'))


def get_ds(path, training):
    files = glob.glob(path)
    if not files:
        raise SystemExit(f"No files at {path}")
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=42).map(augment, num_parallel_calls=tf.data.AUTOTUNE).repeat()
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def se_block(x, ratio=16):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(c // ratio, activation='relu', use_bias=False)(s)
    s = layers.Dense(c, activation='sigmoid', use_bias=False)(s)
    return layers.Multiply()([x, layers.Reshape((1, 1, c))(s)])


def res_block(x, f, stride=1):
    sc = x
    x = layers.Conv2D(f, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = se_block(x)
    if stride != 1 or sc.shape[-1] != f:
        sc = layers.Conv2D(f, 1, strides=stride, use_bias=False)(sc)
        sc = layers.BatchNormalization()(sc)
    return layers.ReLU()(layers.Add()([x, sc]))


def build_model(shape):
    inp = Input(shape=shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    for f, s in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
        x = res_block(x, f, s)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="Fire_Risk_v4")


if __name__ == "__main__":
    print("Counting samples...")
    n_tr, n_va = count(TRAIN_GLOB), count(VAL_GLOB)
    print(f"train={n_tr} val={n_va}")
    train_ds = get_ds(TRAIN_GLOB, True)
    val_ds = get_ds(VAL_GLOB, False)
    spe = n_tr // BATCH_SIZE
    vs = max(1, n_va // BATCH_SIZE)

    model = build_model((TARGET_DIM, TARGET_DIM, CHANNELS))
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Recall(name='recall'), keras.metrics.Precision(name='precision')])

    cbs = [
        callbacks.ModelCheckpoint(MODEL_OUTPUT, save_best_only=True, monitor='val_auc', mode='max', verbose=1),
        callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4, mode='max', verbose=1),
    ]
    model.fit(train_ds, steps_per_epoch=spe, epochs=EPOCHS, validation_data=val_ds,
              validation_steps=vs, callbacks=cbs, class_weight={0: 1.0, 1: 1.0}, verbose=2)
    print(f"Done. Saved {MODEL_OUTPUT}")
