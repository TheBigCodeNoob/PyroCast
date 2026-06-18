"""
Rich tabular feature extraction from the v3 patches.

Motivation: the v3 audit's logistic baseline (AUC 0.726) used only whole-patch
MEANS, which average fire-site conditions across a ~5km x 5km patch. The fire is
at the centroid, so this dilutes the signal. Here we extract, per channel:
full-patch stats (mean/std/percentiles/max) AND central-region stats (32x32 and
64x64 around the centroid). A gradient-boosted model on these should exceed the
mean-only baseline if local conditions carry signal.

Streams the TFRecords once (no 48GB-in-RAM), caches features to v3_features.npz.
Also stores a per-sample location-group id (from static channels) so we can do
leakage-free GroupKFold CV on the train set.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # pure numpy reductions; no GPU needed
import numpy as np
import tensorflow as tf

BANDS = [
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
    'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip', 'ERC', 'FM100', 'PDSI',
    'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open', 'Pop_Density'
]
TRAIN = 'Training Data Florida/Florida_Spatial_Train_v3.tfrecord'
VAL   = 'Training Data Florida/Florida_Spatial_Val_v3.tfrecord'
OUT   = 'v3_features.npz'

# Static channels used to identify a physical location (for grouped CV).
STATIC_IDX = [15, 16, 17, 18]  # Elevation, LC_Forest, LC_Wetland, LC_Open


def parse(record_bytes):
    fd = {b: tf.io.VarLenFeature(tf.float32) for b in BANDS}
    fd['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    p = tf.io.parse_single_example(record_bytes, fd)
    bands = []
    for b in BANDS:
        x = tf.sparse.to_dense(p[b], default_value=0.0).numpy()
        if len(x) == 257 * 257:
            arr = x.reshape(257, 257)[:256, :256]
        elif len(x) == 256 * 256:
            arr = x.reshape(256, 256)
        else:
            arr = np.zeros((256, 256), dtype=np.float32)
        bands.append(arr)
    img = np.stack(bands, axis=-1).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0)
    return img, int(p['label'].numpy())


FEAT_SUFFIXES = ['mean', 'std', 'p10', 'p50', 'p90', 'max', 'c32mean', 'c32std', 'c64mean', 'c64std']


def feature_names():
    names = []
    for b in BANDS:
        for s in FEAT_SUFFIXES:
            names.append(f'{b}_{s}')
    return names


def feats(img):
    c32 = img[112:144, 112:144, :]
    c64 = img[96:160, 96:160, :]
    out = np.empty(len(BANDS) * len(FEAT_SUFFIXES), dtype=np.float32)
    k = 0
    for i in range(len(BANDS)):
        ch = img[:, :, i]
        q = np.percentile(ch, [10, 50, 90])
        a = c32[:, :, i]
        b = c64[:, :, i]
        out[k:k + 10] = (ch.mean(), ch.std(), q[0], q[1], q[2], ch.max(),
                         a.mean(), a.std(), b.mean(), b.std())
        k += 10
    return out


def location_group(img):
    vals = []
    for idx in STATIC_IDX:
        vals.append(round(float(img[:, :, idx].mean()), 3))
    return '|'.join(str(v) for v in vals)


def extract(path):
    X, y, groups = [], [], []
    ds = tf.data.TFRecordDataset(path)
    for i, rec in enumerate(ds):
        img, lbl = parse(rec.numpy())
        X.append(feats(img))
        y.append(lbl)
        groups.append(location_group(img))
        if (i + 1) % 500 == 0:
            print(f'  {os.path.basename(path)}: {i + 1}', flush=True)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(groups)


if __name__ == '__main__':
    print('Extracting TRAIN features...', flush=True)
    Xtr, ytr, gtr = extract(TRAIN)
    print(f'  train: {Xtr.shape}, fire={int(ytr.sum())}, groups={len(set(gtr))}', flush=True)

    print('Extracting VAL features...', flush=True)
    Xva, yva, gva = extract(VAL)
    print(f'  val: {Xva.shape}, fire={int(yva.sum())}, groups={len(set(gva))}', flush=True)

    np.savez_compressed(
        OUT,
        X_train=Xtr, y_train=ytr, groups_train=gtr,
        X_val=Xva, y_val=yva, groups_val=gva,
        feature_names=np.array(feature_names()),
    )
    print(f'Saved {OUT}: train {Xtr.shape}, val {Xva.shape}', flush=True)
