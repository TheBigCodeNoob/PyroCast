"""
Rich tabular feature extraction from the v4 patches (26 channels, temporal trends).

Reads all downloaded v4 shards from 'Training Data Florida/v4/' (handles .gz),
computes per-channel full-patch + central-region stats, plus a per-sample
location-group id (from static channels) for leakage-free GroupKFold. Single
pooled output; the modeling script does the location-grouped train/val split.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import glob
import numpy as np
import tensorflow as tf

BANDS = [
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
    'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip', 'ERC', 'FM100', 'PDSI',
    'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open',
    'Precip_30d', 'Precip_90d', 'ERC_30d', 'FM100_30d', 'VPD', 'PDSI_90dago', 'NDVI_90d',
]
DATA_DIR = 'Training Data Florida/v4'
OUT = 'v4_features.npz'
STATIC_IDX = [15, 16, 17, 18]  # Elevation, LC_Forest, LC_Wetland, LC_Open
FEAT_SUFFIXES = ['mean', 'std', 'p10', 'p50', 'p90', 'max', 'c32mean', 'c32std', 'c64mean', 'c64std']


def feature_names():
    return [f'{b}_{s}' for b in BANDS for s in FEAT_SUFFIXES]


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
    img = np.nan_to_num(np.stack(bands, axis=-1).astype(np.float32), nan=0.0)
    return img, int(p['label'].numpy())


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
    return '|'.join(str(round(float(img[:, :, idx].mean()), 3)) for idx in STATIC_IDX)


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '**', '*.tfrecord*'), recursive=True))
    if not files:
        raise SystemExit(f'No v4 shards found in {DATA_DIR}')
    print(f'Found {len(files)} shard(s).', flush=True)
    X, y, groups = [], [], []
    n = 0
    for f in files:
        comp = 'GZIP' if f.endswith('.gz') else None
        ds = tf.data.TFRecordDataset(f, compression_type=comp)
        for rec in ds:
            img, lbl = parse(rec.numpy())
            X.append(feats(img))
            y.append(lbl)
            groups.append(location_group(img))
            n += 1
            if n % 500 == 0:
                print(f'  parsed {n}', flush=True)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups)
    print(f'Total samples: {n}, fire={int(y.sum())}, unique locations={len(set(groups))}', flush=True)
    np.savez_compressed(OUT, X=X, y=y, groups=groups, feature_names=np.array(feature_names()))
    print(f'Saved {OUT}: {X.shape}', flush=True)


if __name__ == '__main__':
    main()
