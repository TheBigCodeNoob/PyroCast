"""
Follow-up audit. Focused on the missing pieces from audit_v2.py:
  A. Full FP/FN per-band feature comparison
  B. Per-location deduplicated metrics (treat each unique fingerprint as one decision)
  C. Logistic-regression-on-mean-features baseline
  D. Temp_Max distribution check (season proxy)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import math
import numpy as np
import tensorflow as tf
import keras
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VAL_FILE = 'Training Data Florida/Florida_Spatial_Val_v2.tfrecord'
TRAIN_FILE = 'Training Data Florida/Florida_Spatial_Train_v2.tfrecord'
MODEL_FILE = 'best_robust_fire_model_v2.keras'

ALL_BANDS = [
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
    'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip', 'ERC', 'FM100',
    'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open', 'Pop_Density'
]


def parse_sample(record_bytes):
    feature_desc = {b: tf.io.VarLenFeature(tf.float32) for b in ALL_BANDS}
    feature_desc['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    parsed = tf.io.parse_single_example(record_bytes, feature_desc)
    bands = []
    for b in ALL_BANDS:
        x = tf.sparse.to_dense(parsed[b], default_value=0.0).numpy()
        if len(x) == 257 * 257:
            arr = x.reshape(257, 257)[:256, :256]
        elif len(x) == 256 * 256:
            arr = x.reshape(256, 256)
        else:
            arr = np.zeros((256, 256), dtype=np.float32)
        bands.append(arr)
    img = np.stack(bands, axis=-1).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0)
    return img, int(parsed['label'].numpy())


def fingerprint_from_img(img):
    vals = []
    for idx in (14, 18, 15):  # Elevation, Pop_Density, LC_Forest
        v = float(img[:, :, idx].mean())
        if math.isnan(v):
            v = 0.0
        vals.append(round(v, 3))
    return tuple(vals)


def main():
    print('=' * 70)
    print('AUDIT PART 2: PyroCast v2')
    print('=' * 70)

    print('\n[A] Loading val + computing mean-feature vectors...')
    Xmeans = []
    y = []
    fps = []
    full_X = []   # keep the full images for FP/FN comparison
    ds = tf.data.TFRecordDataset(VAL_FILE)
    for i, rec in enumerate(ds):
        img, lbl = parse_sample(rec.numpy())
        Xmeans.append(img.mean(axis=(0, 1)))
        y.append(lbl)
        fps.append(fingerprint_from_img(img))
        full_X.append(img)
        if (i + 1) % 100 == 0:
            print(f'    loaded {i + 1}', flush=True)
    Xmeans = np.array(Xmeans, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    full_X = np.array(full_X, dtype=np.float32)
    print(f'  Shape Xmeans: {Xmeans.shape}, y: {y.shape}')

    print('\n[B] Predicting with v2 model (batched, smaller)...')
    model = keras.models.load_model(MODEL_FILE, safe_mode=False)
    preds = model.predict(full_X, batch_size=8, verbose=0).flatten()
    pred05 = (preds >= 0.5).astype(int)

    auc = roc_auc_score(y, preds)
    tp = int(np.sum((pred05 == 1) & (y == 1)))
    fp = int(np.sum((pred05 == 1) & (y == 0)))
    fn = int(np.sum((pred05 == 0) & (y == 1)))
    tn = int(np.sum((pred05 == 0) & (y == 0)))
    print(f'  AUC={auc:.4f}  P={tp/(tp+fp):.4f}  R={tp/(tp+fn):.4f}  TP={tp} FP={fp} FN={fn} TN={tn}')

    # ---- FP / FN per-band comparison (the part that got truncated) ----
    print('\n[A] FPs (predicted fire, actually no-fire):', fp)
    fp_idx = np.where((pred05 == 1) & (y == 0))[0]
    print(f'    {"Band":>14s}  {"FP_mean":>9s}  {"AllNeg_mean":>11s}  {"Diff":>9s}')
    for i, band in enumerate(ALL_BANDS):
        fp_m = full_X[fp_idx, :, :, i].mean() if len(fp_idx) else float('nan')
        neg_m = full_X[y == 0, :, :, i].mean()
        print(f'    {band:>14s}  {fp_m:9.4f}  {neg_m:11.4f}  {fp_m - neg_m:+9.4f}')

    print('\n    FNs (predicted no-fire, actually fire):', fn)
    fn_idx = np.where((pred05 == 0) & (y == 1))[0]
    print(f'    {"Band":>14s}  {"FN_mean":>9s}  {"AllPos_mean":>11s}  {"Diff":>9s}')
    for i, band in enumerate(ALL_BANDS):
        fn_m = full_X[fn_idx, :, :, i].mean() if len(fn_idx) else float('nan')
        pos_m = full_X[y == 1, :, :, i].mean()
        print(f'    {band:>14s}  {fn_m:9.4f}  {pos_m:11.4f}  {fn_m - pos_m:+9.4f}')

    # ---- Per-location deduplicated metrics ----
    print('\n[B] Per-location deduplicated metrics:')
    print('   Each unique fingerprint = ONE decision (mean prediction across samples).')
    by_fp_pred = defaultdict(list)
    by_fp_label = defaultdict(list)
    for i, f in enumerate(fps):
        by_fp_pred[f].append(preds[i])
        by_fp_label[f].append(y[i])

    loc_preds = []
    loc_labels = []
    mixed_fp_count = 0
    for f, ps in by_fp_pred.items():
        labels = by_fp_label[f]
        # Use most common label per location; flag if mixed
        if len(set(labels)) > 1:
            mixed_fp_count += 1
        # Mean prediction at this location
        loc_preds.append(float(np.mean(ps)))
        # Majority label
        loc_labels.append(1 if sum(labels) > len(labels) / 2 else 0)
    loc_preds = np.array(loc_preds)
    loc_labels = np.array(loc_labels)

    print(f'   Unique locations: {len(loc_preds)}  (samples: {len(y)})')
    print(f'   Mixed-label fingerprints: {mixed_fp_count} '
          f'({100 * mixed_fp_count / len(loc_preds):.1f}% — fp aliasing some distinct locations)')
    print(f'   Loc label dist: fire={int(np.sum(loc_labels == 1))}, '
          f'no-fire={int(np.sum(loc_labels == 0))}')

    loc_auc = roc_auc_score(loc_labels, loc_preds)
    print(f'   Per-location AUC: {loc_auc:.4f}  (sample-level: {auc:.4f})')
    for t in [0.3, 0.5, 0.7]:
        pred_bin = (loc_preds >= t).astype(int)
        tp_l = int(np.sum((pred_bin == 1) & (loc_labels == 1)))
        fp_l = int(np.sum((pred_bin == 1) & (loc_labels == 0)))
        fn_l = int(np.sum((pred_bin == 0) & (loc_labels == 1)))
        tn_l = int(np.sum((pred_bin == 0) & (loc_labels == 0)))
        p_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0
        r_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0
        print(f'   t={t}: TP={tp_l:3d} FP={fp_l:3d} FN={fn_l:3d} TN={tn_l:3d} | '
              f'P={p_l:.3f} R={r_l:.3f}')

    # ---- Logistic regression on mean features (key baseline) ----
    print('\n[C] Logistic regression on mean features only (NO CNN):')
    print('    Train LR on val/2, test on val/2 — gives a lower bound on what a trivial model can do.')
    np.random.seed(0)
    perm = np.random.permutation(len(y))
    half = len(perm) // 2
    train_idx, test_idx = perm[:half], perm[half:]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xmeans)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xs[train_idx], y[train_idx])
    pr_test = lr.predict_proba(Xs[test_idx])[:, 1]
    auc_lr = roc_auc_score(y[test_idx], pr_test)

    # And metrics at threshold 0.5
    pr_bin = (pr_test >= 0.5).astype(int)
    tp_lr = int(np.sum((pr_bin == 1) & (y[test_idx] == 1)))
    fp_lr = int(np.sum((pr_bin == 1) & (y[test_idx] == 0)))
    fn_lr = int(np.sum((pr_bin == 0) & (y[test_idx] == 1)))
    tn_lr = int(np.sum((pr_bin == 0) & (y[test_idx] == 0)))
    p_lr = tp_lr / (tp_lr + fp_lr) if (tp_lr + fp_lr) > 0 else 0
    r_lr = tp_lr / (tp_lr + fn_lr) if (tp_lr + fn_lr) > 0 else 0
    print(f'    LR AUC: {auc_lr:.4f}  P={p_lr:.4f}  R={r_lr:.4f}  '
          f'(CNN: AUC={auc:.4f} P={tp/(tp+fp):.4f})')

    # Show top coefficients
    coefs = list(zip(ALL_BANDS, lr.coef_[0]))
    coefs.sort(key=lambda x: -abs(x[1]))
    print('    LR coefficient ranking (most predictive at top):')
    for band, c in coefs:
        bar = '+' * max(1, int(abs(c) * 20)) if c > 0 else '-' * max(1, int(abs(c) * 20))
        print(f'      {band:>14s}: {c:+.4f}  {bar}')

    # ---- Temp_Max distribution as season proxy ----
    print('\n[D] Temp_Max distribution (proxy for fire-season check):')
    pos_t = full_X[y == 1, :, :, 8].mean(axis=(1, 2))  # 8 = Temp_Max
    neg_t = full_X[y == 0, :, :, 8].mean(axis=(1, 2))
    print('    Temp_Max bin distributions (normalized 0-1):')
    edges = np.linspace(0, 1, 11)
    print(f'    {"bin":>10s}    {"pos":>5s}   {"neg":>5s}')
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        np_count = int(((pos_t >= lo) & (pos_t < hi if i < 9 else pos_t <= hi)).sum())
        nn_count = int(((neg_t >= lo) & (neg_t < hi if i < 9 else neg_t <= hi)).sum())
        print(f'    [{lo:.1f}-{hi:.1f}]  {np_count:5d}   {nn_count:5d}')
    print(f'    pos mean Temp_Max: {pos_t.mean():.4f}')
    print(f'    neg mean Temp_Max: {neg_t.mean():.4f}')

    print('\n' + '=' * 70)
    print('SUMMARY (Part 2):')
    print('=' * 70)
    print(f'  Sample-level AUC (CNN):        {auc:.4f}')
    print(f'  Sample-level precision (CNN):  {tp/(tp+fp):.4f}')
    print(f'  Per-LOCATION AUC (CNN):        {loc_auc:.4f}')
    print(f'  LR baseline AUC (means only):  {auc_lr:.4f}')
    print(f'  LR baseline precision:         {p_lr:.4f}')


if __name__ == '__main__':
    main()
