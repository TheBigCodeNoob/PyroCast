"""
Audit script for v2 model.
Goal: figure out whether 0.94 precision is real or a measurement artifact.

Checks performed:
  1. Val class distribution (fire vs no-fire).
  2. Per-channel mean by class (looking for label-correlated features).
  3. Probability histogram by class (is the model confident or hedging?).
  4. Precision/recall at multiple thresholds.
  5. Ablation: zero out weather channels and re-score.
  6. Ablation: zero out S2 imagery channels and re-score.
  7. Ablation: keep only weather+elevation, blank everything else.
  8. The zero-fingerprint cluster: how big, what labels, what predictions.
  9. Cross-band correlation between pos/neg distributions.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import math
import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

VAL_FILE = 'Training Data Florida/Florida_Spatial_Val_v2.tfrecord'
TRAIN_FILE = 'Training Data Florida/Florida_Spatial_Train_v2.tfrecord'
MODEL_FILE = 'best_robust_fire_model_v2.keras'

ALL_BANDS = [
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
    'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip', 'ERC', 'FM100',
    'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open', 'Pop_Density'
]

WEATHER_IDX = list(range(8, 14))     # Temp_Max..FM100
S2_IDX      = list(range(0, 8))      # Blue..NDMI
TERRAIN_IDX = [14, 15, 16, 17, 18]   # Elevation..Pop_Density


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


def load_records(path, limit=None):
    X, y = [], []
    ds = tf.data.TFRecordDataset(path)
    for i, rec in enumerate(ds):
        if limit and i >= limit:
            break
        img, lbl = parse_sample(rec.numpy())
        X.append(img)
        y.append(lbl)
        if (i + 1) % 100 == 0:
            print(f'    loaded {i + 1}', flush=True)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def fingerprint(img):
    vals = []
    for idx in (14, 18, 15):   # Elevation, Pop_Density, LC_Forest
        v = float(img[:, :, idx].mean())
        if math.isnan(v):
            v = 0.0
        vals.append(round(v, 3))
    return tuple(vals)


def metrics_at_threshold(y, preds, thresh):
    pred_bin = (preds >= thresh).astype(int)
    tp = int(np.sum((pred_bin == 1) & (y == 1)))
    fp = int(np.sum((pred_bin == 1) & (y == 0)))
    fn = int(np.sum((pred_bin == 0) & (y == 1)))
    tn = int(np.sum((pred_bin == 0) & (y == 0)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return tp, fp, fn, tn, p, r


def main():
    print('=' * 70)
    print('AUDIT: PyroCast v2')
    print('=' * 70)

    print('\n[1/9] Loading validation set...')
    Xv, yv = load_records(VAL_FILE)
    print(f'  Val shape: {Xv.shape}')
    print(f'  Label distribution: fire={int(np.sum(yv == 1))}, no-fire={int(np.sum(yv == 0))}')
    print(f'  Class balance: fire={np.mean(yv == 1):.3f}')

    print('\n[2/9] Per-channel mean by class:')
    print(f'  {"Band":>14s}  {"pos_mean":>9s}  {"neg_mean":>9s}  {"diff":>7s}  {"pos_std":>7s}  {"neg_std":>7s}')
    diffs = []
    for i, band in enumerate(ALL_BANDS):
        pm = Xv[yv == 1, :, :, i].mean()
        nm = Xv[yv == 0, :, :, i].mean()
        ps = Xv[yv == 1, :, :, i].std()
        ns = Xv[yv == 0, :, :, i].std()
        d = pm - nm
        diffs.append((band, pm, nm, d))
        print(f'  {band:>14s}  {pm:9.4f}  {nm:9.4f}  {d:+7.4f}  {ps:7.4f}  {ns:7.4f}')

    print('\n[3/9] Loading v2 model + running predictions...')
    model = keras.models.load_model(MODEL_FILE, safe_mode=False)
    print(f'  Model input: {model.input_shape}')

    preds = model.predict(Xv, batch_size=16, verbose=1).flatten()
    print(f'  Predictions: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}')
    print(f'  Pos predictions mean: {preds[yv == 1].mean():.4f}')
    print(f'  Neg predictions mean: {preds[yv == 0].mean():.4f}')

    print('\n[4/9] Prediction-probability histogram (val):')
    print(f'  {"bin":>10s}   {"total":>5s}  {"fire":>5s}  {"nofire":>6s}')
    edges = np.linspace(0, 1, 11)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            m = (preds >= lo) & (preds < hi)
        else:
            m = (preds >= lo) & (preds <= hi)
        tot = int(m.sum())
        nf = int(np.sum(yv[m] == 1))
        nn = int(np.sum(yv[m] == 0))
        print(f'  [{lo:.1f}-{hi:.1f}]  {tot:5d}  {nf:5d}  {nn:6d}')

    print('\n[5/9] Precision/recall at multiple thresholds:')
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp, fp, fn, tn, p, r = metrics_at_threshold(yv, preds, t)
        print(f'  t={t:.1f}: TP={tp:4d} FP={fp:4d} FN={fn:4d} TN={tn:4d} | P={p:.3f} R={r:.3f}')
    auc = roc_auc_score(yv, preds)
    print(f'  AUC: {auc:.4f}')

    print('\n[6/9] Ablation: feature-group importance')
    print('  Strategy: replace channel groups with 0.5 (neutral) and re-score AUC.')

    def replace_channels(X, idx, value=0.5):
        Y = X.copy()
        Y[:, :, :, idx] = value
        return Y

    X_no_w = replace_channels(Xv, WEATHER_IDX)
    preds_no_w = model.predict(X_no_w, batch_size=16, verbose=0).flatten()
    auc_no_w = roc_auc_score(yv, preds_no_w)
    print(f'  AUC, weather wiped:       {auc_no_w:.4f}  (full: {auc:.4f}, delta {auc_no_w - auc:+.4f})')

    X_no_s2 = replace_channels(Xv, S2_IDX)
    preds_no_s2 = model.predict(X_no_s2, batch_size=16, verbose=0).flatten()
    auc_no_s2 = roc_auc_score(yv, preds_no_s2)
    print(f'  AUC, S2 imagery wiped:    {auc_no_s2:.4f}  (full: {auc:.4f}, delta {auc_no_s2 - auc:+.4f})')

    X_no_terr = replace_channels(Xv, TERRAIN_IDX)
    preds_no_terr = model.predict(X_no_terr, batch_size=16, verbose=0).flatten()
    auc_no_terr = roc_auc_score(yv, preds_no_terr)
    print(f'  AUC, terrain/LC wiped:    {auc_no_terr:.4f}  (full: {auc:.4f}, delta {auc_no_terr - auc:+.4f})')

    # Keep only weather (zero everything else)
    X_only_w = np.full_like(Xv, 0.5)
    X_only_w[:, :, :, WEATHER_IDX] = Xv[:, :, :, WEATHER_IDX]
    preds_only_w = model.predict(X_only_w, batch_size=16, verbose=0).flatten()
    auc_only_w = roc_auc_score(yv, preds_only_w)
    print(f'  AUC, ONLY weather:        {auc_only_w:.4f}  (full: {auc:.4f}, delta {auc_only_w - auc:+.4f})')

    # Keep only S2
    X_only_s2 = np.full_like(Xv, 0.5)
    X_only_s2[:, :, :, S2_IDX] = Xv[:, :, :, S2_IDX]
    preds_only_s2 = model.predict(X_only_s2, batch_size=16, verbose=0).flatten()
    auc_only_s2 = roc_auc_score(yv, preds_only_s2)
    print(f'  AUC, ONLY S2 imagery:     {auc_only_s2:.4f}  (full: {auc:.4f}, delta {auc_only_s2 - auc:+.4f})')

    # Keep only terrain
    X_only_terr = np.full_like(Xv, 0.5)
    X_only_terr[:, :, :, TERRAIN_IDX] = Xv[:, :, :, TERRAIN_IDX]
    preds_only_terr = model.predict(X_only_terr, batch_size=16, verbose=0).flatten()
    auc_only_terr = roc_auc_score(yv, preds_only_terr)
    print(f'  AUC, ONLY terrain/LC:     {auc_only_terr:.4f}  (full: {auc:.4f}, delta {auc_only_terr - auc:+.4f})')

    print('\n[7/9] Zero-fingerprint cluster:')
    fps = [fingerprint(Xv[i]) for i in range(len(Xv))]
    zero_mask = np.array([fp == (0.0, 0.0, 0.0) for fp in fps])
    n_zero = int(zero_mask.sum())
    print(f'  Val samples with FP (0,0,0): {n_zero} of {len(Xv)} ({100 * n_zero / len(Xv):.1f}%)')
    if n_zero:
        print(f'    Their labels: fire={int(np.sum(yv[zero_mask] == 1))}, nofire={int(np.sum(yv[zero_mask] == 0))}')
        print(f'    Their prediction mean: {preds[zero_mask].mean():.4f}')

    # Look at most common fingerprints
    from collections import Counter
    fp_counter = Counter(fps)
    print('  Top 5 most-common val fingerprints:')
    for fp, cnt in fp_counter.most_common(5):
        idxs = [i for i, f in enumerate(fps) if f == fp]
        labels = yv[idxs]
        ps = preds[idxs]
        print(f'    {fp}  count={cnt}  fire={int(np.sum(labels == 1))} nofire={int(np.sum(labels == 0))}  pred_mean={ps.mean():.3f}')

    print('\n[8/9] Examining FPs and FNs at default threshold 0.5:')
    pred05 = (preds >= 0.5).astype(int)
    fp_idx = np.where((pred05 == 1) & (yv == 0))[0]
    fn_idx = np.where((pred05 == 0) & (yv == 1))[0]
    print(f'  FPs (predicted fire, actually no-fire): {len(fp_idx)}')
    if len(fp_idx):
        # Per-band mean for FPs vs all negatives
        print(f'    {"Band":>14s}  {"FP_mean":>9s}  {"AllNeg_mean":>11s}  {"Diff":>7s}')
        for i, band in enumerate(ALL_BANDS):
            fp_m = Xv[fp_idx, :, :, i].mean()
            neg_m = Xv[yv == 0, :, :, i].mean()
            print(f'    {band:>14s}  {fp_m:9.4f}  {neg_m:11.4f}  {fp_m - neg_m:+7.4f}')
    print(f'\n  FNs (predicted no-fire, actually fire): {len(fn_idx)}')
    if len(fn_idx):
        print(f'    {"Band":>14s}  {"FN_mean":>9s}  {"AllPos_mean":>11s}  {"Diff":>7s}')
        for i, band in enumerate(ALL_BANDS):
            fn_m = Xv[fn_idx, :, :, i].mean()
            pos_m = Xv[yv == 1, :, :, i].mean()
            print(f'    {band:>14s}  {fn_m:9.4f}  {pos_m:11.4f}  {fn_m - pos_m:+7.4f}')

    print('\n[9/9] Quick single-pixel logistic regression baseline:')
    print('  Question: how much do the mean-feature values alone predict fire?')
    # Compute mean per channel per sample; fit logistic regression
    Xv_means = Xv.mean(axis=(1, 2))  # (N, 19)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xv_means_s = scaler.fit_transform(Xv_means)
    # Fit on half, test on half (NB: not strictly held-out from training set; this is
    # just a sanity baseline on the val set itself).
    np.random.seed(0)
    perm = np.random.permutation(len(yv))
    half = len(perm) // 2
    train_idx, test_idx = perm[:half], perm[half:]
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xv_means_s[train_idx], yv[train_idx])
    pr_test = lr.predict_proba(Xv_means_s[test_idx])[:, 1]
    auc_lr = roc_auc_score(yv[test_idx], pr_test)
    print(f'  Logistic regression on mean features only, AUC: {auc_lr:.4f}')
    # Coefficients to see which features dominate
    print('  Logistic regression coefficients (standardized):')
    coefs = list(zip(ALL_BANDS, lr.coef_[0]))
    coefs.sort(key=lambda x: -abs(x[1]))
    for band, c in coefs:
        print(f'    {band:>14s}: {c:+.4f}')

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'  Full model AUC:                {auc:.4f}')
    print(f'  AUC w/o weather:               {auc_no_w:.4f}  (drop: {auc - auc_no_w:.4f})')
    print(f'  AUC w/o S2:                    {auc_no_s2:.4f}  (drop: {auc - auc_no_s2:.4f})')
    print(f'  AUC w/o terrain:               {auc_no_terr:.4f}  (drop: {auc - auc_no_terr:.4f})')
    print(f'  AUC weather-only:              {auc_only_w:.4f}')
    print(f'  AUC S2-only:                   {auc_only_s2:.4f}')
    print(f'  AUC terrain-only:              {auc_only_terr:.4f}')
    print(f'  LR baseline on mean features:  {auc_lr:.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
