"""Finalize the deployable v2: hgb+lgb two-member ensemble (RF dropped: it dragged accuracy
AND made the bundle 445MB). Computes the paired delta vs the old 2017-19 regime for THIS
composition (arm-A refits are cheap: 4.6k rows), then fits on all 98,848 rows and saves
best_model_fl_ensemble_v2.joblib.  conda run -n base python finalize_v2.py"""
import warnings, numpy as np, joblib
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
from train_fl_hist import load_combined

df, FEATS, X, y, yr, block, src_cur = load_combined()


def members():
    return [('hgb', HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)),
            ('lgb', LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1))]


# arm A (2017-19 train) OOFs for the same composition; arm B comes from the saved eval
oofA = {n: np.full(len(y), np.nan) for n, _ in members()}
for tg, eg in GroupKFold(5).split(X, y, block):
    te = eg[yr[eg] >= 2020]
    trA = tg[(yr[tg] <= 2019) & src_cur[tg]]
    if len(te) == 0 or y[trA].sum() < 20:
        continue
    for n, m in members():
        m.fit(X[trA], y[trA]); oofA[n][te] = m.predict_proba(X[te])[:, 1]

z = np.load('oof_v2_members.npz', allow_pickle=True)
mask = z['mask']; yt = z['y']; bt = z['block']
ensB = (z['hgb'] + z['lgb']) / 2
ensA = (oofA['hgb'][mask] + oofA['lgb'][mask]) / 2
assert not np.isnan(ensA).any()
aA, aB = roc_auc_score(yt, ensA), roc_auc_score(yt, ensB)
_, inv = np.unique(bt, return_inverse=True)
rng = np.random.default_rng(0); nb = inv.max() + 1
dB, dD = [], []
for _ in range(1000):
    w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[inv]
    if w[yt == 1].sum() and w[yt == 0].sum():
        b = roc_auc_score(yt, ensB, sample_weight=w)
        dB.append(b); dD.append(b - roc_auc_score(yt, ensA, sample_weight=w))
ciB = np.percentile(dB, [2.5, 97.5]); ciD = np.percentile(dD, [2.5, 97.5])
ap = average_precision_score(yt, ensB)
print(f'hgb+lgb: A(2017-19)={aA:.4f}  B(2001-19)={aB:.4f}  delta={aB - aA:+.4f}')
print(f'B CI=[{ciB[0]:.4f},{ciB[1]:.4f}]  delta CI=[{ciD[0]:+.4f},{ciD[1]:+.4f}]  P(delta>0)={np.mean(np.array(dD) > 0):.3f}  AP={ap:.4f}')

fitted = [m.fit(X, y) or m for _, m in members()]
joblib.dump({'models': fitted, 'features': FEATS, 'region': 'Florida',
             'honest_metrics': {'auc': round(float(aB), 4), 'auc_ci': [round(float(ciB[0]), 4), round(float(ciB[1]), 4)],
                                'auc_prev_regime': round(float(aA), 4),
                                'delta_ci': [round(float(ciD[0]), 4), round(float(ciD[1]), 4)],
                                'ap': round(float(ap), 4)},
             'trained_on': f'{len(df)} FL rows 2001-2020, 56,834 real ignitions (hgb+lgb ensemble), blocked space+time AUROC {aB:.4f}'},
            'best_model_fl_ensemble_v2.joblib')
import os
print(f"Saved best_model_fl_ensemble_v2.joblib ({os.path.getsize('best_model_fl_ensemble_v2.joblib') / 1e6:.1f} MB)")
