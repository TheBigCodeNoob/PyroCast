"""
v4b: squeeze the v4 features harder.
  - Engineer explicit TREND / interaction features (the raw model has PDSI and
    PDSI_90dago separately, but the *trajectory* is the signal).
  - Tuned HGB + RF + ExtraTrees, plus a stacked logistic meta-model.
  - Leakage-free location-grouped val + GroupKFold CV.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

d = np.load('v4_features.npz', allow_pickle=True)
X, y, groups = d['X'], d['y'], d['groups']
names = list(d['feature_names'])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
idx = {n: i for i, n in enumerate(names)}


def col(name):
    return X[:, idx[name]]


# ---- Engineered trend / interaction features ----
eng, eng_names = [], []
def add(name, vals):
    eng.append(vals.astype(np.float32)); eng_names.append(name)

add('pdsi_trend', col('PDSI_mean') - col('PDSI_90dago_mean'))            # drought worsening (<0)
add('ndvi_curing', col('NDVI_mean') - col('NDVI_90d_mean'))             # browning/curing
add('erc_trend', col('ERC_mean') - col('ERC_30d_mean'))
add('fm100_trend', col('FM100_mean') - col('FM100_30d_mean'))           # fuel drying
add('precip_ratio', col('Precip_30d_mean') / (col('Precip_90d_mean') + 0.02))
add('precip90_deficit', 1.0 - col('Precip_90d_mean'))
add('vpd_x_drydrought', col('VPD_mean') * (1.0 - col('PDSI_mean')))     # hot-dry-air x drought
add('erc_x_fm100', col('ERC_30d_mean') * (1.0 - col('FM100_30d_mean')))
add('dry_index', (1 - col('Precip_90d_mean')) + col('VPD_mean') + (1 - col('PDSI_mean')))
add('ndmi_curing', col('NDMI_mean') - col('NDVI_90d_mean'))

Xe = np.column_stack([X] + eng)
all_names = names + eng_names

# Drop zero-variance
var = Xe.var(axis=0)
keep = var > 1e-12
Xe = Xe[:, keep]
all_names = [all_names[i] for i in range(len(all_names)) if keep[i]]
print(f'Features: {Xe.shape[1]} ({len(eng_names)} engineered) | samples {Xe.shape[0]} | locations {len(set(groups))}')

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
tr, va = next(gss.split(Xe, y, groups))
Xtr, ytr, gtr, Xva, yva = Xe[tr], y[tr], groups[tr], Xe[va], y[va]
print(f'Train {Xtr.shape} ({len(set(gtr))} locs) | Val {Xva.shape} ({len(set(groups[va]))} locs)')
print('=' * 60)


def make_models():
    return {
        'HGB': HistGradientBoostingClassifier(max_iter=1200, learning_rate=0.02,
                max_leaf_nodes=63, l2_regularization=3.0, min_samples_leaf=40,
                early_stopping=False, random_state=0),
        'RF': RandomForestClassifier(n_estimators=1000, max_features='sqrt',
                min_samples_leaf=3, n_jobs=-1, random_state=0),
        'ET': ExtraTreesClassifier(n_estimators=1000, max_features='sqrt',
                min_samples_leaf=3, n_jobs=-1, random_state=0),
    }


# Fit base models, collect val preds + OOF preds (for stacking) via GroupKFold
base = make_models()
val_preds, oof = {}, {}
gkf = GroupKFold(n_splits=5)
for nm, m in base.items():
    m.fit(Xtr, ytr)
    val_preds[nm] = m.predict_proba(Xva)[:, 1]
    o = np.zeros(len(ytr))
    for a, b in gkf.split(Xtr, ytr, gtr):
        mm = make_models()[nm]; mm.fit(Xtr[a], ytr[a]); o[b] = mm.predict_proba(Xtr[b])[:, 1]
    oof[nm] = o
    print(f'  {nm:<4s} val={roc_auc_score(yva, val_preds[nm]):.4f}  CV={roc_auc_score(ytr, o):.4f}')

# Simple average blend
avg = np.mean([val_preds[k] for k in base], axis=0)
print(f'  AVG  val={roc_auc_score(yva, avg):.4f}')

# Stacked logistic meta-model on OOF
Z_tr = np.column_stack([oof[k] for k in base])
Z_va = np.column_stack([val_preds[k] for k in base])
meta = LogisticRegression(max_iter=2000).fit(Z_tr, ytr)
stack_val = meta.predict_proba(Z_va)[:, 1]
print(f'  STACK val={roc_auc_score(yva, stack_val):.4f}')

best = max(roc_auc_score(yva, avg), roc_auc_score(yva, stack_val),
          *[roc_auc_score(yva, val_preds[k]) for k in base])
print('=' * 60)
print(f'BEST val AUC: {best:.4f}   {"*** >=0.80 ***" if best >= 0.80 else "(target 0.80)"}')
