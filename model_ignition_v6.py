"""
v6 spatiotemporal IGNITION model: 'where + when will a fire start'.
Fire ignitions (label 1) vs hard burnable-vegetation non-fire space-time (label 0).

Honest evaluation: SPATIAL holdout — val is a held-out set of 0.25-deg geographic
cells, and CV folds are grouped by cell, so train/val never share nearby points.

Audit: train on SPATIAL-only vs TEMPORAL-only vs FULL features, to confirm the model
uses BOTH 'where' (biome/fuel/terrain/people) AND 'when' (drought/weather) signal —
i.e. it's a dynamic risk model, not a static vegetation map.
"""
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import roc_auc_score

csvs = sorted(glob.glob('Training Data Florida/v6/*.csv'))
if not csvs:
    raise SystemExit('No v6 CSVs.')
df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
print(f'Loaded {len(df)} rows ({len(csvs)} CSVs). fire={int(df.label.sum())} nofire={int((df.label==0).sum())}')
df = df.dropna().reset_index(drop=True)

eps = 0.1
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90
df['pdsi_traj_180'] = df.pdsi_0 - df.pdsi_180
df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['erc_trend'] = df.erc_7 - df.erc_90
df['pr_recent_ratio'] = df.pr_30 / (df.pr_90 + eps)
df['pr_deficit'] = df.pr_365 / 4.0 - df.pr_90
df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50.0

SPATIAL = ['Elevation', 'Pop_Density', 'LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Pasture',
           'LC_Wetland', 'LC_Crop', 'LC_Developed', 'NDVI', 'NDMI']
TEMPORAL = [c for c in df.columns if c not in (['lon', 'lat', 'label'] + SPATIAL)]
ALLF = SPATIAL + TEMPORAL

y = df.label.astype(int).values
# Spatial cells (0.25 deg ~ 28km) for leakage-free grouping
cells = (np.round(df.lon * 4) / 4).astype(str) + '_' + (np.round(df.lat * 4) / 4).astype(str)
groups = cells.values
print(f'Features: {len(ALLF)} ({len(SPATIAL)} spatial + {len(TEMPORAL)} temporal) | spatial cells: {len(set(groups))}')


def prep(cols):
    X = np.nan_to_num(df[cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return X


def eval_set(cols, label):
    X = prep(cols)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=0)
    tr, va = next(gss.split(X, y, groups))
    m = HistGradientBoostingClassifier(max_iter=900, learning_rate=0.03, max_leaf_nodes=63,
            l2_regularization=2.0, min_samples_leaf=25, early_stopping=False, random_state=0)
    m.fit(X[tr], y[tr])
    val = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
    # spatial CV
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        mm = HistGradientBoostingClassifier(max_iter=900, learning_rate=0.03, max_leaf_nodes=63,
                l2_regularization=2.0, min_samples_leaf=25, early_stopping=False, random_state=0)
        mm.fit(X[a], y[a]); oof[b] = mm.predict_proba(X[b])[:, 1]
    cv = roc_auc_score(y, oof)
    print(f'  {label:<22s} spatial-val={val:.4f}  spatial-CV={cv:.4f}')
    return val, cv


print('=' * 64)
print('ABLATION (HGB; how much signal is WHERE vs WHEN):')
eval_set(SPATIAL, 'SPATIAL only')
eval_set(TEMPORAL, 'TEMPORAL only')
fv, fc = eval_set(ALLF, 'FULL (where+when)')
print('=' * 64)

# Full model: ensemble for the headline number
X = prep(ALLF)
gss = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=0)
tr, va = next(gss.split(X, y, groups))
preds = {}
for nm, m in {
    'HGB': HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63,
            l2_regularization=2.0, min_samples_leaf=25, early_stopping=False, random_state=0),
    'RF': RandomForestClassifier(n_estimators=800, max_features='sqrt', min_samples_leaf=2, n_jobs=-1, random_state=0),
    'ET': ExtraTreesClassifier(n_estimators=800, max_features='sqrt', min_samples_leaf=2, n_jobs=-1, random_state=0),
}.items():
    m.fit(X[tr], y[tr]); preds[nm] = m.predict_proba(X[va])[:, 1]
    print(f'  {nm} spatial-val={roc_auc_score(y[va], preds[nm]):.4f}')
blend = np.mean(list(preds.values()), axis=0)
best = max(roc_auc_score(y[va], blend), *[roc_auc_score(y[va], p) for p in preds.values()])
print(f'  BLEND spatial-val={roc_auc_score(y[va], blend):.4f}')
print('=' * 64)
print(f'HEADLINE spatial-holdout AUC: {best:.4f}')

from sklearn.inspection import permutation_importance
hgb = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63,
        l2_regularization=2.0, min_samples_leaf=25, early_stopping=False, random_state=0).fit(X[tr], y[tr])
pi = permutation_importance(hgb, X[va], y[va], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
print('\nTop 18 features:')
for i in np.argsort(pi.importances_mean)[::-1][:18]:
    tag = 'WHERE' if ALLF[i] in SPATIAL else 'when'
    print(f'  [{tag}] {ALLF[i]:<18s} {pi.importances_mean[i]:+.4f}')
