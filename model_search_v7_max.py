"""
STAGE A: squeeze maximum honest performance from the existing v7 data (no new export).
Tests richer feature engineering + multiple model families + stacking, all judged on the
HONEST triple (spatial-CV / temporal / space+time). Saves the best model.
"""
import glob, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
try:
    from lightgbm import LGBMClassifier
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v7/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
eps = 0.1
# base engineered
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['pdsi_traj_180'] = df.pdsi_0 - df.pdsi_180
df['vpd_trend'] = df.vpd_7 - df.vpd_90; df['erc_trend'] = df.erc_7 - df.erc_90
df['pr_recent_ratio'] = df.pr_30 / (df.pr_90 + eps); df['pr_deficit'] = df.pr_365 / 4 - df.pr_90
df['fm100_trend'] = df.fm100_30 - df.fm100_90; df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
# richer interactions (new this stage)
df['vpd_x_erc'] = df.vpd_30 * df.erc_30
df['dry_x_forest'] = df.dryness * df.LC_Forest
df['pr7_over_pr30'] = df.pr_7 / (df.pr_30 + eps)
df['pdsi_mean'] = df[['pdsi_0', 'pdsi_30', 'pdsi_90']].mean(axis=1)
df['fuel_dry'] = (1 - df.fm100_90 / 40.0) * df.NDVI
df['heat'] = df.tmmx_30 * df.vpd_30

y = df.label.astype(int).values
META = ['lon', 'lat', 'label', 'month', 'year', 'doy']
FEATS = [c for c in df.columns if c not in META]
print(f'rows={len(df)} fire={int(y.sum())} feats={len(FEATS)} lgbm={HAVE_LGB}')
cells = ((np.round(df.lon * 4) / 4).astype(str) + '_' + (np.round(df.lat * 4) / 4).astype(str)).values
med = df.lon.median()


def spatial_cv(make, cols):
    X = np.nan_to_num(df[cols].values.astype('float32')); oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, cells):
        m = make(); m.fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]
    return roc_auc_score(y, oof), oof


def holdout(make, cols, mask_tr, mask_te):
    X = np.nan_to_num(df[cols].values.astype('float32'))
    m = make(); m.fit(X[mask_tr], y[mask_tr])
    p = m.predict_proba(X[mask_te])[:, 1]
    return roc_auc_score(y[mask_te], p)


def triple(make, cols):
    sc, _ = spatial_cv(make, cols)
    tmp = holdout(make, cols, (df.year <= 2021).values, (df.year >= 2022).values)
    st = holdout(make, cols, ((df.year <= 2021) & (df.lon < med)).values, ((df.year >= 2022) & (df.lon >= med)).values)
    return sc, tmp, st


MK = {
    'HGB': lambda: HistGradientBoostingClassifier(max_iter=700, learning_rate=0.03, max_leaf_nodes=63, l2_regularization=3.0, min_samples_leaf=30, random_state=0),
    'RF': lambda: RandomForestClassifier(n_estimators=700, max_features='sqrt', min_samples_leaf=2, n_jobs=-1, random_state=0),
    'ET': lambda: ExtraTreesClassifier(n_estimators=700, max_features='sqrt', min_samples_leaf=2, n_jobs=-1, random_state=0),
}
if HAVE_LGB:
    MK['LGBM'] = lambda: LGBMClassifier(n_estimators=900, learning_rate=0.02, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1)

print('\n=== model families (honest triple) on richer features ===')
print(f'{"model":<8s} {"spatialCV":>9s} {"temporal":>9s} {"space+time":>10s}')
results = {}
for nm, mk in MK.items():
    sc, tmp, st = triple(mk, FEATS)
    results[nm] = (sc, tmp, st)
    print(f'{nm:<8s} {sc:>9.4f} {tmp:>9.4f} {st:>10.4f}')

# Stacked ensemble (OOF spatial-CV preds -> logistic meta)
print('\n=== stacked ensemble ===')
oofs, tests = {}, {}
tr_m = (df.year <= 2021).values; te_m = (df.year >= 2022).values
for nm, mk in MK.items():
    _, oof = spatial_cv(mk, FEATS); oofs[nm] = oof
Z = np.column_stack([oofs[k] for k in MK])
meta = LogisticRegression(max_iter=2000).fit(Z, y)
stack_cv = roc_auc_score(y, meta.predict_proba(Z)[:, 1])
print(f'  stack spatial-CV (optimistic, meta refit on oof) = {stack_cv:.4f}')

best = max(results.items(), key=lambda kv: kv[1][2])
print(f'\nBEST by space+time: {best[0]} -> spatialCV={best[1][0]:.4f} temporal={best[1][1]:.4f} space+time={best[1][2]:.4f}')

# Operational reality: PR-AUC + precision@recall at the eval prevalence
X = np.nan_to_num(df[FEATS].values.astype('float32'))
_, oof = spatial_cv(MK[best[0]], FEATS)
print(f'\nOperational (eval prevalence {y.mean():.2f}): PR-AUC={average_precision_score(y, oof):.4f} (vs baseline {y.mean():.2f})')

import joblib
m = MK[best[0]](); m.fit(X, y)
joblib.dump({'model': m, 'features': FEATS, 'best': best[0], 'triple': best[1]}, 'v7_best_model.joblib')
print(f'\nSaved v7_best_model.joblib ({best[0]})')
print(json.dumps({'stage': 'A', 'best_model': best[0], 'spatialCV': best[1][0], 'temporal': best[1][1], 'spacetime': best[1][2]}))
