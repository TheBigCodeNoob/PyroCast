"""
v8 evaluation + built-in audit. FIRMS positives (real fire occurrence) + DistDev.
Reports the HONEST triple (spatial-CV / temporal / space+time), a where/when/human
ablation, and univariate giveaways — so we can see if FIRMS skill is real dynamics or
a prescribed-fire land/season proxy. Compares to v7 (0.876/0.847/0.798).
"""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
try:
    from lightgbm import LGBMClassifier
    MK = lambda: LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1)
except Exception:
    MK = lambda: HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v8/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
eps = 0.1
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
y = df.label.astype(int).values
META = ['lon', 'lat', 'label', 'month', 'year', 'doy']
HUMAN = ['Pop_Density', 'DistDev', 'LC_Developed']
SPATIAL = ['Elevation', 'Pop_Density', 'DistDev', 'LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Pasture', 'LC_Wetland', 'LC_Crop', 'LC_Developed', 'NDVI', 'NDMI']
TEMPORAL = [c for c in df.columns if c not in META + SPATIAL]
FEATS = SPATIAL + TEMPORAL
print(f'rows={len(df)} fire={int(y.sum())} nofire={int((y==0).sum())} feats={len(FEATS)}')
print(f'season check mean month: fire={df[y==1].month.mean():.2f} nofire={df[y==0].month.mean():.2f}')
cells = ((np.round(df.lon * 4) / 4).astype(str) + '_' + (np.round(df.lat * 4) / 4).astype(str)).values
med = df.lon.median()


def spatial_cv(cols):
    X = np.nan_to_num(df[cols].values.astype('float32')); oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, cells):
        m = MK(); m.fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]
    return roc_auc_score(y, oof), oof


def holdout(cols, tr, te):
    X = np.nan_to_num(df[cols].values.astype('float32'))
    m = MK(); m.fit(X[tr], y[tr]); return roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])


tr_t = (df.year <= 2021).values; te_t = (df.year >= 2022).values
tr_st = ((df.year <= 2021) & (df.lon < med)).values; te_st = ((df.year >= 2022) & (df.lon >= med)).values


def triple(cols, label):
    sc, _ = spatial_cv(cols); tm = holdout(cols, tr_t, te_t); st = holdout(cols, tr_st, te_st)
    print(f'  {label:<22s} spatialCV={sc:.4f}  temporal={tm:.4f}  space+time={st:.4f}')
    return sc, tm, st


print('\n=== HONEST TRIPLE (compare v7: 0.876 / 0.847 / 0.798) ===')
full = triple(FEATS, 'FULL (where+when)')
print('\n=== ABLATION ===')
triple(SPATIAL, 'SPATIAL only')
triple(TEMPORAL, 'TEMPORAL only')
triple(HUMAN, 'HUMAN-access only')
triple([c for c in FEATS if c not in HUMAN], 'NO human-access')

print('\n=== univariate giveaways (top 12) ===')
uni = sorted([(f, max(roc_auc_score(y, df[f].values), 1 - roc_auc_score(y, df[f].values))) for f in FEATS], key=lambda t: -t[1])
for f, a in uni[:12]:
    print(f'  {f:<14s} |AUC|={a:.3f}')

_, oof = spatial_cv(FEATS)
print(f'\nOperational: PR-AUC={average_precision_score(y, oof):.4f} (prevalence {y.mean():.2f})')
print(f'\nVERDICT vs v7: space+time {full[2]:.4f} (v7 0.798). {"BETTER" if full[2] > 0.81 else "similar/worse"}')
