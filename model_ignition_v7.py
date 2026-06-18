"""
v7 evaluation: did the season-matched negatives deflate the v6 0.93?
Same model/features/spatial-grouping as v6 (only the negative dates changed).
Reports: spatial-holdout AUC, where/when ablation, whether tmmx/vpd are STILL
giveaways (they shouldn't be now), and a TEMPORAL holdout (train <=2021, test >=2022).
"""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v7/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
eps = 0.1
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['pdsi_traj_180'] = df.pdsi_0 - df.pdsi_180
df['vpd_trend'] = df.vpd_7 - df.vpd_90; df['erc_trend'] = df.erc_7 - df.erc_90
df['pr_recent_ratio'] = df.pr_30 / (df.pr_90 + eps); df['pr_deficit'] = df.pr_365 / 4 - df.pr_90
df['fm100_trend'] = df.fm100_30 - df.fm100_90; df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
y = df.label.astype(int).values
META = ['lon', 'lat', 'label', 'month', 'year', 'doy']
SPATIAL = ['Elevation', 'Pop_Density', 'LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Pasture', 'LC_Wetland', 'LC_Crop', 'LC_Developed', 'NDVI', 'NDMI']
TEMPORAL = [c for c in df.columns if c not in META + SPATIAL]
FEATS = SPATIAL + TEMPORAL
print(f'rows={len(df)} fire={int(y.sum())} nofire={int((y==0).sum())} features={len(FEATS)}')
print(f'season check -> mean month: fire={df[y==1].month.mean():.2f} nofire={df[y==0].month.mean():.2f} (matched if ~equal)')

cells = ((np.round(df.lon * 4) / 4).astype(str) + '_' + (np.round(df.lat * 4) / 4).astype(str)).values


def cv(cols, groups=cells):
    X = np.nan_to_num(df[cols].values.astype('float32'))
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
        m.fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]
    return roc_auc_score(y, oof)


print('\n=== SPATIAL-HOLDOUT AUC (compare to v6 0.928) ===')
print(f'  FULL (where+when)   {cv(FEATS):.4f}   [v6 was 0.928]')
print(f'  SPATIAL only        {cv(SPATIAL):.4f}   [v6 0.860]')
print(f'  TEMPORAL only       {cv(TEMPORAL):.4f}   [v6 0.858]')

print('\n=== Are temp/VPD STILL giveaways now that season is matched? (univariate |AUC|) ===')
for f in ['tmmx_90', 'vpd_90', 'tmmx_30', 'Pop_Density', 'fm100_90', 'pdsi_traj_90', 'pr_deficit', 'dryness']:
    a = roc_auc_score(y, df[f].values)
    print(f'  {f:<14s} |AUC|={max(a,1-a):.3f}  (v6: tmmx_90 0.728, vpd_90 0.731, Pop 0.785)')

print('\n=== TEMPORAL HOLDOUT (train years <=2021, test years >=2022) — realistic forecast ===')
tr = df.year <= 2021; te = df.year >= 2022
print(f'  train n={int(tr.sum())} (fire={int(y[tr.values].sum())}) | test n={int(te.sum())} (fire={int(y[te.values].sum())})')
Xtr = np.nan_to_num(df[tr][FEATS].values.astype('float32')); Xte = np.nan_to_num(df[te][FEATS].values.astype('float32'))
m = HistGradientBoostingClassifier(max_iter=700, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
m.fit(Xtr, y[tr.values])
print(f'  temporal-holdout AUC = {roc_auc_score(y[te.values], m.predict_proba(Xte)[:,1]):.4f}')

print('\n=== COMBINED honest estimate: spatial cells AND future years both held out ===')
# train on pre-2022 + western cells, test on 2022+ eastern cells (toughest)
med = df.lon.median()
tr2 = (df.year <= 2021) & (df.lon < med); te2 = (df.year >= 2022) & (df.lon >= med)
if tr2.sum() > 100 and te2.sum() > 50:
    Xtr2 = np.nan_to_num(df[tr2][FEATS].values.astype('float32')); Xte2 = np.nan_to_num(df[te2][FEATS].values.astype('float32'))
    m2 = HistGradientBoostingClassifier(max_iter=700, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
    m2.fit(Xtr2, y[tr2.values])
    print(f'  space+time held out: train n={int(tr2.sum())} test n={int(te2.sum())} -> AUC = {roc_auc_score(y[te2.values], m2.predict_proba(Xte2)[:,1]):.4f}')
