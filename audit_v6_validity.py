"""
Adversarial audit of the v6 0.93 AUC: is it real or inflated?
Tests: univariate 'giveaway' features, spatial-leakage robustness (cell size +
regional holdout), population-skew, weather/season-matched hard subset, duplicate
check, and base-rate reality. No agents — pure empirical stress-testing.
"""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v6/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
eps = 0.1
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['pdsi_traj_180'] = df.pdsi_0 - df.pdsi_180
df['vpd_trend'] = df.vpd_7 - df.vpd_90; df['erc_trend'] = df.erc_7 - df.erc_90
df['pr_recent_ratio'] = df.pr_30 / (df.pr_90 + eps); df['pr_deficit'] = df.pr_365 / 4 - df.pr_90
df['fm100_trend'] = df.fm100_30 - df.fm100_90; df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
y = df.label.astype(int).values
SPATIAL = ['Elevation', 'Pop_Density', 'LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Pasture', 'LC_Wetland', 'LC_Crop', 'LC_Developed', 'NDVI', 'NDMI']
TEMPORAL = [c for c in df.columns if c not in (['lon', 'lat', 'label'] + SPATIAL)]
FEATS = SPATIAL + TEMPORAL
print(f'rows={len(df)} fire={int(y.sum())} nofire={int((y==0).sum())} features={len(FEATS)}')


def cells(d, deg):
    return ((np.round(d.lon / deg) * deg).round(3).astype(str) + '_' + (np.round(d.lat / deg) * deg).round(3).astype(str)).values


def cv(sub, cols, deg=0.25, mi=400):
    X = np.nan_to_num(sub[cols].values.astype('float32'))
    yy = sub.label.astype(int).values
    g = cells(sub, deg)
    if len(set(g)) < 6 or yy.sum() < 20 or (yy == 0).sum() < 20:
        return float('nan')
    oof = np.zeros(len(yy))
    for a, b in GroupKFold(5).split(X, yy, g):
        m = HistGradientBoostingClassifier(max_iter=mi, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
        m.fit(X[a], yy[a]); oof[b] = m.predict_proba(X[b])[:, 1]
    return roc_auc_score(yy, oof)


print('\n[1] UNIVARIATE giveaway AUC (single feature alone; |AUC| away from 0.5 = separation):')
uni = []
for f in FEATS:
    a = roc_auc_score(y, df[f].values)
    uni.append((f, max(a, 1 - a), a))
for f, m, a in sorted(uni, key=lambda t: -t[1])[:12]:
    print(f'  {f:<16s} |AUC|={m:.3f} (raw {a:.3f})')

print('\n[2] DUP/overlap check:')
print(f'  unique (lon,lat) pairs: {df[["lon","lat"]].drop_duplicates().shape[0]} of {len(df)}')
posset = set(map(tuple, df[y == 1][['lon', 'lat']].round(4).values))
negset = set(map(tuple, df[y == 0][['lon', 'lat']].round(4).values))
print(f'  pos/neg location overlap (rounded 4dp): {len(posset & negset)}')

print('\n[3] SPATIAL-LEAKAGE robustness (full features, vary holdout block size):')
for deg in [0.25, 0.5, 1.0, 2.0]:
    print(f'  {deg:>4}deg cells: AUC={cv(df, FEATS, deg):.4f}')

print('\n[4] REGIONAL holdout (train one half of SE-US, test the other):')
med = df.lon.median()
west, east = df[df.lon < med].copy(), df[df.lon >= med].copy()
for nm, tr, te in [('train WEST -> test EAST', west, east), ('train EAST -> test WEST', east, west)]:
    Xtr = np.nan_to_num(tr[FEATS].values.astype('float32')); Xte = np.nan_to_num(te[FEATS].values.astype('float32'))
    m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
    m.fit(Xtr, tr.label.astype(int).values)
    print(f'  {nm}: AUC={roc_auc_score(te.label.astype(int).values, m.predict_proba(Xte)[:,1]):.4f}')

print('\n[5] POPULATION-SKEW: distribution + remote-only subset:')
print(f'  Pop_Density>0:  fire={ (df[y==1].Pop_Density>0).mean():.2%}  nofire={ (df[y==0].Pop_Density>0).mean():.2%}')
remote = df[df.Pop_Density == 0].copy()
print(f'  remote-only (Pop==0): n={len(remote)} fire={int(remote.label.sum())} -> AUC={cv(remote, FEATS):.4f}')
print(f'  remote-only, TEMPORAL features only -> AUC={cv(remote, TEMPORAL):.4f}')

print('\n[6] WEATHER/SEASON-MATCHED hard subset (keep only negatives whose conditions resemble positives):')
posdf = df[y == 1]
mask = pd.Series(True, index=df.index)
for f in ['tmmx_30', 'vpd_30', 'fm100_90', 'pr_90']:
    lo, hi = posdf[f].quantile(0.10), posdf[f].quantile(0.90)
    keep_neg = (df[f] >= lo) & (df[f] <= hi)
    mask &= (y == 1) | keep_neg
hard = df[mask].copy()
print(f'  matched subset: n={len(hard)} fire={int(hard.label.sum())} nofire={int((hard.label==0).sum())}')
print(f'  AUC (full features) on weather-matched negatives = {cv(hard, FEATS):.4f}')
print(f'  AUC (TEMPORAL only) on weather-matched negatives  = {cv(hard, TEMPORAL):.4f}')
print(f'  AUC (SPATIAL only)  on weather-matched negatives  = {cv(hard, SPATIAL):.4f}')

print('\n[7] BASE-RATE reality (AUC != accuracy):')
print('  Our eval prevalence: {:.1%} fire. Real per-cell-day ignition prevalence ~1e-4 to 1e-6.'.format(y.mean()))
print('  At AUC~0.9 + realistic low prevalence, precision at useful recall is low (many false alarms):')
print('  the model RANKS risk well but is NOT "93% correct" operationally.')
