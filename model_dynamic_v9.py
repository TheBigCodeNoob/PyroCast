"""
v9 eval: do the new dynamic fire-danger features add GENUINE skill (pop+season-matched)?
Merges v9 features onto the exact v7 samples, then compares v7-features vs v7+v9-features
on the honest, population-matched triple. Reports new-feature importances.
"""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

v7 = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v7/*.csv')], ignore_index=True).dropna()
v9 = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v9dyn/*.csv')], ignore_index=True).dropna()
print(f'v7 rows={len(v7)}  v9 rows={len(v9)}')


def key(d):
    return d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str) + '_' + d.label.astype(int).astype(str)


v7['k'] = key(v7); v9['k'] = key(v9)
NEW = ['bi_30', 'bi_90', 'fm1000_30', 'fm1000_90', 'vs_7', 'vs_30', 'dry_days_14', 'dry_days_30', 'pr_270', 'pr_540', 'pdsi_270', 'pdsi_365', 'vpd_max_30', 'tmmx_max_7']
df = v7.merge(v9[['k'] + NEW], on='k', how='inner').reset_index(drop=True)
print(f'merged rows={len(df)} (of v7 {len(v7)})')

eps = 0.1
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
# new engineered
df['fm1000_trend'] = df.fm1000_30 - df.fm1000_90
df['pr_long_deficit'] = df.pr_540 / 2 - df.pr_270
df['pdsi_long_traj'] = df.pdsi_0 - df.pdsi_365
META = ['lon', 'lat', 'label', 'month', 'year', 'doy', 'k']
V7F = [c for c in v7.columns if c not in META]
ALLF = V7F + NEW + ['fm1000_trend', 'pr_long_deficit', 'pdsi_long_traj']


def popmatch(d):
    nz = d.Pop_Density[d.Pop_Density > 0]
    edges = np.unique([-1, 1e-9] + list(nz.quantile([0.25, 0.5, 0.75]).values) + [np.inf])
    d = d.copy(); d['pb'] = pd.cut(d.Pop_Density, bins=edges)
    posf = d[d.label == 1].pb.value_counts(normalize=True); negc = d[d.label == 0].pb.value_counts()
    N = int(min(negc.get(b, 0) / posf[b] for b in posf.index if posf[b] > 0))
    keep = [d[(d.label == 1)]]
    for b in posf.index:
        pool = d[(d.label == 0) & (d.pb == b)]; k = int(round(posf[b] * N))
        if len(pool) and k:
            keep.append(pool.sample(min(k, len(pool)), random_state=0))
    return pd.concat(keep).reset_index(drop=True)


def triple(d, cols, tag):
    y = d.label.astype(int).values
    cells = ((np.round(d.lon * 4) / 4).astype(str) + '_' + (np.round(d.lat * 4) / 4).astype(str)).values
    med = d.lon.median(); X = np.nan_to_num(d[cols].values.astype('float32'))

    def mk():
        return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, cells):
        m = mk(); m.fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]
    sc = roc_auc_score(y, oof)
    tr, te = (d.year <= 2021).values, (d.year >= 2022).values
    m = mk(); m.fit(X[tr], y[tr]); tm = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    tr2 = ((d.year <= 2021) & (d.lon < med)).values; te2 = ((d.year >= 2022) & (d.lon >= med)).values
    m = mk(); m.fit(X[tr2], y[tr2]); st = roc_auc_score(y[te2], m.predict_proba(X[te2])[:, 1])
    print(f'  {tag:<26s} spatialCV={sc:.4f} temporal={tm:.4f} space+time={st:.4f}')
    return sc, tm, st


print('\n=== POP+SEASON-MATCHED honest eval (the genuine, crutch-free metric) ===')
pm = popmatch(df)
print(f'pop-matched: pos={int((pm.label==1).sum())} neg={int((pm.label==0).sum())}')
triple(pm, V7F, 'v7 feats only (baseline)')
new_full = triple(pm, ALLF, 'v7 + NEW dynamic feats')

print('\n=== (reference) as-is, not pop-matched ===')
triple(df, ALLF, 'v7+NEW, not matched')

print('\n=== new-feature permutation importance (pop-matched spatial holdout) ===')
y = pm.label.astype(int).values; med = pm.lon.median()
tr = (pm.lon < med).values
X = np.nan_to_num(pm[ALLF].values.astype('float32'))
m = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(X[tr], y[tr])
pi = permutation_importance(m, X[~tr], y[~tr], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
order = np.argsort(pi.importances_mean)[::-1]
print('top 15 (NEW feats marked *):')
for i in order[:15]:
    star = '*' if ALLF[i] in NEW + ['fm1000_trend', 'pr_long_deficit', 'pdsi_long_traj'] else ' '
    print(f'  {star} {ALLF[i]:<16s} {pi.importances_mean[i]:+.4f}')
