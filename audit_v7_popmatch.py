"""
#2: What is MTBS-v7 honestly worth WITHOUT the remoteness/population crutch?
Match the negatives' Pop_Density distribution to the positives' (so population can no
longer separate fire from non-fire), then re-score the honest triple. If it falls to
~0.70 it confirms v7's 0.80 was largely the MTBS remote-large-fire artifact.
"""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v7/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
eps = 0.1
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'month', 'year', 'doy']
FEATS = [c for c in df.columns if c not in META]


def mk():
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def triple(d, cols, tag):
    y = d.label.astype(int).values
    cells = ((np.round(d.lon * 4) / 4).astype(str) + '_' + (np.round(d.lat * 4) / 4).astype(str)).values
    med = d.lon.median()
    X = np.nan_to_num(d[cols].values.astype('float32'))
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, cells):
        m = mk(); m.fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]
    sc = roc_auc_score(y, oof)
    tr, te = (d.year <= 2021).values, (d.year >= 2022).values
    m = mk(); m.fit(X[tr], y[tr]); tm = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    tr2 = ((d.year <= 2021) & (d.lon < med)).values; te2 = ((d.year >= 2022) & (d.lon >= med)).values
    m = mk(); m.fit(X[tr2], y[tr2]); st = roc_auc_score(y[te2], m.predict_proba(X[te2])[:, 1])
    print(f'  {tag:<28s} spatialCV={sc:.4f} temporal={tm:.4f} space+time={st:.4f}')
    return st


print(f'v7 full: pos={int((df.label==1).sum())} neg={int((df.label==0).sum())}')
print('\n=== v7 as-is (with remoteness signal) ===')
triple(df, FEATS, 'v7 FULL')

# ---- Population-match negatives to positives ----
pos, neg = df[df.label == 1], df[df.label == 0]
# bins: 0 separately, then quantiles of >0 population
nz = df.Pop_Density[df.Pop_Density > 0]
edges = [-1, 1e-9] + list(nz.quantile([0.25, 0.5, 0.75]).values) + [np.inf]
df['popbin'] = pd.cut(df.Pop_Density, bins=np.unique(edges))
posf = df[df.label == 1].popbin.value_counts(normalize=True)
negc = df[df.label == 0].popbin.value_counts()
# largest N s.t. round(posf[b]*N) <= negc[b] for all bins with pos
N = int(min(negc.get(b, 0) / posf[b] for b in posf.index if posf[b] > 0))
keep = []
rng = np.random.default_rng(0)
for b in posf.index:
    k = int(round(posf[b] * N))
    pool = df[(df.label == 0) & (df.popbin == b)]
    if len(pool) and k:
        keep.append(pool.sample(min(k, len(pool)), random_state=0))
neg_matched = pd.concat(keep) if keep else neg.iloc[:0]
matched = pd.concat([pos, neg_matched]).reset_index(drop=True)
print(f'\nAfter pop-matching: pos={len(pos)} neg={len(neg_matched)} (neg Pop dist now matches pos)')
print(f'  Pop>0 fraction: pos={ (pos.Pop_Density>0).mean():.2%} neg_matched={ (neg_matched.Pop_Density>0).mean():.2%}')
print('\n=== v7 with population MATCHED (remoteness crutch removed) ===')
triple(matched, FEATS, 'v7 pop-matched FULL')
triple(matched, [c for c in FEATS if c not in ('Pop_Density',)], 'v7 pop-matched, no Pop feat')
