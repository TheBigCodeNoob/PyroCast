"""
Accurate space+time AUC via leave-spatial-block-out x past->future CV.
For each fold: train = (other spatial blocks) & (year<=2021); test = (held-out blocks) & (year>=2022).
Pool every 2022+ sample as test exactly once -> full future set as test (large, accurate),
with strict new-place-AND-future separation. Re-checks v8 baseline and v7 pop-matched.
"""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def engineer(df):
    df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
    df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
    df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
    return df


def feats(df):
    return [c for c in df.columns if c not in ('lon', 'lat', 'label', 'month', 'year', 'doy', 'pb')]


def popmatch(d):
    nz = d.Pop_Density[d.Pop_Density > 0]
    edges = np.unique([-1, 1e-9] + list(nz.quantile([0.25, 0.5, 0.75]).values) + [np.inf])
    d = d.copy(); d['pb'] = pd.cut(d.Pop_Density, bins=edges)
    posf = d[d.label == 1].pb.value_counts(normalize=True); negc = d[d.label == 0].pb.value_counts()
    N = int(min(negc.get(b, 0) / posf[b] for b in posf.index if posf[b] > 0))
    keep = [d[d.label == 1]]
    for b in posf.index:
        pool = d[(d.label == 0) & (d.pb == b)]; k = int(round(posf[b] * N))
        if len(pool) and k:
            keep.append(pool.sample(min(k, len(pool)), random_state=0))
    return pd.concat(keep).reset_index(drop=True)


def blocked_st_auc(d, cols, block_deg=1.0, k=5):
    y = d.label.astype(int).values
    block = ((np.floor(d.lon / block_deg)).astype(int).astype(str) + '_' + (np.floor(d.lat / block_deg)).astype(int).astype(str)).values
    yr = d.year.values
    X = np.nan_to_num(d[cols].values.astype('float32'))
    oof = np.full(len(y), np.nan)
    nblocks = len(set(block))
    for tr_g, te_g in GroupKFold(min(k, nblocks)).split(X, y, block):
        tr = tr_g[yr[tr_g] <= 2021]; te = te_g[yr[te_g] >= 2022]
        if len(tr) < 100 or len(te) < 20 or y[tr].sum() < 20 or (y[tr] == 0).sum() < 20:
            continue
        m = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof)
    return roc_auc_score(y[mask], oof[mask]), int(mask.sum()), int(y[mask].sum()), nblocks


def load(folder):
    return engineer(pd.concat([pd.read_csv(c) for c in glob.glob(f'Training Data Florida/{folder}/*.csv')], ignore_index=True).dropna().reset_index(drop=True))


print('=== ACCURATE space+time (leave-block-out x past->future, full future set pooled) ===\n')
v8 = load('v8')
for tag, d in [('v8 FIRMS as-is', v8), ('v8 FIRMS pop-matched', popmatch(v8))]:
    auc, n, nf, nb = blocked_st_auc(d, feats(d))
    print(f'  {tag:<24s} space+time AUC = {auc:.4f}   (test n={n}, fire={nf}, blocks={nb})')
v7 = load('v7')
for tag, d in [('v7 MTBS as-is', v7), ('v7 MTBS pop-matched', popmatch(v7))]:
    auc, n, nf, nb = blocked_st_auc(d, feats(d))
    print(f'  {tag:<24s} space+time AUC = {auc:.4f}   (test n={n}, fire={nf}, blocks={nb})')
