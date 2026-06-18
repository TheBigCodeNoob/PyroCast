"""
v10 eval: real FPA-FOD wildfire ignitions. Judged ONLY on the accurate blocked
space+time CV (FPA-FOD is 2017-2020 -> train years<=2019, test 2020). Checks:
- full vs baseline 0.716 (FIRMS)
- pop-matched + human-access-matched (so DistDev/Pop don't become a new crutch)
- ablation (human-access / weather / spatial / full)
- univariate giveaways
- human vs lightning predictability (cause column)
"""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
eps = 0.1
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'pb', 'hb']
HUMAN = ['Pop_Density', 'DistDev', 'LC_Developed']
SPATIAL = ['Elevation', 'Pop_Density', 'DistDev', 'LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Pasture', 'LC_Wetland', 'LC_Crop', 'LC_Developed', 'NDVI', 'EVI']
TEMPORAL = [c for c in df.columns if c not in META + SPATIAL]
FEATS = SPATIAL + TEMPORAL
print(f'rows={len(df)} fire={int((df.label==1).sum())} (human={int((df.cause==1).sum())}, lightning={int((df.cause==0).sum())}) nofire={int((df.label==0).sum())}')
print(f'season mean month: fire={df[df.label==1].month.mean():.2f} nofire={df[df.label==0].month.mean():.2f}')


def MK():
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def blocked_st(d, cols, deg=1.0, k=5, pos_mask=None):
    """train years<=2019 (other blocks); test year 2020 (held-out blocks). pos_mask optionally restricts which positives count as test."""
    y = d.label.astype(int).values
    block = (np.floor(d.lon / deg).astype(int).astype(str) + '_' + np.floor(d.lat / deg).astype(int).astype(str)).values
    yr = d.year.values; X = np.nan_to_num(d[cols].values.astype('float32')); oof = np.full(len(y), np.nan)
    for tg, eg in GroupKFold(min(k, len(set(block)))).split(X, y, block):
        tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
        if len(tr) < 100 or y[tr].sum() < 20 or (y[tr] == 0).sum() < 20:
            continue
        m = MK(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof)
    if pos_mask is not None:  # keep all negatives + only selected positives as test
        mask = mask & ((y == 0) | pos_mask)
    return roc_auc_score(y[mask], oof[mask]), int(mask.sum()), int(y[mask].sum())


def match(d, col, bins=5):
    nz = d[col][d[col] > 0]
    edges = np.unique([d[col].min() - 1] + list(nz.quantile(np.linspace(0, 1, bins + 1)[1:-1]).values) + [np.inf]) if len(nz) else np.array([-1, np.inf])
    d = d.copy(); d['hb'] = pd.cut(d[col], bins=edges, duplicates='drop')
    posf = d[d.label == 1].hb.value_counts(normalize=True); negc = d[d.label == 0].hb.value_counts()
    N = int(min(negc.get(b, 0) / posf[b] for b in posf.index if posf[b] > 0))
    keep = [d[d.label == 1]]
    for b in posf.index:
        pool = d[(d.label == 0) & (d.hb == b)]; kk = int(round(posf[b] * N))
        if len(pool) and kk:
            keep.append(pool.sample(min(kk, len(pool)), random_state=0))
    return pd.concat(keep).reset_index(drop=True)


print('\n=== blocked space+time (vs FIRMS baseline 0.716) ===')
a, n, nf = blocked_st(df, FEATS); print(f'  FULL as-is              {a:.4f}  (test n={n}, fire={nf})')
pm = match(df, 'Pop_Density'); a, n, nf = blocked_st(pm, FEATS); print(f'  pop-matched             {a:.4f}  (test n={n})')
dm = match(df, 'DistDev'); a, n, nf = blocked_st(dm, FEATS); print(f'  DistDev-matched         {a:.4f}  (test n={n})')

print('\n=== ablation (as-is) ===')
for nm, cols in [('HUMAN-access only', HUMAN), ('TEMPORAL only', TEMPORAL), ('SPATIAL only', SPATIAL), ('NO human-access', [c for c in FEATS if c not in HUMAN])]:
    a, n, _ = blocked_st(df, cols); print(f'  {nm:<18s} {a:.4f}')

print('\n=== univariate giveaways (top 10) ===')
for f, a in sorted([(c, max(roc_auc_score(df.label, df[c]), 1 - roc_auc_score(df.label, df[c]))) for c in FEATS], key=lambda t: -t[1])[:10]:
    print(f'  {f:<14s} |AUC|={a:.3f}')

print('\n=== human vs lightning predictability (full model, blocked space+time) ===')
a, n, nf = blocked_st(df, FEATS, pos_mask=(df.cause == 1).values); print(f'  HUMAN ignitions    {a:.4f} (test fire={nf})')
a, n, nf = blocked_st(df, FEATS, pos_mask=(df.cause == 0).values); print(f'  LIGHTNING ignitions {a:.4f} (test fire={nf})')
