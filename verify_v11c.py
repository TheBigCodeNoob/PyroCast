"""Is the v11c +0.017 gain HONEST or a remoteness crutch? Re-run base vs base+new under
the same crutch controls used to validate v10b: as-is, pop-matched, DistDev-matched.
If base+new beats base under matching too, the gain is real where-signal (fuel/context),
not just 'fires are reported near development'."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

d10 = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna()
dx = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11c_extra/*.csv')], ignore_index=True).dropna()
NEWF = ['NightLights', 'nbhd_dev_500m', 'nbhd_forest_2km', 'nbhd_wetland_2km']  # drop dead nbhd_dev_2km
for d in (d10, dx):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
df0 = d10.merge(dx[['k'] + NEWF].drop_duplicates('k'), on='k', how='inner').reset_index(drop=True)
df0['pdsi_traj_90'] = df0.pdsi_0 - df0.pdsi_90; df0['vpd_trend'] = df0.vpd_7 - df0.vpd_90
df0['pr_deficit'] = df0.pr_365 / 4 - df0.pr_90; df0['fm100_trend'] = df0.fm100_30 - df0.fm100_90
df0['dryness'] = df0.vpd_30 + df0.erc_30 - df0.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
BASE = [c for c in df0.columns if c not in META + NEWF]


def st(df, cols):
    y = df.label.astype(int).values
    block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
    yr = df.year.values; X = np.nan_to_num(df[cols].values.astype('float32')); oof = np.full(len(y), np.nan)
    for tg, eg in GroupKFold(5).split(X, y, block):
        tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
        if len(tr) < 100 or y[tr].sum() < 20:
            continue
        m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof); return roc_auc_score(y[mask], oof[mask])


def match(d, col):
    nz = d[col][d[col] > 0]
    edges = np.unique([d[col].min() - 1, 1e-9] + list(nz.quantile([.25, .5, .75]).values) + [np.inf]) if len(nz) else np.array([-1, np.inf])
    d = d.copy(); d['pb'] = pd.cut(d[col], bins=edges, duplicates='drop')
    posf = d[d.label == 1].pb.value_counts(normalize=True)
    negc = d[d.label == 0].pb.value_counts()
    N = int(min(negc.get(b, 0) / posf[b] for b in posf.index if posf[b] > 0))
    keep = [d[d.label == 1]]
    for b in posf.index:
        pool = d[(d.label == 0) & (d.pb == b)]; kk = int(round(posf[b] * N))
        if len(pool) and kk:
            keep.append(pool.sample(min(kk, len(pool)), random_state=0))
    return pd.concat(keep).reset_index(drop=True)


print(f'merged={len(df0)}  (new feats: {NEWF})')
print(f'\n{"control":<22s}{"base":>8s}{"base+new":>10s}{"delta":>8s}')
for name, d in [('as-is', df0), ('pop-matched', match(df0, 'Pop_Density')), ('DistDev-matched', match(df0, 'DistDev'))]:
    b = st(d, BASE); n = st(d, BASE + NEWF)
    print(f'{name:<22s}{b:>8.4f}{n:>10.4f}{n-b:>+8.4f}')
print('\n-> if delta stays POSITIVE under matching, the gain is real fuel/context signal, not remoteness.')
