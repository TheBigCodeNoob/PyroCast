"""v11d eval: merge v10b + v11c + v11d features; does round-2 context help, and is it
honest (crutch-matched)? Reports locked metric stacking + matched controls + v11d importance."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

d10 = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna()
dc = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11c_extra/*.csv')], ignore_index=True).dropna()
dd = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11d_extra/*.csv')], ignore_index=True).dropna()
V11C = ['NightLights', 'nbhd_dev_500m', 'nbhd_forest_2km', 'nbhd_wetland_2km']
V11D = ['nbhd_dev_1km', 'nbhd_forest_1km', 'nbhd_forest_5km', 'nbhd_wetland_5km', 'nbhd_grass_2km', 'nbhd_crop_2km', 'nbhd_shrub_2km', 'nbhd_pasture_2km', 'dist_water']
for d in (d10, dc, dd):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
df = d10.merge(dc[['k'] + V11C].drop_duplicates('k'), on='k').merge(dd[['k'] + V11D].drop_duplicates('k'), on='k').reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
BASE = [c for c in df.columns if c not in META + V11C + V11D]
print(f'merged={len(df)}')


def st(d, cols):
    y = d.label.astype(int).values
    block = (np.floor(d.lon).astype(int).astype(str) + '_' + np.floor(d.lat).astype(int).astype(str)).values
    yr = d.year.values; X = np.nan_to_num(d[cols].values.astype('float32')); oof = np.full(len(y), np.nan)
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
    posf = d[d.label == 1].pb.value_counts(normalize=True); negc = d[d.label == 0].pb.value_counts()
    N = int(min(negc.get(b, 0) / posf[b] for b in posf.index if posf[b] > 0)); keep = [d[d.label == 1]]
    for b in posf.index:
        pool = d[(d.label == 0) & (d.pb == b)]; kk = int(round(posf[b] * N))
        if len(pool) and kk:
            keep.append(pool.sample(min(kk, len(pool)), random_state=0))
    return pd.concat(keep).reset_index(drop=True)


print('\nLOCKED metric stacking:')
print(f'  base               {st(df, BASE):.4f}')
print(f'  base+v11c          {st(df, BASE+V11C):.4f}')
print(f'  base+v11c+v11d     {st(df, BASE+V11C+V11D):.4f}')
print('\ncrutch controls (base+v11c  vs  base+v11c+v11d):')
for name, d in [('as-is', df), ('pop-matched', match(df, 'Pop_Density')), ('DistDev-matched', match(df, 'DistDev'))]:
    a = st(d, BASE + V11C); b = st(d, BASE + V11C + V11D); print(f'  {name:<16s}{a:.4f} -> {b:.4f}  ({b-a:+.4f})')
trm = (df.lon < df.lon.median()).values
cols = BASE + V11C + V11D; X = np.nan_to_num(df[cols].values.astype('float32'))
y = df.label.astype(int).values
m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(X[trm], y[trm])
pi = permutation_importance(m, X[~trm], y[~trm], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
print('\n  v11d feature importance:')
for i in np.argsort(pi.importances_mean)[::-1]:
    if cols[i] in V11D:
        print(f'    {cols[i]:<18s} {pi.importances_mean[i]:+.4f}')
