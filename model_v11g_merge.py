"""v11g eval: does vertical fuel structure (canopy height + tree cover) help the full
model AND the ENV floor? Canopy structure can't be a reporting crutch."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

d10 = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna()
dc = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11c_extra/*.csv')], ignore_index=True).dropna()
dd = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11d_extra/*.csv')], ignore_index=True).dropna()
dg = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11g_extra/*.csv')], ignore_index=True).dropna()
V11C = ['NightLights', 'nbhd_dev_500m', 'nbhd_forest_2km', 'nbhd_wetland_2km']
V11D = ['nbhd_crop_2km', 'nbhd_wetland_5km', 'nbhd_pasture_2km']
V11G = ['canopy_ht', 'canopy_ht_2km', 'treecover', 'treecover_2km']
for d in (d10, dc, dd, dg):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
df = (d10.merge(dc[['k'] + V11C].drop_duplicates('k'), on='k')
          .merge(dd[['k'] + V11D].drop_duplicates('k'), on='k')
          .merge(dg[['k'] + V11G].drop_duplicates('k'), on='k')).reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
ALL = [c for c in df.columns if c not in META]
HUMAN = ['Pop_Density', 'DistDev', 'LC_Developed', 'NightLights', 'nbhd_dev_500m']
LANDUSE = ['LC_Crop', 'LC_Pasture', 'nbhd_crop_2km', 'nbhd_pasture_2km']
ENV = [c for c in ALL if c not in HUMAN + LANDUSE]
PREV = [c for c in ALL if c not in V11G]
ENV_NOG = [c for c in ENV if c not in V11G]
print(f'merged={len(df)}')


def st(d, cols):
    yy = d.label.astype(int).values
    bl = (np.floor(d.lon).astype(int).astype(str) + '_' + np.floor(d.lat).astype(int).astype(str)).values
    yrr = d.year.values; X = np.nan_to_num(d[cols].values.astype('float32')); oof = np.full(len(yy), np.nan)
    for tg, eg in GroupKFold(5).split(X, yy, bl):
        tr = tg[yrr[tg] <= 2019]; te = eg[yrr[eg] >= 2020]
        if len(tr) < 100 or yy[tr].sum() < 20 or (yy[tr] == 0).sum() < 20:
            continue
        m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
        m.fit(X[tr], yy[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof); return roc_auc_score(yy[mask], oof[mask])


def match(d, col):
    nz = d[col][d[col] > 0]
    edges = np.unique([d[col].min() - 1, 1e-9] + list(nz.quantile([.2, .4, .6, .8]).values) + [np.inf]) if len(nz) else np.array([-1, np.inf])
    d = d.copy(); d['pb'] = pd.cut(d[col], bins=edges, duplicates='drop')
    posf = d[d.label == 1].pb.value_counts(normalize=True); negc = d[d.label == 0].pb.value_counts()
    N = int(min(negc.get(b, 0) / posf[b] for b in posf.index if posf[b] > 0)); keep = [d[d.label == 1]]
    for b in posf.index:
        pool = d[(d.label == 0) & (d.pb == b)]; kk = int(round(posf[b] * N))
        if len(pool) and kk:
            keep.append(pool.sample(min(kk, len(pool)), random_state=0))
    return pd.concat(keep).reset_index(drop=True)


print('\nLOCKED metric (full model):')
print(f'  prev (base+v11c+v11d)   {st(df, PREV):.4f}')
print(f'  + v11g canopy/treecover {st(df, ALL):.4f}')
print('\nENVIRONMENTAL FLOOR (human-access stripped):')
print(f'  env-only (no v11g)      {st(df, ENV_NOG):.4f}')
print(f'  env-only + v11g         {st(df, ENV):.4f}')
print('\nstricter crutch metric (match nbhd_dev_500m):')
print(f'  prev {st(match(df,"nbhd_dev_500m"), PREV):.4f} -> +v11g {st(match(df,"nbhd_dev_500m"), ALL):.4f}')
print(f'pop-matched: prev {st(match(df,"Pop_Density"), PREV):.4f} -> +v11g {st(match(df,"Pop_Density"), ALL):.4f}')
y = df.label.astype(int).values; trm = (df.lon < df.lon.median()).values
X = np.nan_to_num(df[ALL].values.astype('float32'))
m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(X[trm], y[trm])
pi = permutation_importance(m, X[~trm], y[~trm], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
print('\n  v11g importance:')
for i in np.argsort(pi.importances_mean)[::-1]:
    if ALL[i] in V11G:
        print(f'    {ALL[i]:<16s} {pi.importances_mean[i]:+.4f}')
