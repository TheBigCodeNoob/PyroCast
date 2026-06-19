"""
v11c eval: do the new WHERE features (nightlights + neighborhood land-cover context)
raise the LOCKED metric (blocked space+time)? Merges new feats onto the exact v10b
points by (lon,lat), then A/B tests base vs base+new.
"""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

d10 = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna()
dx = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11c_extra/*.csv')], ignore_index=True).dropna()
NEWF = ['NightLights', 'nbhd_dev_2km', 'nbhd_dev_500m', 'nbhd_forest_2km', 'nbhd_wetland_2km']
for d in (d10, dx):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
df = d10.merge(dx[['k'] + NEWF].drop_duplicates('k'), on='k', how='inner').reset_index(drop=True)
print(f'v10b={len(d10)} extra={len(dx)} merged={len(df)}')
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
BASE = [c for c in df.columns if c not in META + NEWF]
y = df.label.astype(int).values
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values


def st(cols, ret_oof=False):
    X = np.nan_to_num(df[cols].values.astype('float32')); oof = np.full(len(y), np.nan)
    for tg, eg in GroupKFold(5).split(X, y, block):
        tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
        if len(tr) < 100 or y[tr].sum() < 20:
            continue
        m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof); a = roc_auc_score(y[mask], oof[mask])
    return (a, oof) if ret_oof else a


base = st(BASE)
print(f'\nLOCKED metric (blocked space+time):')
print(f'  base (v10b feats)            {base:.4f}')
for f in NEWF:
    print(f'  + {f:<18s}{st(BASE+[f]):.4f}')
print(f'  + ALL new                    {st(BASE+NEWF):.4f}')

# permutation importance of the new features within the full model
trm = (df.lon < df.lon.median()).values
X = np.nan_to_num(df[BASE + NEWF].values.astype('float32'))
m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(X[trm], y[trm])
pi = permutation_importance(m, X[~trm], y[~trm], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
cols = BASE + NEWF
print('\n  new-feature importance rank (among all):')
order = np.argsort(pi.importances_mean)[::-1]
for rank, i in enumerate(order):
    if cols[i] in NEWF:
        print(f'    #{rank+1:>2d}/{len(cols)}  {cols[i]:<18s} {pi.importances_mean[i]:+.4f}')
