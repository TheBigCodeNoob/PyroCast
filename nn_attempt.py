"""Is there model-capacity headroom above gradient-boosting? Try a neural net (MLP) on the
final features. Blocked space+time. If it beats ~0.862 GBM / 0.866 ensemble, real gain; if not,
the ~0.86 ceiling is confirmed across architectures (GBM, RF, LGBM, linear, AND deep)."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def _safe(g):
    out = []
    for c in glob.glob(g):
        try:
            out.append(pd.read_csv(c))
        except Exception:
            pass
    return pd.concat(out, ignore_index=True)


ve = _safe('Training Data Florida/v11e/*.csv').dropna()
vh = _safe('Training Data Florida/v11h_canopy/*.csv').dropna()
vm = _safe('Training Data Florida/v12_moisture/*.csv')
v13 = _safe('Training Data Florida/v13_human/*.csv')
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']; MOIST = ['ndmi', 'smap_root', 'lst_day', 'pet']
for d in (ve, vh, vm, v13):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
vm['et_stress'] = vm.et / (vm.pet + 1)
df = (ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k')
        .merge(vm[['k'] + MOIST + ['et_stress']].drop_duplicates('k'), on='k', how='left')
        .merge(v13[['k', 'built']].dropna().drop_duplicates('k'), on='k', how='left')).reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
FEATS = [c for c in df.columns if c not in ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']]
y = df.label.astype(int).values
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values; Xr = df[FEATS].values.astype('float32')

oof_nn = np.full(len(y), np.nan); oof_gbm = np.full(len(y), np.nan)
for tg, eg in GroupKFold(5).split(Xr, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20:
        continue
    imp = SimpleImputer(strategy='median'); sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(Xr[tr])); Xte = sc.transform(imp.transform(Xr[te]))
    nn = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                       early_stopping=True, n_iter_no_change=8, max_iter=200, random_state=0)
    nn.fit(Xtr, y[tr]); oof_nn[te] = nn.predict_proba(Xte)[:, 1]
    g = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
    g.fit(Xr[tr], y[tr]); oof_gbm[te] = g.predict_proba(Xr[te])[:, 1]
m = ~np.isnan(oof_nn)
nn_auc = roc_auc_score(y[m], oof_nn[m]); gbm_auc = roc_auc_score(y[m], oof_gbm[m])
# blended (rank-average GBM + NN)
blend = pd.Series(oof_gbm[m]).rank().values + pd.Series(oof_nn[m]).rank().values
print(f'neural net (MLP 128-64): {nn_auc:.4f}')
print(f'gradient boosting:       {gbm_auc:.4f}')
print(f'GBM + NN blend:          {roc_auc_score(y[m], blend):.4f}')
print('-> if NN/blend > GBM, there is capacity headroom; if not, ceiling confirmed across architectures.')
