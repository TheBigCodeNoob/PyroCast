"""Does post-processing reduce false positives/negatives in Florida? Tests (1) spatial
smoothing of predictions and (2) a model ensemble, on held-out FL data."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
try:
    from lightgbm import LGBMClassifier
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False


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
FL = (df.lat < 31.0) & (df.lon > -87.6) & (df.lon < -79.8)
FEATS = [c for c in df.columns if c not in ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']]
y = df.label.astype(int).values
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values; X = df[FEATS].values.astype('float32')


def mdls():
    m = {'hgb': HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0),
         'rf': RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0),
         'et': ExtraTreesClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0)}
    if HAVE_LGB:
        m['lgb'] = LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1)
    return m


names = list(mdls().keys())
oofs = {n: np.full(len(y), np.nan) for n in names}
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20:
        continue
    Xtr = np.nan_to_num(X[tr]); Xte = np.nan_to_num(X[te])
    for n, m in mdls().items():
        m.fit(Xtr, y[tr]); oofs[n][te] = m.predict_proba(Xte)[:, 1]
m = (~np.isnan(oofs['hgb'])) & FL.values
yt = y[m]; lon = df.lon.values[m]; lat = df.lat.values[m]
hgb = oofs['hgb'][m]
ens = np.mean([pd.Series(oofs[n][m]).rank().values for n in names], axis=0)


def smooth(p, radius_km=8, blend=0.5):
    deg = radius_km / 111.0
    out = p.copy()
    for i in range(len(p)):
        d2 = (lon - lon[i]) ** 2 + (lat - lat[i]) ** 2
        w = np.exp(-d2 / (2 * (deg / 2) ** 2)); w[i] = 0
        if w.sum() > 0:
            nb = (w * p).sum() / w.sum()
            out[i] = (1 - blend) * p[i] + blend * nb
    return out


print(f'Florida held-out test points: {m.sum()} (fires {yt.sum()})')
print(f'  HGB (current)            AUC {roc_auc_score(yt, hgb):.4f}  Brier {brier_score_loss(yt, hgb/hgb.max() if hgb.max()>1 else hgb):.3f}')
print(f'  ensemble (4 models)      AUC {roc_auc_score(yt, ens):.4f}')
for r in [5, 8, 12]:
    sm = smooth(hgb, radius_km=r, blend=0.4)
    print(f'  HGB + spatial smooth {r}km AUC {roc_auc_score(yt, sm):.4f}')
sm_ens = smooth(ens, radius_km=8, blend=0.4)
print(f'  ensemble + smooth 8km    AUC {roc_auc_score(yt, sm_ens):.4f}')
