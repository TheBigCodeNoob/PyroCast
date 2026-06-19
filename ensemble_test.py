"""Quick CPU-only check: does averaging diverse model families beat the single HGB on the
final data? Blocked space+time."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
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
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
for d in (ve, vh):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
df = ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k').reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values; X = np.nan_to_num(df[FEATS].values.astype('float32'))


def models():
    m = {'hgb': HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0),
         'rf': RandomForestClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0),
         'et': ExtraTreesClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0)}
    if HAVE_LGB:
        m['lgb'] = LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1)
    return m


names = list(models().keys())
oofs = {n: np.full(len(y), np.nan) for n in names}
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20:
        continue
    for n, mdl in models().items():
        mdl.fit(X[tr], y[tr]); oofs[n][te] = mdl.predict_proba(X[te])[:, 1]
mask = ~np.isnan(oofs['hgb']); yt = y[mask]


def rank(o):
    return pd.Series(o[mask]).rank().values


print('single models (blocked space+time):')
for n in names:
    print(f'  {n:<5s} {roc_auc_score(yt, oofs[n][mask]):.4f}')
ens = np.mean([rank(oofs[n]) for n in names], axis=0)
print(f'\n  rank-average ensemble ({"+".join(names)}): {roc_auc_score(yt, ens):.4f}')
ens2 = np.mean([rank(oofs[n]) for n in ['hgb', 'lgb'] if n in oofs], axis=0)
print(f'  hgb+lgb only: {roc_auc_score(yt, ens2):.4f}')
print(f'  hgb alone:    {roc_auc_score(yt, oofs["hgb"][mask]):.4f}  <- current')
