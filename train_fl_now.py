"""Train the Florida ensemble on the EXISTING merged FL data (no waiting on the dense export)
so the web demo gets the variance-reduced model now. The dense-data version retrains later."""
import glob, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
try:
    from lightgbm import LGBMClassifier
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False


def _safe(g):
    out = [pd.read_csv(c) for c in glob.glob(g)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


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
        .merge(v13[['k', 'built']].dropna().drop_duplicates('k'), on='k', how='left'))
df = df[(df.lat < 31.0) & (df.lon > -87.6) & (df.lon < -79.8)].reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
print(f'FL rows={len(df)} fire={int(y.sum())}')
block = (np.floor(df.lon / 0.6).astype(int).astype(str) + '_' + np.floor(df.lat / 0.6).astype(int).astype(str)).values
yr = df.year.values; X = np.nan_to_num(df[FEATS].values.astype('float32'))


def members():
    m = [('hgb', HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)),
         ('rf', RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0))]
    if HAVE_LGB:
        m.append(('lgb', LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1)))
    return m


names = [n for n, _ in members()]; oof = {n: np.full(len(y), np.nan) for n in names}
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20 or (y[tr] == 0).sum() < 20:
        continue
    for n, mdl in members():
        mdl.fit(X[tr], y[tr]); oof[n][te] = mdl.predict_proba(X[te])[:, 1]
mask = ~np.isnan(oof['hgb']); yt = y[mask]; ens = np.mean([oof[n][mask] for n in names], axis=0)
order = np.argsort(ens)[::-1]; p5 = yt[order[:max(1, int(len(ens) * .05))]].mean()
print(f'  single HGB AUC {roc_auc_score(yt, oof["hgb"][mask]):.4f} | ENSEMBLE AUC {roc_auc_score(yt, ens):.4f} | top5% prec {p5:.2f}')
fitted = [mdl.fit(X, y) or mdl for _, mdl in members()]
joblib.dump({'models': fitted, 'features': FEATS, 'region': 'Florida',
             'honest_metrics': {'auc': round(float(roc_auc_score(yt, ens)), 4), 'top5pct_precision': round(float(p5), 3)},
             'trained_on': f'{len(df)} FL rows (ensemble {"+".join(names)})'}, 'best_model_fl_ensemble.joblib')
print(f'Saved best_model_fl_ensemble.joblib (ensemble AUC {roc_auc_score(yt, ens):.4f})')
