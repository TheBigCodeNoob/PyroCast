"""Florida model: trains on the FL-DENSE data (10k FL fires) as a variance-reduced ENSEMBLE
(HGB + LightGBM + RandomForest). Reports whether more FL data + ensembling beats the SE-wide
0.786, and saves best_model_fl_ensemble.joblib for the web demo."""
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
    out = []
    for c in glob.glob(g):
        try:
            out.append(pd.read_csv(c))
        except Exception:
            pass
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


df = _safe('Training Data Florida/FL/*.csv').dropna(subset=['lon', 'lat', 'label']).reset_index(drop=True)
if len(df) == 0:
    print('no FL data yet'); raise SystemExit
df['et_stress'] = df.et / (df.pet + 1)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
DROP = ['smap_surface', 'lst_night', 'et']
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
FEATS = [c for c in df.columns if c not in META + DROP]
y = df.label.astype(int).values
print(f'FL-dense rows={len(df)} fire={int(y.sum())} feats={len(FEATS)}')
block = (np.floor(df.lon / 0.6).astype(int).astype(str) + '_' + np.floor(df.lat / 0.6).astype(int).astype(str)).values
yr = df.year.values; X = np.nan_to_num(df[FEATS].values.astype('float32'))


def members():
    m = [('hgb', HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)),
         ('rf', RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0))]
    if HAVE_LGB:
        m.append(('lgb', LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1)))
    return m


names = [n for n, _ in members()]
oof = {n: np.full(len(y), np.nan) for n in names}
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20 or (y[tr] == 0).sum() < 20:
        continue
    for n, mdl in members():
        mdl.fit(X[tr], y[tr]); oof[n][te] = mdl.predict_proba(X[te])[:, 1]
mask = ~np.isnan(oof['hgb']); yt = y[mask]
ens = np.mean([oof[n][mask] for n in names], axis=0)
order = np.argsort(ens)[::-1]; p5 = yt[order[:max(1, int(len(ens) * .05))]].mean()
print('\nFL-DENSE (blocked space+time within FL):')
print(f'  single HGB   AUC {roc_auc_score(yt, oof["hgb"][mask]):.4f}')
print(f'  ENSEMBLE     AUC {roc_auc_score(yt, ens):.4f} | top5% prec {p5:.2f}')
print(f'  vs SE-wide-in-FL: 0.786 / top5% 0.90')

# train ensemble on ALL FL data, save for the demo
fitted = []
for n, mdl in members():
    mdl.fit(X, y); fitted.append(mdl)
joblib.dump({'models': fitted, 'features': FEATS, 'handles_missing': False, 'region': 'Florida',
             'honest_metrics': {'auc': round(float(roc_auc_score(yt, ens)), 4), 'top5pct_precision': round(float(p5), 3)},
             'trained_on': f'{len(df)} FL rows (ensemble: {"+".join(names)})'}, 'best_model_fl_ensemble.joblib')
print(f'\nSaved best_model_fl_ensemble.joblib ({len(df)} FL rows, ensemble AUC {roc_auc_score(yt, ens):.4f})')
