"""Does the Florida-DENSE data (10k FL fires) build a better Florida model than the SE-wide
one (0.786)? Blocked space+time within FL. Saves best_model_fl.joblib if it helps."""
import glob, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


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
# match best_model_v13 feature set (drop the unused moisture cols)
DROP = ['smap_surface', 'lst_night', 'et']
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
FEATS = [c for c in df.columns if c not in META + DROP]
y = df.label.astype(int).values
print(f'FL-dense rows={len(df)} fire={int(y.sum())} feats={len(FEATS)}')
block = (np.floor(df.lon / 0.6).astype(int).astype(str) + '_' + np.floor(df.lat / 0.6).astype(int).astype(str)).values
yr = df.year.values; X = np.nan_to_num(df[FEATS].values.astype('float32'))


def MK():
    return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


oof = np.full(len(y), np.nan)
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20 or (y[tr] == 0).sum() < 20:
        continue
    m = MK(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
mask = ~np.isnan(oof); yt, pt = y[mask], oof[mask]
auc = roc_auc_score(yt, pt); order = np.argsort(pt)[::-1]
p5 = yt[order[:max(1, int(len(pt) * .05))]].mean(); p10 = yt[order[:max(1, int(len(pt) * .1))]].mean()
print(f'\nFL-DENSE model (blocked space+time within FL):')
print(f'  AUC {auc:.4f} | top5% prec {p5:.2f} | top10% prec {p10:.2f} | (n={mask.sum()})')
print(f'  vs SE-wide model in FL: 0.786 / top5% 0.90')

# save the FL model trained on all FL data
final = MK().fit(X, y)
joblib.dump({'model': final, 'features': FEATS, 'handles_missing': True, 'region': 'Florida',
             'honest_metrics': {'auc': round(auc, 4), 'top5pct_precision': round(float(p5), 3)},
             'trained_on': f'{len(df)} FL rows'}, 'best_model_fl.joblib')
print(f'\nSaved best_model_fl.joblib ({len(df)} FL rows, AUC {auc:.4f})')
