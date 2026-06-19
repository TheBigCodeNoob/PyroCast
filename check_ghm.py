"""Is the +0.040 gHM gain REAL or a reporting-bias crutch? gHM is human-modification
(roads/power/built/ag). If it predicts HUMAN fires (which people cause) but NOT lightning
fires (which people don't), it's causal. If it predicts BOTH, it's reporting bias. Also
check the gHM distributions and whether negatives are unfairly low-gHM."""
import glob, warnings, numpy as np, pandas as pd
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


ve = _safe('Training Data Florida/v11e/*.csv').dropna()
v13 = _safe('Training Data Florida/v13_human/*.csv')
for d in (ve, v13):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
df = ve.merge(v13[['k', 'ghm', 'ghm_2km', 'built']].drop_duplicates('k'), on='k', how='left').reset_index(drop=True)
y = df.label.astype(int).values

print('=== gHM distributions (0=wild, 1=fully human-modified) ===')
print(f'  human-caused fires:   median gHM {df[df.cause==1].ghm.median():.3f}  (p25 {df[df.cause==1].ghm.quantile(.25):.3f}, p75 {df[df.cause==1].ghm.quantile(.75):.3f})')
print(f'  lightning fires:      median gHM {df[df.cause==0].ghm.median():.3f}  (p25 {df[df.cause==0].ghm.quantile(.25):.3f}, p75 {df[df.cause==0].ghm.quantile(.75):.3f})')
print(f'  random-land negs:     median gHM {df[df.label==0].ghm.median():.3f}  (p25 {df[df.label==0].ghm.quantile(.25):.3f}, p75 {df[df.label==0].ghm.quantile(.75):.3f})')

# univariate gHM AUC: human-fires vs negs, and lightning-fires vs negs
def uni_auc(possub):
    m = possub | (df.label == 0).values
    s = df.ghm.values[m]; lab = df.label.values[m]
    ok = ~np.isnan(s)
    return roc_auc_score(lab[ok], s[ok])


print('\n=== gHM ALONE as a predictor (univariate AUC) ===')
print(f'  human fires vs random land:    {uni_auc((df.cause==1).values):.3f}')
print(f'  lightning fires vs random land:{uni_auc((df.cause==0).values):.3f}')
print('  -> if lightning ~ human, gHM is reporting bias (lightning fires DONT need human pressure to start)')
print('  -> if lightning << human, gHM is CAUSAL for human ignition')

# blocked space+time: gHM-model AUC on human-only vs lightning-only test
FEATS = [c for c in df.columns if c not in ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']]
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values; X = df[FEATS].values.astype('float32'); oof = np.full(len(y), np.nan)
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20:
        continue
    m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
    m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
mask = ~np.isnan(oof); ca = df.cause.values
print('\n=== full gHM-model, blocked space+time, by cause ===')
hm = mask & ((ca == 1) | (y == 0)); lm = mask & ((ca == 0) | (y == 0))
print(f'  human ignitions AUC    {roc_auc_score(y[hm], oof[hm]):.4f}')
print(f'  lightning ignitions AUC{roc_auc_score(y[lm], oof[lm]):.4f}')
