"""v13a eval: do human-pressure features (gHM = roads+power+infrastructure, finer built-up)
help the FINAL model? These are human-access, so the test is whether they SURVIVE matching
(if the gain vanishes pop/dev-matched, it's just more remoteness; if it survives, gHM is
capturing real ignition-cause signal DistDev missed)."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
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
vh = _safe('Training Data Florida/v11h_canopy/*.csv').dropna()
vm = _safe('Training Data Florida/v12_moisture/*.csv')
v13 = _safe('Training Data Florida/v13_human/*.csv')
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
MOIST = ['ndmi', 'smap_root', 'lst_day', 'pet', 'et_stress']
V13 = ['ghm', 'ghm_2km', 'built', 'built_2km', 'wsf_2km']
for d in (ve, vh, vm, v13):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
vm['et_stress'] = vm.et / (vm.pet + 1)
df = (ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k')
        .merge(vm[['k'] + MOIST].drop_duplicates('k'), on='k', how='left')
        .merge(v13[['k'] + V13].drop_duplicates('k'), on='k', how='left')).reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
ALL = [c for c in df.columns if c not in META]
PREV = [c for c in ALL if c not in V13]
print(f'merged={len(df)} v13 coverage {df.ghm.notna().mean():.0%}')


def HGB():
    return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def st(d, cols):
    yy = d.label.astype(int).values
    bl = (np.floor(d.lon).astype(int).astype(str) + '_' + np.floor(d.lat).astype(int).astype(str)).values
    yrr = d.year.values; X = d[cols].values.astype('float32'); o = np.full(len(yy), np.nan)
    for tg, eg in GroupKFold(5).split(X, yy, bl):
        tr = tg[yrr[tg] <= 2019]; te = eg[yrr[eg] >= 2020]
        if len(tr) < 100 or yy[tr].sum() < 20 or (yy[tr] == 0).sum() < 20:
            continue
        m = HGB(); m.fit(X[tr], yy[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(o); return roc_auc_score(yy[mask], o[mask])


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


print('\n=== v13 human-pressure on the final model ===')
print(f'  as-is:          prev {st(df, PREV):.4f}  ->  +v13 {st(df, ALL):.4f}')
print(f'  pop-matched:    prev {st(match(df,"Pop_Density"),PREV):.4f}  ->  +v13 {st(match(df,"Pop_Density"),ALL):.4f}')
print(f'  dev500-matched: prev {st(match(df,"nbhd_dev_500m"),PREV):.4f}  ->  +v13 {st(match(df,"nbhd_dev_500m"),ALL):.4f}')
print(f'  ghm-matched:    prev {st(match(df,"ghm"),PREV):.4f}  ->  +v13 {st(match(df,"ghm"),ALL):.4f}  (does it survive matching on itself?)')
y = df.label.astype(int).values; trm = (df.lon < df.lon.median()).values
X = df[ALL].values.astype('float32')
m = HGB().fit(X[trm], y[trm])
pi = permutation_importance(m, X[~trm], y[~trm], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
print('\n  v13 importance:')
for i in np.argsort(pi.importances_mean)[::-1]:
    if ALL[i] in V13:
        print(f'    {ALL[i]:<12s} {pi.importances_mean[i]:+.4f}')
