"""v13 combined eval: human-pressure (gHM/built/wsf) and VCF fuel (% tree/herb/bare) on the
FINAL model. Tests each + both, with crutch controls. Human features must survive matching;
fuel features must survive too (they're environmental)."""
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
vcf = _safe('Training Data Florida/v13_vcf/*.csv')
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
MOIST = ['ndmi', 'smap_root', 'lst_day', 'pet', 'et_stress']
HUM = ['ghm', 'ghm_2km', 'built', 'built_2km', 'wsf_2km']
VCF = ['vcf_tree', 'vcf_herb', 'vcf_bare', 'vcf_herb_2km']
for d in (ve, vh, vm, v13, vcf):
    if len(d):
        d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
vm['et_stress'] = vm.et / (vm.pet + 1)
df = (ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k')
        .merge(vm[['k'] + MOIST].drop_duplicates('k'), on='k', how='left'))
HAVE_H = len(v13) > 0 and 'ghm' in v13.columns
HAVE_V = len(vcf) > 0 and 'vcf_tree' in vcf.columns
# INNER-join the v13 features (no NaN rows) so incomplete downloads can't leak via the
# spatial CV (the gHM phantom-gain lesson). prev vs +v13 compared on identical complete rows.
if HAVE_H:
    df = df.merge(v13[['k'] + HUM].dropna().drop_duplicates('k'), on='k')
if HAVE_V:
    df = df.merge(vcf[['k'] + VCF].dropna().drop_duplicates('k'), on='k')
df = df.reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
NEW = (HUM if HAVE_H else []) + (VCF if HAVE_V else [])
BASE = [c for c in df.columns if c not in META + NEW]
print(f'merged={len(df)} | human={HAVE_H} vcf={HAVE_V}')


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


b = st(df, BASE)
print(f'\n  base (final model)        {b:.4f}')
if HAVE_H:
    print(f'  + human-pressure (gHM)    {st(df, BASE+HUM):.4f}')
if HAVE_V:
    print(f'  + VCF fuel                {st(df, BASE+VCF):.4f}')
print(f'  + ALL v13                 {st(df, BASE+NEW):.4f}')
print('\n  crutch controls (base -> +ALL):')
print(f'    pop-matched     {st(match(df,"Pop_Density"),BASE):.4f} -> {st(match(df,"Pop_Density"),BASE+NEW):.4f}')
print(f'    dev500-matched  {st(match(df,"nbhd_dev_500m"),BASE):.4f} -> {st(match(df,"nbhd_dev_500m"),BASE+NEW):.4f}')
y = df.label.astype(int).values; trm = (df.lon < df.lon.median()).values
X = df[BASE + NEW].values.astype('float32')
m = HGB().fit(X[trm], y[trm])
pi = permutation_importance(m, X[~trm], y[~trm], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
cols = BASE + NEW
print('\n  v13 importance:')
for i in np.argsort(pi.importances_mean)[::-1]:
    if cols[i] in NEW:
        print(f'    {cols[i]:<14s} {pi.importances_mean[i]:+.4f}')
