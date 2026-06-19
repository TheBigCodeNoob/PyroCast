"""How much does PyroCast beat trivial baselines? Same blocked space+time test for all.
Shows the model adds real value over 'just predict near people' and over a spatial climatology."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
MOIST = ['ndmi', 'smap_root', 'lst_day', 'pet']
for d in (ve, vh, vm, v13):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
vm['et_stress'] = vm.et / (vm.pet + 1)
df = (ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k')
        .merge(vm[['k'] + MOIST + ['et_stress']].drop_duplicates('k'), on='k', how='left')
        .merge(v13[['k', 'built']].dropna().drop_duplicates('k'), on='k', how='left')).reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values
tr_all = yr <= 2019
te_all = yr >= 2020


def st_split():
    return GroupKFold(5).split(df[FEATS].values, y, block)


def auc_model(cols, linear=False):
    X = np.nan_to_num(df[cols].values.astype('float32')); oof = np.full(len(y), np.nan)
    for tg, eg in GroupKFold(5).split(X, y, block):
        tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
        if len(tr) < 100 or y[tr].sum() < 20:
            continue
        if linear:
            sc = StandardScaler(); m = LogisticRegression(max_iter=3000)
            m.fit(sc.fit_transform(X[tr]), y[tr]); oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof); return roc_auc_score(y[mask], oof[mask])


def auc_univariate(col, sign=1):
    # rank by a single feature, evaluated on the 2020 test rows only
    s = sign * df[col].values; m = te_all & ~np.isnan(s)
    return roc_auc_score(y[m], s[m])


def auc_climatology():
    # baseline: historical fire fraction per 0.5-deg cell from TRAIN years, applied to TEST
    cell = (np.floor(df.lon / 0.5).astype(int).astype(str) + '_' + np.floor(df.lat / 0.5).astype(int).astype(str)).values
    rate = pd.Series(y[tr_all]).groupby(cell[tr_all]).mean()
    pred = pd.Series(cell[te_all]).map(rate).fillna(rate.mean()).values
    return roc_auc_score(y[te_all], pred)


print('=== PyroCast vs baselines (blocked space+time, new place + future) ===')
res = {
    'always "near people"\n(distance-to-developed)': auc_univariate('DistDev', sign=-1),
    'population density': auc_univariate('Pop_Density'),
    'recent dryness (VPD 90d)': auc_univariate('vpd_90'),
    'spatial fire climatology\n(history per cell)': auc_climatology(),
    'linear model\n(all features)': auc_model(FEATS, linear=True),
    'PyroCast\n(full model)': auc_model(FEATS),
}
for k, v in res.items():
    print(f'  {k.split(chr(10))[0]:<28s} {v:.3f}')

fig, ax = plt.subplots(figsize=(8.5, 4.8))
plt.rcParams.update({'font.size': 11})
names = list(res.keys()); vals = list(res.values())
colors = ['#7f8c8d'] * (len(names) - 1) + ['#c0392b']
b = ax.barh(range(len(names)), vals, color=colors)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
ax.axvline(0.5, color='gray', ls='--'); ax.set_xlim(0.45, 0.9); ax.set_xlabel('AUC (new place + future year)')
for i, v in enumerate(vals):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=9)
ax.set_title('PyroCast beats trivial baselines', fontweight='bold')
ax.invert_yaxis(); fig.tight_layout(); fig.savefig('figures/13_baselines.png', dpi=140); plt.close(fig)
print('\nsaved figures/13_baselines.png')
