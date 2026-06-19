"""Cleaner predicted-risk heatmaps (SE-wide + Florida zoom) using held-out predictions."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
plt.rcParams.update({'figure.dpi': 140, 'font.size': 11, 'figure.autolayout': True})


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
oof = np.full(len(y), np.nan)
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20:
        continue
    m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
    m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
te = (~np.isnan(oof)) & (df.year.values >= 2020)
lon, lat, p = df.lon.values[te], df.lat.values[te], oof[te]
fires = te & (y == 1)

# SE-wide hexbin of mean predicted risk
fig, ax = plt.subplots(figsize=(9.5, 6))
hb = ax.hexbin(lon, lat, C=p, reduce_C_function=np.mean, gridsize=38, cmap='YlOrRd', mincnt=1, vmin=0, vmax=1)
ax.set_xlabel('longitude'); ax.set_ylabel('latitude'); ax.set_aspect(1.2)
ax.set_title('Predicted wildfire-ignition risk — Southeast US (held-out 2020)', fontweight='bold')
plt.colorbar(hb, ax=ax, label='mean predicted ignition probability', shrink=0.8)
fig.savefig('figures/09_risk_map.png'); plt.close(fig)

# Florida zoom with actual fires overlaid
flm = te & (df.lat.values < 31.0) & (df.lon.values > -87.6)
fig, ax = plt.subplots(figsize=(6.4, 7))
hb = ax.hexbin(df.lon.values[flm], df.lat.values[flm], C=oof[flm], reduce_C_function=np.mean, gridsize=28, cmap='YlOrRd', mincnt=1, vmin=0, vmax=1)
ff = flm & (y == 1)
ax.scatter(df.lon.values[ff], df.lat.values[ff], s=4, c='black', alpha=0.25, label='actual 2020 fires')
ax.set_xlabel('longitude'); ax.set_ylabel('latitude'); ax.set_aspect(1.1)
ax.set_title('Florida — predicted risk vs actual fires', fontweight='bold')
plt.colorbar(hb, ax=ax, label='predicted ignition probability', shrink=0.7)
ax.legend(loc='upper right', fontsize=8)
fig.savefig('figures/09b_risk_map_florida.png'); plt.close(fig)
print('risk maps saved')
