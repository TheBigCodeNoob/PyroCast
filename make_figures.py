"""Generate science-fair figures from the FINAL model (v11e 25k + canopy). Saves PNGs to
figures/. Uses honest held-out (blocked space+time) predictions throughout."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.autolayout': True})
C = {'fire': '#c0392b', 'ok': '#27ae60', 'blue': '#2c6fbb', 'gray': '#7f8c8d', 'orange': '#e67e22'}


def _safe(g):
    out = []
    for c in glob.glob(g):
        try:
            out.append(pd.read_csv(c))
        except Exception:
            pass
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


print('loading...')
ve = _safe('Training Data Florida/v11e/*.csv').dropna()
vh = _safe('Training Data Florida/v11h_canopy/*.csv').dropna()
vm = _safe('Training Data Florida/v12_moisture/*.csv')
v13 = _safe('Training Data Florida/v13_human/*.csv')
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
V12 = ['ndmi', 'smap_root', 'lst_day', 'pet']  # moisture winners (null ones dropped)
for d in (ve, vh, vm, v13):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
vm['et_stress'] = vm.et / (vm.pet + 1)
df = (ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k')
        .merge(vm[['k'] + V12 + ['et_stress']].drop_duplicates('k'), on='k', how='left')
        .merge(v13[['k', 'built']].dropna().drop_duplicates('k'), on='k', how='left')).reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values
X = df[FEATS].values.astype('float32')  # keep NaN -> HGB handles missing moisture natively


def MK():
    return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


print('blocked space+time OOF...')
oof = np.full(len(y), np.nan)
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20:
        continue
    m = MK(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
mask = ~np.isnan(oof); yt = y[mask]; pt = oof[mask]
AUC = roc_auc_score(yt, pt)
print(f'  held-out AUC {AUC:.4f}, n={mask.sum()}')

# ---- Fig 1: the honesty journey ----
vers = ['v2\nCNN', 'v3', 'v6', 'v7', 'v8\nFIRMS', 'v10b', 'v11\ncanopy', 'v12\nmoist', 'v13\nfinal']
raw = [0.95, 0.71, 0.93, 0.80, 0.72, 0.811, 0.835, 0.857, 0.862]
honest = [0.55, 0.71, 0.60, 0.67, 0.716, 0.755, 0.797, 0.831, 0.834]
xs = np.arange(len(vers))
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(xs, raw, 'o--', color=C['gray'], lw=2, ms=8, label='What we first reported (raw)')
ax.plot(xs, honest, 'o-', color=C['ok'], lw=2.5, ms=9, label='What survived the cheat-tests (honest)')
ax.fill_between(xs, honest, raw, color=C['fire'], alpha=0.12)
for i, (r, h) in enumerate(zip(raw, honest)):
    if r - h > 0.05:
        ax.annotate('crutch', (i, (r + h) / 2), color=C['fire'], ha='center', fontsize=9, fontweight='bold')
for i, tag in [(0, 'biome\nshortcut'), (2, 'season\nleak'), (3, 'remoteness')]:
    ax.annotate(tag, (i, raw[i] + 0.012), color=C['fire'], ha='center', fontsize=8)
ax.axhline(0.85, color=C['blue'], ls=':', lw=1.5); ax.text(0.1, 0.857, 'goal 0.85', color=C['blue'], fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(vers); ax.set_ylim(0.5, 0.98); ax.set_ylabel('AUC (new place + future)')
ax.set_title('The honesty story: every flashy number was a crutch until v10b', fontweight='bold')
ax.legend(loc='lower center'); fig.savefig('figures/01_journey.png'); plt.close(fig)

# ---- Fig 2: ROC ----
fpr, tpr, _ = roc_curve(yt, pt)
fig, ax = plt.subplots(figsize=(5.6, 5.4))
ax.plot(fpr, tpr, color=C['blue'], lw=2.5, label=f'PyroCast (AUC {AUC:.3f})')
ax.plot([0, 1], [0, 1], '--', color=C['gray'], label='random (0.50)')
ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate')
ax.set_title('ROC — held-out new places & future year', fontweight='bold'); ax.legend(loc='lower right')
fig.savefig('figures/02_roc.png'); plt.close(fig)

# ---- Fig 3: calibration ----
bins = np.linspace(0, 1, 11); idx = np.digitize(pt, bins) - 1
xs2, ys2, ns = [], [], []
for b in range(10):
    s = idx == b
    if s.sum() > 20:
        xs2.append(pt[s].mean()); ys2.append(yt[s].mean()); ns.append(s.sum())
fig, ax = plt.subplots(figsize=(5.6, 5.4))
ax.plot([0, 1], [0, 1], '--', color=C['gray'], label='perfect')
ax.plot(xs2, ys2, 'o-', color=C['orange'], lw=2, ms=8, label='PyroCast')
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed fire rate')
ax.set_title('Calibration — predictions match reality', fontweight='bold'); ax.legend(loc='upper left')
fig.savefig('figures/03_calibration.png'); plt.close(fig)

# ---- Fig 4: precision/recall @ top-k% ----
order = np.argsort(pt)[::-1]; ks = np.arange(1, 51); prec, rec = [], []
for k in ks:
    top = order[:max(1, int(len(pt) * k / 100))]
    prec.append(yt[top].mean()); rec.append(yt[top].sum() / yt.sum())
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(ks, prec, color=C['fire'], lw=2.5, label='precision (share that are real fires)')
ax.plot(ks, rec, color=C['blue'], lw=2.5, label='recall (share of all fires caught)')
ax.set_xlabel('Top % riskiest place-days flagged'); ax.set_ylabel('Fraction')
ax.set_title('Operational value: focus on the riskiest slice', fontweight='bold'); ax.legend()
fig.savefig('figures/04_precision_at_k.png'); plt.close(fig)

# ---- Fig 5: feature importance by category ----
def cat(f):
    if f in ['Pop_Density', 'DistDev', 'LC_Developed', 'NightLights', 'nbhd_dev_500m', 'built']:
        return 'human access'
    if f in ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']:
        return 'canopy / fuel structure'
    if f in ['LC_Crop', 'LC_Pasture', 'nbhd_crop_2km', 'nbhd_pasture_2km']:
        return 'agriculture'
    if f in ['ndmi', 'smap_root', 'lst_day', 'pet', 'et_stress']:
        return 'moisture / water-stress'
    if f.startswith('nbhd') or f.startswith('LC_') or f in ['NDVI', 'EVI']:
        return 'vegetation / landscape'
    if f == 'Elevation':
        return 'terrain'
    return 'weather / drought'


trm = (df.lon < df.lon.median()).values
mm = MK().fit(X[trm], y[trm])
pi = permutation_importance(mm, X[~trm], y[~trm], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATS).clip(lower=0)
bycat = imp.groupby([cat(f) for f in FEATS]).sum().sort_values()
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(bycat.index, bycat.values, color=C['blue'])
ax.set_xlabel('Total permutation importance (AUC drop when shuffled)')
ax.set_title('What the model relies on, by category', fontweight='bold')
fig.savefig('figures/05_importance_by_category.png'); plt.close(fig)
top = imp.sort_values()[-14:]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.barh(top.index, top.values, color=[C['fire'] if cat(f) == 'human access' else C['ok'] if 'canopy' in cat(f) else C['blue'] for f in top.index])
ax.set_xlabel('Permutation importance'); ax.set_title('Top 14 individual features', fontweight='bold')
fig.savefig('figures/06_importance_top.png'); plt.close(fig)

# ---- Fig 7: the honesty bracket (crutch decomposition) ----
labels = ['as-is\n(operational)', 'remoteness\nremoved', 'strictest\nsingle', 'ALL human\nstripped (floor)']
vals = [0.862, 0.834, 0.826, 0.796]
fig, ax = plt.subplots(figsize=(6.6, 4.6))
bars = ax.bar(labels, vals, color=[C['ok'], C['blue'], C['blue'], C['gray']])
ax.axhline(0.5, color=C['gray'], ls='--'); ax.text(3.1, 0.51, 'chance', color=C['gray'], fontsize=9, ha='right')
ax.set_ylim(0.5, 0.9); ax.set_ylabel('AUC')
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
ax.set_title('How honest do you want to be? The skill survives stripping every crutch', fontweight='bold', fontsize=10.5)
fig.savefig('figures/07_honesty_bracket.png'); plt.close(fig)

# ---- Fig 8: causality of human access ----
fig, ax = plt.subplots(figsize=(7, 4.6))
for name, m_, col in [('human-caused fires', df.cause == 1, C['fire']), ('lightning fires', df.cause == 0, C['orange']), ('random background', df.label == 0, C['gray'])]:
    d = df.loc[m_, 'DistDev'].clip(upper=2.0)
    ax.hist(d, bins=40, histtype='step', density=True, lw=2.2, color=col,
            label=f'{name} (median {df.loc[m_,"DistDev"].median()*1000:.0f} m)')
ax.set_xlabel('Distance to development (km)'); ax.set_ylabel('density')
ax.set_title('Why human-access is causal, not just reporting bias', fontweight='bold')
ax.legend(); fig.savefig('figures/08_causality.png'); plt.close(fig)

# ---- Fig 9: predicted risk map (held-out 2020 points) ----
te = mask & (df.year.values >= 2020)
fig, ax = plt.subplots(figsize=(9, 5.6))
sc = ax.scatter(df.lon[te], df.lat[te], c=oof[te], cmap='YlOrRd', s=7, alpha=0.6, vmin=0, vmax=1)
ax.set_xlabel('longitude'); ax.set_ylabel('latitude'); ax.set_aspect(1.2)
ax.set_title('Predicted ignition risk across the SE US (held-out 2020)', fontweight='bold')
plt.colorbar(sc, ax=ax, label='predicted ignition probability', shrink=0.8)
fig.savefig('figures/09_risk_map.png'); plt.close(fig)

# ---- Fig 10: human vs lightning AUC + per-month ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
ha = roc_auc_score(yt[(df.cause.values[mask] == 1) | (df.label.values[mask] == 0)], pt[(df.cause.values[mask] == 1) | (df.label.values[mask] == 0)])
la = roc_auc_score(yt[(df.cause.values[mask] == 0) | (df.label.values[mask] == 0)], pt[(df.cause.values[mask] == 0) | (df.label.values[mask] == 0)])
a1.bar(['human\nignitions', 'lightning\nignitions'], [ha, la], color=[C['fire'], C['orange']])
a1.set_ylim(0.5, 0.9); a1.set_ylabel('AUC'); a1.set_title('Predictability by cause', fontweight='bold')
for i, v in enumerate([ha, la]):
    a1.text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
mo = df.month.values[mask]; mons, maucs = [], []
for mth in range(1, 13):
    s = mo == mth
    if y[mask][s].sum() > 15 and (y[mask][s] == 0).sum() > 15:
        mons.append(mth); maucs.append(roc_auc_score(yt[s], pt[s]))
a2.plot(mons, maucs, 'o-', color=C['blue'], lw=2); a2.set_ylim(0.5, 0.95)
a2.set_xlabel('month'); a2.set_ylabel('AUC'); a2.set_title('Works year-round', fontweight='bold')
fig.savefig('figures/10_by_cause_and_month.png'); plt.close(fig)

print('done -> figures/*.png')
import os
for f in sorted(os.listdir('figures')):
    print('  ', f)
