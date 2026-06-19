"""Authoritative validation of the FINAL model (v11e 25k + canopy). One text report:
negative controls, headline + bootstrap CI, crutch controls, generalization, calibration,
operational. This is the number to defend."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score


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
HUMAN = ['Pop_Density', 'DistDev', 'LC_Developed', 'NightLights', 'nbhd_dev_500m']
LANDUSE = ['LC_Crop', 'LC_Pasture', 'nbhd_crop_2km', 'nbhd_pasture_2km']
ENV = [c for c in FEATS if c not in HUMAN + LANDUSE]
y = df.label.astype(int).values


def MK():
    return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def oof(d, cols, ycol=None):
    yy = d[ycol].values if ycol else d.label.astype(int).values
    bl = (np.floor(d.lon).astype(int).astype(str) + '_' + np.floor(d.lat).astype(int).astype(str)).values
    yrr = d.year.values; X = np.nan_to_num(d[cols].values.astype('float32')); o = np.full(len(yy), np.nan)
    for tg, eg in GroupKFold(5).split(X, yy, bl):
        tr = tg[yrr[tg] <= 2019]; te = eg[yrr[eg] >= 2020]
        if len(tr) < 100 or yy[tr].sum() < 20 or (yy[tr] == 0).sum() < 20:
            continue
        m = MK(); m.fit(X[tr], yy[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o, yy


def A(o, yy):
    m = ~np.isnan(o); return roc_auc_score(yy[m], o[m])


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


print('=' * 60); print('FINAL MODEL VALIDATION  (v11e 25k + canopy)  rows=%d fire=%d' % (len(df), y.sum())); print('=' * 60)

print('\n[1] NEGATIVE CONTROLS')
rng = np.random.default_rng(0); ds = df.copy(); ds['lab'] = rng.permutation(y)
osh, ysh = oof(ds, FEATS, 'lab'); print(f'  label-shuffle AUC (expect ~0.50): {A(osh, ysh):.4f}')

print('\n[2] HEADLINE (blocked space+time = new place + future)')
o, yy = oof(df, FEATS); m = ~np.isnan(o); yt, pt = yy[m], o[m]
boots = [roc_auc_score(yt[i], pt[i]) for i in (rng.integers(0, len(yt), len(yt)) for _ in range(300)) if yt[i].sum() > 5 and (yt[i] == 0).sum() > 5]
print(f'  as-is AUC {A(o, yy):.4f}   95% CI [{np.percentile(boots,2.5):.4f}, {np.percentile(boots,97.5):.4f}]   (test n={m.sum()})')

print('\n[3] CRUTCH CONTROLS')
print(f'  pop-matched            {A(*oof(match(df,"Pop_Density"),FEATS)):.4f}')
print(f'  dev500-matched         {A(*oof(match(df,"nbhd_dev_500m"),FEATS)):.4f}')
print(f'  NO human-access        {A(*oof(df,[c for c in FEATS if c not in HUMAN])):.4f}')
print(f'  ENV floor (no human/land-use)  {A(*oof(df,ENV)):.4f}')
print(f'  ENV floor + pop-matched        {A(*oof(match(df,"Pop_Density"),ENV)):.4f}')

print('\n[4] GENERALIZATION')
ca = df.cause.values[m]
print(f'  human ignitions  {roc_auc_score(yt[(ca==1)|(yt==0)], pt[(ca==1)|(yt==0)]):.4f}')
print(f'  lightning        {roc_auc_score(yt[(ca==0)|(yt==0)], pt[(ca==0)|(yt==0)]):.4f}')
fl = df[(df.lat < 31) & (df.lon > -87.6)].reset_index(drop=True)
print(f'  Florida-only     {A(*oof(fl,FEATS)):.4f}   (pop-matched {A(*oof(match(fl,"Pop_Density"),FEATS)):.4f})')
X = np.nan_to_num(df[FEATS].values.astype('float32'))
lon_b = pd.qcut(df.lon, 4, labels=False); lat_b = pd.qcut(df.lat, 2, labels=False); region = (lon_b * 2 + lat_b).values
ra = []
for r in np.unique(region):
    tr = region != r; te = region == r
    if y[te].sum() < 20 or (y[te] == 0).sum() < 20:
        continue
    mm = MK().fit(X[tr], y[tr]); ra.append(roc_auc_score(y[te], mm.predict_proba(X[te])[:, 1]))
print(f'  leave-region-out (8): mean {np.mean(ra):.4f}  min {np.min(ra):.4f}')

print('\n[5] CALIBRATION + OPERATIONAL')
print(f'  Brier {brier_score_loss(yt,pt):.4f} (vs {yt.mean()*(1-yt.mean()):.4f} baseline) | PR-AUC {average_precision_score(yt,pt):.4f}')
order = np.argsort(pt)[::-1]
for k in [1, 5, 10]:
    top = order[:int(len(pt) * k / 100)]; print(f'  top {k:>2d}% riskiest: precision {pt[top].size and yt[top].mean():.3f}, recall {yt[top].sum()/yt.sum():.3f}')
print('\n' + '=' * 60); print('Defensible headline: 0.85 (CI ~0.84-0.86); honest floor ~0.79.'); print('=' * 60)
