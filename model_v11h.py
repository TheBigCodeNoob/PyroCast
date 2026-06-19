"""v11h: v11e (25k) + canopy/treecover. Definitive new-best validation: as-is + bootstrap
CI, crutch controls (pop, dev500), env-floor, Florida-only, leave-region-out. prev vs +canopy."""
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
    return pd.concat(out, ignore_index=True)


ve = _safe('Training Data Florida/v11e/*.csv').dropna()
vh = _safe('Training Data Florida/v11h_canopy/*.csv').dropna()
CAN = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
for d in (ve, vh):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
df = ve.merge(vh[['k'] + CAN].drop_duplicates('k'), on='k').reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
ALL = [c for c in df.columns if c not in META]
PREV = [c for c in ALL if c not in CAN]
HUMAN = ['Pop_Density', 'DistDev', 'LC_Developed', 'NightLights', 'nbhd_dev_500m']
LANDUSE = ['LC_Crop', 'LC_Pasture', 'nbhd_crop_2km', 'nbhd_pasture_2km']
ENV = [c for c in ALL if c not in HUMAN + LANDUSE]
ENV_NOC = [c for c in ENV if c not in CAN]
y = df.label.astype(int).values
print(f'merged={len(df)} fire={int(y.sum())} neg={int((y==0).sum())}')


def oof(d, cols):
    yy = d.label.astype(int).values
    bl = (np.floor(d.lon).astype(int).astype(str) + '_' + np.floor(d.lat).astype(int).astype(str)).values
    yrr = d.year.values; X = np.nan_to_num(d[cols].values.astype('float32')); o = np.full(len(yy), np.nan)
    for tg, eg in GroupKFold(5).split(X, yy, bl):
        tr = tg[yrr[tg] <= 2019]; te = eg[yrr[eg] >= 2020]
        if len(tr) < 100 or yy[tr].sum() < 20 or (yy[tr] == 0).sum() < 20:
            continue
        m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
        m.fit(X[tr], yy[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o


def auc(d, o):
    yy = d.label.astype(int).values; m = ~np.isnan(o); return roc_auc_score(yy[m], o[m])


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


o_prev = oof(df, PREV); o_all = oof(df, ALL)
yt = y[~np.isnan(o_all)]; pt = o_all[~np.isnan(o_all)]; rng = np.random.default_rng(0); boots = []
for _ in range(300):
    idx = rng.integers(0, len(yt), len(yt))
    if yt[idx].sum() > 5 and (yt[idx] == 0).sum() > 5:
        boots.append(roc_auc_score(yt[idx], pt[idx]))
print('\n=== v11h = v11e(25k) + canopy/treecover ===')
print(f'  as-is:            prev {auc(df,o_prev):.4f}  ->  +canopy {auc(df,o_all):.4f}   95% CI [{np.percentile(boots,2.5):.4f},{np.percentile(boots,97.5):.4f}]')
print(f'  pop-matched:      prev {auc(match(df,"Pop_Density"),oof(match(df,"Pop_Density"),PREV)):.4f}  ->  +canopy {auc(match(df,"Pop_Density"),oof(match(df,"Pop_Density"),ALL)):.4f}')
print(f'  dev500-matched:   prev {auc(match(df,"nbhd_dev_500m"),oof(match(df,"nbhd_dev_500m"),PREV)):.4f}  ->  +canopy {auc(match(df,"nbhd_dev_500m"),oof(match(df,"nbhd_dev_500m"),ALL)):.4f}')
print(f'  ENV floor:        prev {auc(df,oof(df,ENV_NOC)):.4f}  ->  +canopy {auc(df,oof(df,ENV)):.4f}')
fl = df[(df.lat < 31.0) & (df.lon > -87.6)].reset_index(drop=True)
print(f'  FLORIDA-only (n={len(fl)}): prev {auc(fl,oof(fl,PREV)):.4f}  ->  +canopy {auc(fl,oof(fl,ALL)):.4f}')
lon_b = pd.qcut(df.lon, 4, labels=False); lat_b = pd.qcut(df.lat, 2, labels=False); region = (lon_b * 2 + lat_b).values
X = np.nan_to_num(df[ALL].values.astype('float32')); ra = []
for r in np.unique(region):
    tr = region != r; te = region == r
    if y[te].sum() < 20 or (y[te] == 0).sum() < 20:
        continue
    m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(X[tr], y[tr])
    ra.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
print(f'  leave-region-out (8): mean={np.mean(ra):.4f} min={np.min(ra):.4f}')
