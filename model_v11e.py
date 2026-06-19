"""v11e DEFINITIVE validation on FRESH 25k data (never tuned against): does the 0.837
hold? Locked space+time metric + bootstrap CI + crutch controls + leave-region-out."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11e/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
print(f'rows={len(df)} fire={int(y.sum())} neg={int((y==0).sum())}  ({len(FEATS)} feats)')
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values


def MK():
    return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def st_oof(d, cols):
    yy = d.label.astype(int).values
    bl = (np.floor(d.lon).astype(int).astype(str) + '_' + np.floor(d.lat).astype(int).astype(str)).values
    yrr = d.year.values; X = np.nan_to_num(d[cols].values.astype('float32')); oof = np.full(len(yy), np.nan)
    for tg, eg in GroupKFold(5).split(X, yy, bl):
        tr = tg[yrr[tg] <= 2019]; te = eg[yrr[eg] >= 2020]
        if len(tr) < 100 or yy[tr].sum() < 20:
            continue
        m = MK(); m.fit(X[tr], yy[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof); return roc_auc_score(yy[mask], oof[mask]), oof, mask


def match(d, col):
    nz = d[col][d[col] > 0]
    edges = np.unique([d[col].min() - 1, 1e-9] + list(nz.quantile([.25, .5, .75]).values) + [np.inf]) if len(nz) else np.array([-1, np.inf])
    d = d.copy(); d['pb'] = pd.cut(d[col], bins=edges, duplicates='drop')
    posf = d[d.label == 1].pb.value_counts(normalize=True); negc = d[d.label == 0].pb.value_counts()
    N = int(min(negc.get(b, 0) / posf[b] for b in posf.index if posf[b] > 0)); keep = [d[d.label == 1]]
    for b in posf.index:
        pool = d[(d.label == 0) & (d.pb == b)]; kk = int(round(posf[b] * N))
        if len(pool) and kk:
            keep.append(pool.sample(min(kk, len(pool)), random_state=0))
    return pd.concat(keep).reset_index(drop=True)


a, oof, mask = st_oof(df, FEATS)
yt, pt = y[mask], oof[mask]; rng = np.random.default_rng(0); boots = []
for _ in range(300):
    idx = rng.integers(0, len(yt), len(yt))
    if yt[idx].sum() > 5 and (yt[idx] == 0).sum() > 5:
        boots.append(roc_auc_score(yt[idx], pt[idx]))
HUMAN = ['Pop_Density', 'DistDev', 'LC_Developed', 'NightLights', 'nbhd_dev_500m']
LANDUSE = ['LC_Crop', 'LC_Pasture', 'nbhd_crop_2km', 'nbhd_pasture_2km']
ENV = [c for c in FEATS if c not in HUMAN + LANDUSE]
print('\n=== LOCKED metric on FRESH 25k data ===')
print(f'  as-is             {a:.4f}   95% CI [{np.percentile(boots,2.5):.4f}, {np.percentile(boots,97.5):.4f}]')
print(f'  pop-matched       {st_oof(match(df,"Pop_Density"),FEATS)[0]:.4f}')
print(f'  DistDev-matched   {st_oof(match(df,"DistDev"),FEATS)[0]:.4f}')
print(f'  dev500-matched    {st_oof(match(df,"nbhd_dev_500m"),FEATS)[0]:.4f}   (STRICTER: strongest single human-access lever)')
print(f'  NO human-access   {st_oof(df,[c for c in FEATS if c not in HUMAN])[0]:.4f}')
print(f'  ENV floor (no human, no land-use)  {st_oof(df,ENV)[0]:.4f}')
print(f'  ENV floor + pop-matched            {st_oof(match(df,"Pop_Density"),ENV)[0]:.4f}')
fl = df[(df.lat < 31.0) & (df.lon > -87.6)].reset_index(drop=True)
print(f'  FLORIDA-only (n={len(fl)}): as-is {st_oof(fl,FEATS)[0]:.4f} / pop-matched {st_oof(match(fl,"Pop_Density"),FEATS)[0]:.4f}')
print('  [tuned 10k for reference: as-is 0.835 / pop 0.784 / dev500 0.773 / env-floor 0.72]')

# leave-one-region-out generalization
lon_b = pd.qcut(df.lon, 4, labels=False); lat_b = pd.qcut(df.lat, 2, labels=False); region = (lon_b * 2 + lat_b).values
X = np.nan_to_num(df[FEATS].values.astype('float32')); ra = []
for r in np.unique(region):
    tr = region != r; te = region == r
    if y[te].sum() < 20 or (y[te] == 0).sum() < 20:
        continue
    m = MK(); m.fit(X[tr], y[tr]); ra.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
print(f'\n  leave-region-out (8): mean={np.mean(ra):.4f} min={np.min(ra):.4f} max={np.max(ra):.4f}')
