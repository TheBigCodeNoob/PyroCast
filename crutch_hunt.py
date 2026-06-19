"""
"Is the crutch hiding in a DIFFERENT factor?" — exhaustive crutch hunt on the current
best (v10b + v11c + v11d). Plus the consistent real-world-AUC table for the FPA-FOD
series, and a Florida-only evaluation.
Metric throughout = blocked space+time (leave-spatial-block-out x train<=2019/test 2020)
= genuinely new place + future.
"""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

d10 = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna()
dc = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11c_extra/*.csv')], ignore_index=True).dropna()
dd = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11d_extra/*.csv')], ignore_index=True).dropna()
V11C = ['NightLights', 'nbhd_dev_500m', 'nbhd_forest_2km', 'nbhd_wetland_2km']
V11D = ['nbhd_crop_2km', 'nbhd_wetland_5km', 'nbhd_pasture_2km']
for d in (d10, dc, dd):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
df = d10.merge(dc[['k'] + V11C].drop_duplicates('k'), on='k').merge(dd[['k'] + V11D].drop_duplicates('k'), on='k').reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
ALL = [c for c in df.columns if c not in META]

# --- factor groups ---
HUMAN = ['Pop_Density', 'DistDev', 'LC_Developed', 'NightLights', 'nbhd_dev_500m']
LANDUSE = ['LC_Crop', 'LC_Pasture', 'nbhd_crop_2km', 'nbhd_pasture_2km']
FUEL = ['LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Wetland', 'nbhd_forest_2km', 'nbhd_wetland_2km', 'nbhd_wetland_5km', 'NDVI', 'EVI']
TERRAIN = ['Elevation']
WEATHER = [c for c in ALL if c not in HUMAN + LANDUSE + FUEL + TERRAIN]
ENV = FUEL + TERRAIN + WEATHER  # no human, no land-use


def st(d, cols):
    yy = d.label.astype(int).values
    bl = (np.floor(d.lon).astype(int).astype(str) + '_' + np.floor(d.lat).astype(int).astype(str)).values
    yrr = d.year.values; X = np.nan_to_num(d[cols].values.astype('float32')); oof = np.full(len(yy), np.nan)
    for tg, eg in GroupKFold(5).split(X, yy, bl):
        tr = tg[yrr[tg] <= 2019]; te = eg[yrr[eg] >= 2020]
        if len(tr) < 100 or yy[tr].sum() < 20 or (yy[tr] == 0).sum() < 20:
            continue
        m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
        m.fit(X[tr], yy[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof)
    return roc_auc_score(yy[mask], oof[mask]) if mask.sum() and yy[mask].sum() and (yy[mask] == 0).sum() else float('nan')


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


print('=' * 60); print('CRUTCH HUNT (current best = v10b+v11c+v11d)  rows=%d' % len(df)); print('=' * 60)

print('\n[1] UNIVARIATE giveaways — blocked space+time AUC using ONE feature alone (top 15):')
uni = sorted([(f, st(df, [f])) for f in ALL], key=lambda x: -x[1])
for f, a in uni[:15]:
    grp = 'HUMAN' if f in HUMAN else 'LANDUSE' if f in LANDUSE else 'FUEL' if f in FUEL else 'TERRAIN' if f in TERRAIN else 'weather'
    print(f'    {a:.3f}  [{grp:<7s}] {f}')

print('\n[2] NEUTRALIZE each big factor (distribution-match pos&neg on it), full model AUC:')
print(f'    {"(no matching)":<22s} {st(df, ALL):.4f}')
for col in HUMAN + ['Elevation', 'nbhd_forest_2km', 'nbhd_crop_2km', 'NDVI']:
    print(f'    match on {col:<22s} {st(match(df, col), ALL):.4f}')

print('\n[3] ABLATION — drop whole factor groups:')
print(f'    full ({len(ALL)})                    {st(df, ALL):.4f}')
print(f'    NO human-access              {st(df, [c for c in ALL if c not in HUMAN]):.4f}')
print(f'    NO human + NO land-use       {st(df, [c for c in ALL if c not in HUMAN+LANDUSE]):.4f}')
print(f'    ENVIRONMENT only (weather/fuel/terrain) {st(df, ENV):.4f}')
print(f'    HUMAN-access only            {st(df, HUMAN):.4f}')
print(f'    WEATHER only                 {st(df, WEATHER):.4f}')

print('\n[4] STRIP EVERYTHING AT ONCE — the floor:')
print(f'    env-only + pop-matched       {st(match(df, "Pop_Density"), ENV):.4f}')
print(f'    env-only + DistDev-matched   {st(match(df, "DistDev"), ENV):.4f}')

print('\n[5] FLORIDA-ONLY (test fires inside FL bbox lat 24.5-31, lon -87.6..-80):')
fl = df[(df.lat < 31.0) & (df.lon > -87.6)].reset_index(drop=True)
print(f'    Florida subset rows={len(fl)} fire={int((fl.label==1).sum())}')
print(f'    as-is           {st(fl, ALL):.4f}')
print(f'    pop-matched     {st(match(fl, "Pop_Density"), ALL):.4f}')

print('\n' + '=' * 60); print('CONSISTENT REAL-WORLD-AUC TABLE (FPA-FOD series, identical test)'); print('=' * 60)
B = [c for c in ALL if c not in V11C + V11D]
print(f'{"iteration":<16s}{"as-is":>8s}{"pop-match":>11s}{"DistDev-m":>11s}{"no-human":>10s}')
for name, cols in [('v10b', B), ('v10b+v11c', B + V11C), ('v10b+v11c+v11d', B + V11C + V11D)]:
    hum = [c for c in cols if c not in HUMAN]
    print(f'{name:<16s}{st(df, cols):>8.4f}{st(match(df,"Pop_Density"), cols):>11.4f}{st(match(df,"DistDev"), cols):>11.4f}{st(df, hum):>10.4f}')
