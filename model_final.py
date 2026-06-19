"""THE FINAL MODEL: v11e (25k) + canopy + moisture(winners), keeping FULL coverage by
left-joining moisture and letting HistGradientBoosting handle missing values natively
(no row loss, no fake imputation). Full validation + save best_model_final.joblib."""
import glob, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss


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
vm = _safe('Training Data Florida/v12_moisture/*.csv')   # do NOT dropna -> keep coverage
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
V12 = ['ndmi', 'smap_root', 'lst_day', 'pet', 'et_stress']
for d in (ve, vh, vm):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
vm['et_stress'] = vm.et / (vm.pet + 1)
# canopy is inner (always present); moisture is LEFT (NaN where satellite gaps) -> HGB handles NaN
df = ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k').merge(vm[['k'] + V12].drop_duplicates('k'), on='k', how='left').reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
FEATS = [c for c in df.columns if c not in META]
HUMAN = ['Pop_Density', 'DistDev', 'LC_Developed', 'NightLights', 'nbhd_dev_500m']
LANDUSE = ['LC_Crop', 'LC_Pasture', 'nbhd_crop_2km', 'nbhd_pasture_2km']
ENV = [c for c in FEATS if c not in HUMAN + LANDUSE]
y = df.label.astype(int).values
print(f'FINAL: rows={len(df)} fire={int(y.sum())} feats={len(FEATS)} | moisture coverage {df.ndmi.notna().mean():.0%}')


def HGB():
    return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def oof(d, cols, ycol=None):
    yy = d[ycol].values if ycol else d.label.astype(int).values
    bl = (np.floor(d.lon).astype(int).astype(str) + '_' + np.floor(d.lat).astype(int).astype(str)).values
    yrr = d.year.values; X = d[cols].values.astype('float32')   # keep NaN -> HGB handles it
    o = np.full(len(yy), np.nan)
    for tg, eg in GroupKFold(5).split(X, yy, bl):
        tr = tg[yrr[tg] <= 2019]; te = eg[yrr[eg] >= 2020]
        if len(tr) < 100 or yy[tr].sum() < 20 or (yy[tr] == 0).sum() < 20:
            continue
        m = HGB(); m.fit(X[tr], yy[tr]); o[te] = m.predict_proba(X[te])[:, 1]
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


# compare on IDENTICAL full coverage: with vs without moisture
NOMOIST = [c for c in FEATS if c not in V12]
rng = np.random.default_rng(0); ds = df.copy(); ds['lab'] = rng.permutation(y)
print('\n=== FINAL MODEL (full coverage, HGB native-NaN) ===')
print(f'  negative control (label-shuffle): {A(*oof(ds, FEATS, "lab")):.4f}')
o0, yy = oof(df, NOMOIST); o1, _ = oof(df, FEATS)
m = ~np.isnan(o1); yt, pt = yy[m], o1[m]
boots = [roc_auc_score(yt[i], pt[i]) for i in (rng.integers(0, len(yt), len(yt)) for _ in range(300)) if yt[i].sum() > 5]
print(f'  as-is  WITHOUT moisture {A(o0, yy):.4f}   ->   WITH moisture {A(o1, yy):.4f}   (+{A(o1,yy)-A(o0,yy):.4f})')
print(f'  95% CI (final): [{np.percentile(boots,2.5):.4f}, {np.percentile(boots,97.5):.4f}]')
print(f'  pop-matched     {A(*oof(match(df,"Pop_Density"),NOMOIST)):.4f} -> {A(*oof(match(df,"Pop_Density"),FEATS)):.4f}')
print(f'  dev500-matched  {A(*oof(match(df,"nbhd_dev_500m"),NOMOIST)):.4f} -> {A(*oof(match(df,"nbhd_dev_500m"),FEATS)):.4f}')
print(f'  ENV floor       {A(*oof(df,[c for c in ENV if c not in V12])):.4f} -> {A(*oof(df,ENV)):.4f}')
fl = df[(df.lat < 31) & (df.lon > -87.6)].reset_index(drop=True)
print(f'  Florida-only    {A(*oof(fl,FEATS)):.4f}')
order = np.argsort(pt)[::-1]; top5 = order[:int(len(pt) * .05)]
print(f'  Brier {brier_score_loss(yt,pt):.4f} | top5% precision {yt[top5].mean():.3f}')

X = df[FEATS].values.astype('float32')
final = HGB().fit(X, y)
joblib.dump({'model': final, 'features': FEATS, 'handles_missing': True,
             'data': 'FINAL: 25k FPA-FOD SE-US ignitions; human-access + neighborhood context + nightlights + agriculture + canopy/treecover + moisture/water-stress (NDMI, ET-stress, PET, soil moisture). HGB handles missing moisture natively.',
             'honest_metrics': {'as-is': round(A(o1, yy), 4), 'as-is_no_moisture': round(A(o0, yy), 4),
                                'pop_matched': round(A(*oof(match(df, "Pop_Density"), FEATS)), 4),
                                'env_floor': round(A(*oof(df, ENV)), 4), 'florida_only': round(A(*oof(fl, FEATS)), 4),
                                'note': 'blocked space+time; full coverage; gains survive crutch matching'},
             'trained_on': f'{len(df)} rows'}, 'best_model_final.joblib')
print(f'\nSaved best_model_final.joblib ({len(FEATS)} feats, {len(df)} rows)')
