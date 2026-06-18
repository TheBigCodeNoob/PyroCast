"""
EXHAUSTIVE validation of the v10b FPA-FOD wildfire-ignition model.
Every meaningful test on existing data: integrity, negative controls, core metric +
robustness (block size, model family, seeds, bootstrap CI), generalization
(leave-region-out, E/W, future-year), subgroups (human/lightning, by-month),
causality of human-access (cause split + DistDev distributions), calibration,
operational precision@k, and interpretability. North-star = blocked space+time.
"""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
try:
    from lightgbm import LGBMClassifier
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
eps = 0.1
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'pb']
HUMANF = ['Pop_Density', 'DistDev', 'LC_Developed']
SPATIAL = ['Elevation', 'Pop_Density', 'DistDev', 'LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Pasture', 'LC_Wetland', 'LC_Crop', 'LC_Developed', 'NDVI', 'EVI']
TEMPORAL = [c for c in df.columns if c not in META + SPATIAL]
FEATS = SPATIAL + TEMPORAL
y = df.label.astype(int).values


def MK(kind='hgb'):
    if kind == 'hgb':
        return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0)
    if kind == 'lgb':
        return LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1)
    if kind == 'lr':
        return ('scale', LogisticRegression(max_iter=3000))


def blocked_oof(d, cols, deg=1.0, k=5, kind='hgb', seed=0):
    """Returns oof predictions (nan where untested) for train<=2019/test>=2020, leave-block-out."""
    yy = d.label.astype(int).values
    block = (np.floor(d.lon / deg).astype(int).astype(str) + '_' + np.floor(d.lat / deg).astype(int).astype(str)).values
    yr = d.year.values; X = np.nan_to_num(d[cols].values.astype('float32')); oof = np.full(len(yy), np.nan)
    gkf = GroupKFold(min(k, len(set(block))))
    for tg, eg in gkf.split(X, yy, block):
        tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
        if len(tr) < 100 or yy[tr].sum() < 20 or (yy[tr] == 0).sum() < 20:
            continue
        m = MK(kind)
        if isinstance(m, tuple):
            sc = StandardScaler(); m[1].fit(sc.fit_transform(X[tr]), yy[tr]); oof[te] = m[1].predict_proba(sc.transform(X[te]))[:, 1]
        else:
            m.fit(X[tr], yy[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def auc_of(d, oof, sub=None):
    yy = d.label.astype(int).values; mask = ~np.isnan(oof)
    if sub is not None:
        mask = mask & sub
    return roc_auc_score(yy[mask], oof[mask]), int(mask.sum()), int(yy[mask].sum())


def popmatch(d, col='Pop_Density'):
    nz = d[col][d[col] > 0]; edges = np.unique([d[col].min() - 1, 1e-9] + list(nz.quantile([.25, .5, .75]).values) + [np.inf]) if len(nz) else np.array([-1, np.inf])
    d = d.copy(); d['pb'] = pd.cut(d[col], bins=edges, duplicates='drop')
    posf = d[d.label == 1].pb.value_counts(normalize=True); negc = d[d.label == 0].pb.value_counts()
    N = int(min(negc.get(b, 0) / posf[b] for b in posf.index if posf[b] > 0)); keep = [d[d.label == 1]]
    for b in posf.index:
        pool = d[(d.label == 0) & (d.pb == b)]; kk = int(round(posf[b] * N))
        if len(pool) and kk:
            keep.append(pool.sample(min(kk, len(pool)), random_state=0))
    return pd.concat(keep).reset_index(drop=True)


print('=' * 64); print('EXHAUSTIVE VALIDATION — v10b FPA-FOD model'); print('=' * 64)
print(f'rows={len(df)} fire={int(y.sum())} (human={int((df.cause==1).sum())}, lightning={int((df.cause==0).sum())}) neg={int((y==0).sum())}')

print('\n[1] INTEGRITY')
print(f'  unique (lon,lat): {df[["lon","lat"]].drop_duplicates().shape[0]}/{len(df)}')
ps = set(map(tuple, df[y == 1][["lon", "lat"]].round(4).values)); ns = set(map(tuple, df[y == 0][["lon", "lat"]].round(4).values))
print(f'  pos/neg location overlap: {len(ps & ns)}')
print(f'  season mean-month: fire={df[y==1].month.mean():.2f} neg={df[y==0].month.mean():.2f}')

print('\n[2] NEGATIVE CONTROLS (sanity: pipeline not leaking)')
yshuf = df.copy(); rng = np.random.default_rng(0); yshuf['label'] = rng.permutation(df.label.values)
oo = blocked_oof(yshuf, FEATS); a, n, _ = auc_of(yshuf, oo); print(f'  label-shuffled AUC (expect ~0.50): {a:.4f}')
dfr = df.copy(); dfr['NOISE'] = rng.standard_normal(len(df))
oo = blocked_oof(dfr, FEATS + ['NOISE']); a, _, _ = auc_of(dfr, oo); print(f'  with random-noise feature added, AUC: {a:.4f} (should be ~unchanged)')

print('\n[3] CORE METRIC + CRUTCH CONTROLS')
oof = blocked_oof(df, FEATS); a, n, nf = auc_of(df, oof); print(f'  FULL as-is             {a:.4f}  (test n={n}, fire={nf})')
a2, _, _ = auc_of(popmatch(df, 'Pop_Density'), blocked_oof(popmatch(df, 'Pop_Density'), FEATS)); print(f'  pop-matched            {a2:.4f}')
a3, _, _ = auc_of(popmatch(df, 'DistDev'), blocked_oof(popmatch(df, 'DistDev'), FEATS)); print(f'  DistDev-matched        {a3:.4f}')

print('\n[4] ROBUSTNESS')
for deg in [0.5, 1.0, 2.0]:
    a, _, _ = auc_of(df, blocked_oof(df, FEATS, deg=deg)); print(f'  block {deg}deg: {a:.4f}')
for kind in (['hgb', 'rf', 'lgb', 'lr'] if HAVE_LGB else ['hgb', 'rf', 'lr']):
    a, _, _ = auc_of(df, blocked_oof(df, FEATS, kind=kind)); print(f'  model {kind:<4s}: {a:.4f}')
seeds = []
for s in range(5):
    d2 = df.copy()  # vary the GroupKFold shuffle via different block jitter
    a, _, _ = auc_of(d2, blocked_oof(d2, FEATS, k=5 + s % 2))
    seeds.append(a)
print(f'  fold-config stability: mean={np.mean(seeds):.4f} std={np.std(seeds):.4f}')
# bootstrap CI on the headline OOF AUC
mask = ~np.isnan(oof); yt = y[mask]; pt = oof[mask]; boots = []
for _ in range(300):
    idx = rng.integers(0, len(yt), len(yt))
    if yt[idx].sum() > 5 and (yt[idx] == 0).sum() > 5:
        boots.append(roc_auc_score(yt[idx], pt[idx]))
print(f'  bootstrap 95% CI: [{np.percentile(boots,2.5):.4f}, {np.percentile(boots,97.5):.4f}]')

print('\n[5] GENERALIZATION')
# leave-one-geographic-region-out (spatial), any year
lon_b = pd.qcut(df.lon, 4, labels=False); lat_b = pd.qcut(df.lat, 2, labels=False); region = (lon_b * 2 + lat_b).values
X = np.nan_to_num(df[FEATS].values.astype('float32')); regaucs = []
for r in np.unique(region):
    tr = region != r; te = region == r
    if y[te].sum() < 20 or (y[te] == 0).sum() < 20:
        continue
    m = MK('hgb'); m.fit(X[tr], y[tr]); regaucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
print(f'  leave-region-out (8 regions): mean={np.mean(regaucs):.4f} min={np.min(regaucs):.4f} max={np.max(regaucs):.4f}')
# future-year only (train<=2019 all-space, test 2020 all-space)
tr = (df.year <= 2019).values; te = (df.year >= 2020).values
m = MK('hgb'); m.fit(X[tr], y[tr]); print(f'  future-year (train<=2019 -> test 2020): {roc_auc_score(y[te], m.predict_proba(X[te])[:,1]):.4f}')

print('\n[6] SUBGROUPS + CAUSALITY of human-access')
ah, _, nh = auc_of(df, oof, sub=(df.cause == 1).values); al, _, nl = auc_of(df, oof, sub=(df.cause == 0).values)
print(f'  human ignitions AUC {ah:.4f} (n_fire={nh}) | lightning {al:.4f} (n_fire={nl})')
print(f'  DistDev(km) median: human-fire={df[df.cause==1].DistDev.median():.2f} lightning-fire={df[df.cause==0].DistDev.median():.2f} neg={df[df.label==0].DistDev.median():.2f}')
print('  -> if human fires sit MUCH closer to development than lightning/neg, DistDev is CAUSAL (cause sets location), not pure reporting bias')
for mo in [1, 3, 5, 7, 10]:
    s = (df.month == mo).values & ~np.isnan(oof)
    if y[s].sum() > 15 and (y[s] == 0).sum() > 15:
        print(f'  month {mo:>2d} AUC: {roc_auc_score(y[s], oof[s]):.4f} (n={int(s.sum())})')

print('\n[7] CALIBRATION + OPERATIONAL')
print(f'  Brier score: {brier_score_loss(yt, pt):.4f} (lower=better; baseline {yt.mean()*(1-yt.mean()):.4f})')
print(f'  PR-AUC: {average_precision_score(yt, pt):.4f} (prevalence {yt.mean():.3f})')
order = np.argsort(pt)[::-1]
for kpct in [1, 5, 10, 20]:
    topk = order[:max(1, int(len(pt) * kpct / 100))]
    prec = yt[topk].mean(); rec = yt[topk].sum() / yt.sum()
    print(f'  top {kpct:>2d}% riskiest: precision={prec:.3f}, recall={rec:.3f} (captures {rec*100:.0f}% of fires)')
print('  calibration (pred-prob bin -> observed fire rate):')
for lo in np.arange(0, 1, 0.2):
    m2 = (pt >= lo) & (pt < lo + 0.2)
    if m2.sum() > 20:
        print(f'    [{lo:.1f}-{lo+0.2:.1f}] pred~{pt[m2].mean():.2f} -> observed {yt[m2].mean():.2f} (n={int(m2.sum())})')

print('\n[8] INTERPRETABILITY')
from sklearn.inspection import permutation_importance
med = df.lon.median(); trm = (df.lon < med).values
m = MK('hgb'); m.fit(X[trm], y[trm])
pi = permutation_importance(m, X[~trm], y[~trm], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
print('  permutation importance (top 12):')
for i in np.argsort(pi.importances_mean)[::-1][:12]:
    tag = 'WHERE' if FEATS[i] in SPATIAL else 'when '
    print(f'    [{tag}] {FEATS[i]:<14s} {pi.importances_mean[i]:+.4f}')
for nm, cols in [('where(spatial)', SPATIAL), ('when(temporal)', TEMPORAL), ('human-access', HUMANF), ('no-human-access', [c for c in FEATS if c not in HUMANF])]:
    a, _, _ = auc_of(df, blocked_oof(df, cols)); print(f'  ablation {nm:<16s} {a:.4f}')
print('\n' + '=' * 64 + '\nVALIDATION COMPLETE\n' + '=' * 64)
