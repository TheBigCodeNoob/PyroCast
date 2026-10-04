"""What is holding Florida back (0.79 vs 0.86 SE)? Decompose the error to pick the highest-
leverage fix. Reports: AUC by cause, lightning share, where-vs-when, per-subregion, calibration,
feature importance (FL-specific), and a LEARNING CURVE (does more data still help?)."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def _safe(g):
    out = [pd.read_csv(c) for c in glob.glob(g)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


ve = _safe('Training Data Florida/v11e/*.csv').dropna()
vh = _safe('Training Data Florida/v11h_canopy/*.csv').dropna()
vm = _safe('Training Data Florida/v12_moisture/*.csv')
v13 = _safe('Training Data Florida/v13_human/*.csv')
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']; MOIST = ['ndmi', 'smap_root', 'lst_day', 'pet']
for d in (ve, vh, vm, v13):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
vm['et_stress'] = vm.et / (vm.pet + 1)
df = (ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k')
        .merge(vm[['k'] + MOIST + ['et_stress']].drop_duplicates('k'), on='k', how='left')
        .merge(v13[['k', 'built']].dropna().drop_duplicates('k'), on='k', how='left'))
df = df[(df.lat < 31.0) & (df.lon > -87.6) & (df.lon < -79.8)].reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
FEATS = [c for c in df.columns if c not in ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']]
y = df.label.astype(int).values
block = (np.floor(df.lon / 0.6).astype(int).astype(str) + '_' + np.floor(df.lat / 0.6).astype(int).astype(str)).values
yr = df.year.values; X = np.nan_to_num(df[FEATS].values.astype('float32'))


def MK():
    return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def blocked_oof(Xin, sub=None):
    oof = np.full(len(y), np.nan)
    for tg, eg in GroupKFold(5).split(Xin, y, block):
        tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
        if sub is not None:
            tr = tr[sub[tr]]
        if len(tr) < 80 or y[tr].sum() < 15 or (y[tr] == 0).sum() < 15:
            continue
        m = MK(); m.fit(Xin[tr], y[tr]); oof[te] = m.predict_proba(Xin[te])[:, 1]
    return oof


print('=' * 58); print(f'FLORIDA DIAGNOSIS  rows={len(df)} fire={int(y.sum())}'); print('=' * 58)
fires = df[df.label == 1]
print(f'\n[cause mix] human {int((fires.cause==1).sum())} ({(fires.cause==1).mean()*100:.0f}%) | '
      f'lightning {int((fires.cause==0).sum())} ({(fires.cause==0).mean()*100:.0f}%)')

oof = blocked_oof(X); m = ~np.isnan(oof); ca = df.cause.values
print(f'\n[overall FL AUC] {roc_auc_score(y[m], oof[m]):.4f}')
hm = m & ((ca == 1) | (y == 0)); lm = m & ((ca == 0) | (y == 0))
print(f'  human-fire AUC    {roc_auc_score(y[hm], oof[hm]):.4f}')
print(f'  lightning-fire AUC{roc_auc_score(y[lm], oof[lm]):.4f}  <- if low, lightning is the drag')

print('\n[where vs when]')
# where = fire vs random negs (default). when signal ~ month AUC spread
for mo_name, mm in [('Jan (dry)', df.month == 1), ('Apr (peak)', df.month == 4), ('Jul (wet/lightning)', df.month == 7)]:
    s = m & mm.values
    if y[s].sum() > 10 and (y[s] == 0).sum() > 10:
        print(f'  {mo_name:<20s} AUC {roc_auc_score(y[s], oof[s]):.4f} (n={int(s.sum())})')

print('\n[calibration] pred vs observed (are probs meaningful?)')
yt, pt = y[m], oof[m]
for lo in [0, .2, .4, .6, .8]:
    b = (pt >= lo) & (pt < lo + .2)
    if b.sum() > 20:
        print(f'  pred [{lo:.1f}-{lo+.2:.1f}] -> observed {yt[b].mean():.2f} (n={int(b.sum())})')

print('\n[learning curve] does MORE DATA still help? (subsample train, blocked)')
rng = np.random.default_rng(0)
for frac in [0.25, 0.5, 0.75, 1.0]:
    aucs = []
    for tg, eg in GroupKFold(5).split(X, y, block):
        tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
        if len(tr) < 80:
            continue
        keep = rng.choice(tr, max(50, int(len(tr) * frac)), replace=False)
        if y[keep].sum() < 15 or (y[keep] == 0).sum() < 15:
            continue
        mm = MK(); mm.fit(X[keep], y[keep]); p = mm.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    print(f'  {int(frac*100):>3d}% of train -> AUC {np.mean(aucs):.4f}  (slope up at 100% => more data helps)')

print('\n[FL feature importance] top 10')
trm = (df.lon < df.lon.median()).values
mdl = MK().fit(X[trm], y[trm])
pi = permutation_importance(mdl, X[~trm], y[~trm], n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
for i in np.argsort(pi.importances_mean)[::-1][:10]:
    print(f'  {FEATS[i]:<16s} {pi.importances_mean[i]:+.4f}')
