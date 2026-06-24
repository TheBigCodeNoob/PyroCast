"""Is PyroCast accurate enough to deploy as a FLORIDA tool? Measures Florida-only accuracy
several ways, and tests whether a Florida-SPECIALIZED model beats the SE-wide one."""
import glob, warnings, numpy as np, pandas as pd
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
vm = _safe('Training Data Florida/v12_moisture/*.csv')
v13 = _safe('Training Data Florida/v13_human/*.csv')
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']; MOIST = ['ndmi', 'smap_root', 'lst_day', 'pet']
for d in (ve, vh, vm, v13):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
vm['et_stress'] = vm.et / (vm.pet + 1)
df = (ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k')
        .merge(vm[['k'] + MOIST + ['et_stress']].drop_duplicates('k'), on='k', how='left')
        .merge(v13[['k', 'built']].dropna().drop_duplicates('k'), on='k', how='left')).reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
FEATS = [c for c in df.columns if c not in ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']]
# Florida region (peninsula + panhandle)
FL = (df.lat < 31.0) & (df.lon > -87.6) & (df.lon < -79.8)
print(f'Florida rows: {int(FL.sum())} (fire {int((df[FL].label==1).sum())}, human {int((df[FL & (df.cause==1)].shape[0]))}, lightning {int((df[FL & (df.cause==0)].shape[0]))})')
print(f'rest-of-SE rows: {int((~FL).sum())}')


def MK():
    return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def blocked_oof(sub_train, sub_test, deg=1.0):
    """train on sub_train rows (<=2019), test on sub_test rows (2020), leave-block-out."""
    y = df.label.astype(int).values
    block = (np.floor(df.lon / deg).astype(int).astype(str) + '_' + np.floor(df.lat / deg).astype(int).astype(str)).values
    yr = df.year.values; X = df[FEATS].values.astype('float32'); oof = np.full(len(y), np.nan)
    for tg, eg in GroupKFold(5).split(X, y, block):
        tr = tg[(yr[tg] <= 2019) & sub_train[tg]]; te = eg[(yr[eg] >= 2020) & sub_test[eg]]
        if len(tr) < 100 or y[tr].sum() < 20 or (y[tr] == 0).sum() < 20 or len(te) < 30:
            continue
        m = MK(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof, y


def report(oof, y, mask, name):
    m = (~np.isnan(oof)) & mask
    if m.sum() < 30 or y[m].sum() < 10:
        print(f'  {name}: too few'); return
    yt, pt = y[m], oof[m]; auc = roc_auc_score(yt, pt)
    order = np.argsort(pt)[::-1]
    p5 = yt[order[:max(1, int(len(pt) * .05))]].mean(); p10 = yt[order[:max(1, int(len(pt) * .1))]].mean()
    r20 = yt[order[:max(1, int(len(pt) * .2))]].sum() / yt.sum()
    print(f'  {name:<34s} AUC {auc:.4f} | top5% prec {p5:.2f} top10% prec {p10:.2f} recall@20% {r20:.2f} | Brier {brier_score_loss(yt,pt):.3f} (n={m.sum()})')


allmask = np.ones(len(df), bool)
print('\n=== (1) Current SE-WIDE model, evaluated in Florida ===')
oof, y = blocked_oof(allmask, allmask)
report(oof, y, FL.values, 'SE-trained, FL-tested')
report(oof, y, (~FL).values, 'SE-trained, rest-of-SE-tested')

print('\n=== (2) FLORIDA-SPECIALIZED model (train FL only, test FL) ===')
oof_fl, _ = blocked_oof(FL.values, FL.values, deg=0.7)
report(oof_fl, y, FL.values, 'FL-only-trained, FL-tested')

print('\n=== (3) SE+FL-weighted (train all, but does FL benefit from SE breadth?) already = (1) ===')
print('\n=== verdict guide: AUC>0.75 + top5% prec>0.7 = usable risk-prioritization tool ===')
