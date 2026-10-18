"""Does 16 extra years of training data (FPA-FOD 2001-2016, 93k rows) beat the deployed
ensemble? Paired A/B under the EXACT headline protocol (GroupKFold 0.6-deg spatial blocks,
train <=2019 / test 2020-only rows): arm A trains on 2017-2019 (the old regime), arm B on
2001-2019, both predict the SAME held-out 2020 rows in the same folds, so the delta is paired.
Saves the full-data model to best_model_fl_ensemble_v2.joblib (does NOT touch the deployed v1).

RESULT (2026-10-18): A=0.7844, B=0.8071, paired delta +0.0227 (95% block CI +0.0105..+0.0346,
P(delta>0)=0.999). Every member improved; lgb most (0.777->0.810).

Known hist-era quirks (accepted, documented): static layers ~2020 vintage; NightLights=0
pre-2012; SMAP constant-0 pre-2015 (export fallback, consistent with inference nan_to_num).
  conda run -n base python train_fl_hist.py
"""
import glob, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']


def _safe(g):
    out = [pd.read_csv(c) for c in glob.glob(g)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def engineer(d):
    d['pdsi_traj_90'] = d.pdsi_0 - d.pdsi_90; d['vpd_trend'] = d.vpd_7 - d.vpd_90
    d['pr_deficit'] = d.pr_365 / 4 - d.pr_90; d['fm100_trend'] = d.fm100_30 - d.fm100_90
    d['dryness'] = d.vpd_30 + d.erc_30 - d.pr_90 / 50
    return d


def load_combined():
    """(df, FEATS, X, y, yr, block, src_cur): 2017-20 merged era + 2001-16 hist era."""
    ve = _safe('Training Data Florida/v11e/*.csv').dropna()
    vh = _safe('Training Data Florida/v11h_canopy/*.csv').dropna()
    vm = _safe('Training Data Florida/v12_moisture/*.csv')
    v13 = _safe('Training Data Florida/v13_human/*.csv')
    CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
    MOIST = ['ndmi', 'smap_root', 'lst_day', 'pet']
    for d in (ve, vh, vm, v13):
        d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
    vm['et_stress'] = vm.et / (vm.pet + 1)
    cur = (ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k')
             .merge(vm[['k'] + MOIST + ['et_stress']].drop_duplicates('k'), on='k', how='left')
             .merge(v13[['k', 'built']].dropna().drop_duplicates('k'), on='k', how='left'))
    cur = cur[(cur.lat < 31.0) & (cur.lon > -87.6) & (cur.lon < -79.8)].reset_index(drop=True)
    cur = engineer(cur)
    feats = [c for c in cur.columns if c not in META]

    hist = _safe('Training Data Florida/FL_hist/*.csv')
    hist = hist[(hist.lat < 31.0) & (hist.lon > -87.6) & (hist.lon < -79.8)].reset_index(drop=True)
    hist['et_stress'] = hist.et / (hist.pet + 1)
    hist = engineer(hist)
    missing = [c for c in feats if c not in hist.columns]
    assert not missing, f'hist export lacks: {missing}'
    cols = ['lon', 'lat', 'label', 'year'] + feats
    hist = hist[cols].drop_duplicates(subset=['lon', 'lat', 'year', 'label'] + [feats[0]])
    df = pd.concat([cur[cols], hist], ignore_index=True)
    src_cur = np.arange(len(df)) < len(cur)
    y = df.label.astype(int).values
    yr = df.year.values
    block = (np.floor(df.lon / 0.6).astype(int).astype(str) + '_' + np.floor(df.lat / 0.6).astype(int).astype(str)).values
    X = np.nan_to_num(df[feats].values.astype('float32'))
    print(f'combined rows={len(df)} (cur={len(cur)}, hist={len(hist)}), fires={int(y.sum())}, feats={len(feats)}', flush=True)
    return df, feats, X, y, yr, block, src_cur


def members():
    return [('hgb', HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)),
            ('rf', RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0)),
            ('lgb', LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1))]


def main():
    df, FEATS, X, y, yr, block, src_cur = load_combined()
    NAMES = [n for n, _ in members()]
    oofA = {n: np.full(len(y), np.nan) for n in NAMES}   # arm A: 2017-2019 train (old regime)
    oofB = {n: np.full(len(y), np.nan) for n in NAMES}   # arm B: 2001-2019 train (new regime)
    for fold, (tg, eg) in enumerate(GroupKFold(5).split(X, y, block), 1):
        te = eg[yr[eg] >= 2020]
        trA = tg[(yr[tg] <= 2019) & src_cur[tg]]
        trB = tg[yr[tg] <= 2019]
        if len(te) == 0 or y[trA].sum() < 20 or (y[trA] == 0).sum() < 20:
            print(f'fold {fold}: skipped', flush=True); continue
        print(f'fold {fold}: trainA={len(trA)} trainB={len(trB)} test2020={len(te)}', flush=True)
        for n, mdl in members():
            mdl.fit(X[trA], y[trA]); oofA[n][te] = mdl.predict_proba(X[te])[:, 1]
        for n, mdl in members():
            mdl.fit(X[trB], y[trB]); oofB[n][te] = mdl.predict_proba(X[te])[:, 1]

    mask = ~np.isnan(oofA['hgb']) & ~np.isnan(oofB['hgb'])
    yt = y[mask]; bt = block[mask]
    ensA = np.mean([oofA[n][mask] for n in NAMES], axis=0)
    ensB = np.mean([oofB[n][mask] for n in NAMES], axis=0)
    print(f'\nheld-out 2020 rows={int(mask.sum())}, fires={int(yt.sum())}, blocks={len(np.unique(bt))}')
    for n in NAMES:
        print(f'  {n}: A(2017-19)={roc_auc_score(yt, oofA[n][mask]):.4f}  B(2001-19)={roc_auc_score(yt, oofB[n][mask]):.4f}')
    aA, aB = roc_auc_score(yt, ensA), roc_auc_score(yt, ensB)
    print(f'ENSEMBLE AUROC: A(2017-19)={aA:.4f}  B(2001-19)={aB:.4f}  delta={aB - aA:+.4f}')
    print(f'AP:             A={average_precision_score(yt, ensA):.4f}  B={average_precision_score(yt, ensB):.4f}')

    # paired block bootstrap: CI of B, and CI of the DELTA under identical resamples
    _, inv = np.unique(bt, return_inverse=True)
    rng = np.random.default_rng(0); nb = inv.max() + 1
    dB, dD = [], []
    for _ in range(1000):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[inv]
        if w[yt == 1].sum() and w[yt == 0].sum():
            b = roc_auc_score(yt, ensB, sample_weight=w)
            dB.append(b); dD.append(b - roc_auc_score(yt, ensA, sample_weight=w))
    print(f'B 95% block-bootstrap CI=[{np.percentile(dB, 2.5):.4f}, {np.percentile(dB, 97.5):.4f}]')
    print(f'delta (B-A) 95% CI=[{np.percentile(dD, 2.5):+.4f}, {np.percentile(dD, 97.5):+.4f}]  P(delta>0)={np.mean(np.array(dD) > 0):.3f}')

    # final model on ALL data (2001-2020) -> v2 bundle (deployed v1 untouched)
    fitted = [m.fit(X, y) or m for _, m in members()]
    joblib.dump({'models': fitted, 'features': FEATS, 'region': 'Florida',
                 'honest_metrics': {'auc': round(float(aB), 4), 'auc_prev_regime': round(float(aA), 4),
                                    'delta_ci': [round(float(np.percentile(dD, 2.5)), 4), round(float(np.percentile(dD, 97.5)), 4)]},
                 'trained_on': f'{len(df)} FL rows 2001-2020 (ensemble {"+".join(NAMES)}), blocked-eval AUROC {aB:.4f}'},
                'best_model_fl_ensemble_v2.joblib')
    print(f'\nSaved best_model_fl_ensemble_v2.joblib (trained on all {len(df)} rows)')


if __name__ == '__main__':
    main()
