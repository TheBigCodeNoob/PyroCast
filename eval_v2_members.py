"""Pick the deployable v2 ensemble composition. The naive v2 (RF @ min_samples_leaf=3 on 99k
rows) is 445MB - over GitHub's 100MB limit and too big for Railway RAM. Re-evaluate arm B
(train 2001-2019 / test blocked 2020) with lean members and score every composition from the
saved out-of-fold predictions.  conda run -n base python eval_v2_members.py"""
import warnings, numpy as np, joblib
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
from train_fl_hist import load_combined                # reuse the exact combined dataset
df, FEATS, X, y, yr, block, _src = load_combined()


def members():
    return [('hgb', HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)),
            ('lgb', LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1)),
            ('rf50', RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=50, n_jobs=-1, random_state=0))]


NAMES = [n for n, _ in members()]
oof = {n: np.full(len(y), np.nan) for n in NAMES}
for fold, (tg, eg) in enumerate(GroupKFold(5).split(X, y, block), 1):
    te = eg[yr[eg] >= 2020]; tr = tg[yr[tg] <= 2019]
    if len(te) == 0:
        continue
    print(f'fold {fold}: train={len(tr)} test={len(te)}', flush=True)
    for n, mdl in members():
        mdl.fit(X[tr], y[tr]); oof[n][te] = mdl.predict_proba(X[te])[:, 1]

mask = ~np.isnan(oof['hgb'])
yt, bt = y[mask], block[mask]
np.savez('oof_v2_members.npz', mask=mask, y=yt, block=bt, **{n: oof[n][mask] for n in NAMES})


def ci(scores):
    _, inv = np.unique(bt, return_inverse=True)
    rng = np.random.default_rng(0); nb = inv.max() + 1; est = []
    for _ in range(1000):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[inv]
        if w[yt == 1].sum() and w[yt == 0].sum():
            est.append(roc_auc_score(yt, scores, sample_weight=w))
    return np.percentile(est, [2.5, 97.5])


print(f'\nheld-out 2020 rows={int(mask.sum())}, fires={int(yt.sum())}')
combos = {'hgb': ['hgb'], 'lgb': ['lgb'], 'rf50': ['rf50'],
          'hgb+lgb': ['hgb', 'lgb'], 'hgb+lgb+rf50': ['hgb', 'lgb', 'rf50']}
for label, ns in combos.items():
    e = np.mean([oof[n][mask] for n in ns], axis=0)
    lo, hi = ci(e)
    print(f'  {label:14s} AUROC={roc_auc_score(yt, e):.4f}  CI=[{lo:.4f},{hi:.4f}]  AP={average_precision_score(yt, e):.4f}')
