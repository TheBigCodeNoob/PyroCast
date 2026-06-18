"""
Tabular model search on v4 features (26 channels, temporal trends, ~3200 locations).

Holds out ~15% of LOCATIONS as val (leakage-free), trains LogReg / HGB / RF (+ a
HGB+RF blend) on the rest, and reports val AUC plus a GroupKFold CV AUC. Target:
val/CV AUC >= 0.80 on the honest matched-temporal-negative task.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

d = np.load('v4_features.npz', allow_pickle=True)
X, y, groups = d['X'], d['y'], d['groups']
names = list(d['feature_names'])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Drop zero-variance features
var = X.var(axis=0)
keep = var > 1e-12
X = X[:, keep]
names = [names[i] for i in range(len(names)) if keep[i]]
print(f'Samples: {X.shape[0]} | features: {X.shape[1]} | fire={int(y.sum())} | locations={len(set(groups))}')

# Leakage-free held-out val split by location (~15% of groups)
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
tr_idx, va_idx = next(gss.split(X, y, groups))
Xtr, ytr, gtr = X[tr_idx], y[tr_idx], groups[tr_idx]
Xva, yva = X[va_idx], y[va_idx]
print(f'Train: {Xtr.shape} ({len(set(gtr))} locs) | Val: {Xva.shape} ({len(set(groups[va_idx]))} locs)')
print('=' * 64)


def cv_auc(make, X, y, groups, n=5):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        m = make()
        if isinstance(m, tuple):
            sc, clf = m
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


results = {}

sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=5000, C=0.5).fit(sc.transform(Xtr), ytr)
p_lr = lr.predict_proba(sc.transform(Xva))[:, 1]
results['LogReg'] = (roc_auc_score(yva, p_lr),
                     cv_auc(lambda: (StandardScaler(), LogisticRegression(max_iter=5000, C=0.5)), Xtr, ytr, gtr))

hgb = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.03, max_leaf_nodes=63,
                                     l2_regularization=2.0, min_samples_leaf=30,
                                     early_stopping=False, random_state=0).fit(Xtr, ytr)
p_hgb = hgb.predict_proba(Xva)[:, 1]
results['HistGradientBoosting'] = (roc_auc_score(yva, p_hgb),
                                   cv_auc(lambda: HistGradientBoostingClassifier(
                                       max_iter=800, learning_rate=0.03, max_leaf_nodes=63,
                                       l2_regularization=2.0, min_samples_leaf=30,
                                       early_stopping=False, random_state=0), Xtr, ytr, gtr))

rf = RandomForestClassifier(n_estimators=800, max_features='sqrt', min_samples_leaf=3,
                            n_jobs=-1, random_state=0).fit(Xtr, ytr)
p_rf = rf.predict_proba(Xva)[:, 1]
results['RandomForest'] = (roc_auc_score(yva, p_rf),
                           cv_auc(lambda: RandomForestClassifier(
                               n_estimators=800, max_features='sqrt', min_samples_leaf=3,
                               n_jobs=-1, random_state=0), Xtr, ytr, gtr))

# Blend
p_blend = 0.5 * p_hgb + 0.5 * p_rf
results['HGB+RF blend'] = (roc_auc_score(yva, p_blend), float('nan'))

print('\nMODEL                       val_AUC   grouped_CV_AUC')
print('-' * 64)
for k, (v, c) in results.items():
    print(f'  {k:<26s} {v:.4f}      {c:.4f}')
print('-' * 64)
best = max(results.values(), key=lambda t: t[0])[0]
print(f'  BEST val AUC: {best:.4f}   {"*** >= 0.80 TARGET MET ***" if best >= 0.80 else "(target 0.80)"}')
print('  Reference: v3 honest ceiling ~0.70')

print('\nTop 20 features (permutation importance, HGB on val):')
pi = permutation_importance(hgb, Xva, yva, n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
for i in np.argsort(pi.importances_mean)[::-1][:20]:
    print(f'  {names[i]:<24s} {pi.importances_mean[i]:+.4f}')
