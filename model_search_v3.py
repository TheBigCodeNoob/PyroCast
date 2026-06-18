"""
Tabular model search on the rich v3 features (from feature_extract_v3.py).

Reports AUC on the held-out val set AND a leakage-free GroupKFold CV estimate on
the train set (grouped by physical location), because the val set has only ~45
unique locations and is noisy. Goal: beat the v3 audit's mean-only LR (0.726) and
the CNN (0.693), and find out how high existing-data features can reach.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

d = np.load('v3_features.npz', allow_pickle=True)
Xtr, ytr, gtr = d['X_train'], d['y_train'], d['groups_train']
Xva, yva, gva = d['X_val'], d['y_val'], d['groups_val']
names = list(d['feature_names'])

# Clean
Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
Xva = np.nan_to_num(Xva, nan=0.0, posinf=0.0, neginf=0.0)

# Drop zero-variance features (e.g., dead Pop_Density channel)
var = Xtr.var(axis=0)
keep = var > 1e-12
dropped = [names[i] for i in range(len(names)) if not keep[i]]
Xtr, Xva = Xtr[:, keep], Xva[:, keep]
names = [names[i] for i in range(len(names)) if keep[i]]
print(f'Features: {Xtr.shape[1]} kept, {len(dropped)} dropped ({dropped[:6]}{"..." if len(dropped)>6 else ""})')
print(f'Train: {Xtr.shape}, fire={int(ytr.sum())}/{len(ytr)} | Val: {Xva.shape}, fire={int(yva.sum())}/{len(yva)}')
print(f'Train unique locations: {len(set(gtr))} | Val unique locations: {len(set(gva))}')
print('=' * 64)


def grouped_cv_auc(make_model, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = make_model()
        if isinstance(m, tuple):  # (scaler, model) for LR
            sc, clf = m
            Xs = sc.fit_transform(X[tr])
            clf.fit(Xs, y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


results = {}

# 1) Logistic regression on rich features
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(Xtr), ytr)
results['LogReg (rich feats)'] = (
    roc_auc_score(yva, lr.predict_proba(sc.transform(Xva))[:, 1]),
    grouped_cv_auc(lambda: (StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)), Xtr, ytr, gtr),
)

# 2) HistGradientBoosting (sklearn's built-in GBDT)
hgb = HistGradientBoostingClassifier(
    max_iter=600, learning_rate=0.04, max_depth=None, max_leaf_nodes=31,
    l2_regularization=1.0, early_stopping=False, random_state=0,
).fit(Xtr, ytr)
results['HistGradientBoosting'] = (
    roc_auc_score(yva, hgb.predict_proba(Xva)[:, 1]),
    grouped_cv_auc(lambda: HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=0), Xtr, ytr, gtr),
)

# 3) Random forest
rf = RandomForestClassifier(n_estimators=600, max_features='sqrt', min_samples_leaf=2,
                            n_jobs=-1, random_state=0).fit(Xtr, ytr)
results['RandomForest'] = (
    roc_auc_score(yva, rf.predict_proba(Xva)[:, 1]),
    grouped_cv_auc(lambda: RandomForestClassifier(
        n_estimators=600, max_features='sqrt', min_samples_leaf=2, n_jobs=-1, random_state=0),
        Xtr, ytr, gtr),
)

print('\nMODEL                       val_AUC   grouped_CV_AUC')
print('-' * 64)
for k, (vauc, cvauc) in results.items():
    print(f'  {k:<26s} {vauc:.4f}      {cvauc:.4f}')
print('-' * 64)
print('  Reference: v3 CNN val=0.693 | v3 audit mean-only LR=0.726')

# Feature importance from HGB via permutation on val (top 25)
from sklearn.inspection import permutation_importance
print('\nTop 20 features (permutation importance, HGB on val):')
pi = permutation_importance(hgb, Xva, yva, n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
order = np.argsort(pi.importances_mean)[::-1][:20]
for i in order:
    print(f'  {names[i]:<22s} {pi.importances_mean[i]:+.4f}')
