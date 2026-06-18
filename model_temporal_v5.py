"""
Model the v5 rich-temporal features (multi-window weather/drought at each fire point).
Loads CSVs from 'Training Data Florida/v5/', engineers drought-trajectory / deficit /
ratio features, and trains tabular models with leakage-free location-grouped val + CV.
Target: honest AUC >= 0.80.
"""
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

csvs = sorted(glob.glob('Training Data Florida/v5/*.csv'))
if not csvs:
    raise SystemExit('No v5 CSVs found.')
df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
print(f'Loaded {len(df)} rows from {len(csvs)} CSVs. fire={int(df.label.sum())}')
df = df.dropna().reset_index(drop=True)
print(f'After dropna: {len(df)} rows')

groups = (df.lon.round(3).astype(str) + '|' + df.lat.round(3).astype(str)).values
y = df.label.astype(int).values

# Engineered drought/drying dynamics
eps = 0.1
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90
df['pdsi_traj_180'] = df.pdsi_0 - df.pdsi_180
df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['erc_trend'] = df.erc_7 - df.erc_90
df['pr_recent_ratio'] = df.pr_30 / (df.pr_90 + eps)
df['pr_short_ratio'] = df.pr_7 / (df.pr_30 + eps)
df['pr_deficit_90_365'] = df.pr_365 / 4.0 - df.pr_90        # 90d vs quarter of annual
df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50.0

feat_cols = [c for c in df.columns if c not in ('lon', 'lat', 'label')]
X = df[feat_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(f'Features: {X.shape[1]} | locations: {len(set(groups))}')

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
tr, va = next(gss.split(X, y, groups))
Xtr, ytr, gtr, Xva, yva = X[tr], y[tr], groups[tr], X[va], y[va]
print(f'Train {Xtr.shape} ({len(set(gtr))} locs) | Val {Xva.shape} ({len(set(groups[va]))} locs)')
print('=' * 60)


def cv(make):
    oof = np.zeros(len(ytr))
    for a, b in GroupKFold(5).split(Xtr, ytr, gtr):
        m = make()
        if isinstance(m, tuple):
            sc, clf = m; clf.fit(sc.fit_transform(Xtr[a]), ytr[a]); oof[b] = clf.predict_proba(sc.transform(Xtr[b]))[:, 1]
        else:
            m.fit(Xtr[a], ytr[a]); oof[b] = m.predict_proba(Xtr[b])[:, 1]
    return roc_auc_score(ytr, oof)


preds = {}
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=5000, C=0.5).fit(sc.transform(Xtr), ytr)
preds['LR'] = lr.predict_proba(sc.transform(Xva))[:, 1]
print(f'  LR   val={roc_auc_score(yva, preds["LR"]):.4f}  CV={cv(lambda:(StandardScaler(),LogisticRegression(max_iter=5000,C=0.5))):.4f}')

hgb = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63,
        l2_regularization=2.0, min_samples_leaf=30, early_stopping=False, random_state=0).fit(Xtr, ytr)
preds['HGB'] = hgb.predict_proba(Xva)[:, 1]
print(f'  HGB  val={roc_auc_score(yva, preds["HGB"]):.4f}  CV={cv(lambda:HistGradientBoostingClassifier(max_iter=1000,learning_rate=0.03,max_leaf_nodes=63,l2_regularization=2.0,min_samples_leaf=30,early_stopping=False,random_state=0)):.4f}')

rf = RandomForestClassifier(n_estimators=800, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xtr, ytr)
preds['RF'] = rf.predict_proba(Xva)[:, 1]
print(f'  RF   val={roc_auc_score(yva, preds["RF"]):.4f}  CV={cv(lambda:RandomForestClassifier(n_estimators=800,max_features="sqrt",min_samples_leaf=3,n_jobs=-1,random_state=0)):.4f}')

et = ExtraTreesClassifier(n_estimators=800, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xtr, ytr)
preds['ET'] = et.predict_proba(Xva)[:, 1]
print(f'  ET   val={roc_auc_score(yva, preds["ET"]):.4f}')

blend = np.mean([preds['HGB'], preds['RF'], preds['ET']], axis=0)
print(f'  BLEND val={roc_auc_score(yva, blend):.4f}')
best = max(roc_auc_score(yva, blend), *[roc_auc_score(yva, p) for p in preds.values()])
print('=' * 60)
print(f'BEST val AUC: {best:.4f}   {"*** >=0.80 TARGET MET ***" if best>=0.80 else "(target 0.80)"}')

from sklearn.inspection import permutation_importance
pi = permutation_importance(hgb, Xva, yva, n_repeats=5, random_state=0, scoring='roc_auc', n_jobs=-1)
print('\nTop 15 features:')
for i in np.argsort(pi.importances_mean)[::-1][:15]:
    print(f'  {feat_cols[i]:<20s} {pi.importances_mean[i]:+.4f}')
