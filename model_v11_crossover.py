"""
v11b eval: does adding case-crossover negatives (same place, non-fire day) help?
Combines v10b (positives + random-location negatives, cause -1) with v11cross
(same-location 1yr-prior negatives, cause -2). Reports three tasks:
  WHERE  : positives vs random-location negs  (the v10b task; compare to 0.81)
  WHEN   : positives vs same-location crossover negs  (the temporal task)
  COMBINED: positives vs ALL negs
Uses spatial leave-block-out (so all neg types are in test) for WHEN/COMBINED, and
the strict space+time (train<=2019/test 2020) for the WHERE headline.
"""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

v10b = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True)
cross = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11cross/*.csv')], ignore_index=True)
df = pd.concat([v10b, cross], ignore_index=True).dropna().reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
print(f'rows={len(df)}: pos={int((df.label==1).sum())} random-neg={int((df.cause==-1).sum())} crossover-neg={int((df.cause==-2).sum())}')
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values
X = np.nan_to_num(df[FEATS].values.astype('float32'))


def MK():
    return HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def oof_spatial(train_mask):
    """leave-block-out, train on train_mask rows only, OOF over all."""
    oof = np.full(len(y), np.nan)
    for tg, eg in GroupKFold(5).split(X, y, block):
        tr = tg[train_mask[tg]]
        if len(tr) < 100 or y[tr].sum() < 20 or (y[tr] == 0).sum() < 20:
            continue
        m = MK(); m.fit(X[tr], y[tr]); oof[eg] = m.predict_proba(X[eg])[:, 1]
    return oof


def auc_sub(oof, sub):
    mask = (~np.isnan(oof)) & sub
    return roc_auc_score(y[mask], oof[mask]), int(mask.sum())


pos = (df.label == 1).values; rnd = (df.cause == -1).values; cro = (df.cause == -2).values
# Train on ALL negative types (the combined model), spatial leave-block-out
oof = oof_spatial(np.ones(len(y), bool))
print('\n=== combined model (trained on pos + random + crossover), spatial leave-block-out ===')
a, n = auc_sub(oof, pos | rnd); print(f'  WHERE   (pos vs random-neg)    {a:.4f}  (n={n})')
a, n = auc_sub(oof, pos | cro); print(f'  WHEN    (pos vs crossover-neg) {a:.4f}  (n={n})  <- the new temporal task')
a, n = auc_sub(oof, pos | rnd | cro); print(f'  COMBINED(pos vs all negs)      {a:.4f}  (n={n})')

# strict space+time (train<=2019 -> test 2020), WHERE headline (crossover negs are <=2019 so only in train)
oof2 = np.full(len(y), np.nan)
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20:
        continue
    m = MK(); m.fit(X[tr], y[tr]); oof2[te] = m.predict_proba(X[te])[:, 1]
a, n = auc_sub(oof2, pos | rnd); print(f'\n=== space+time headline (train<=2019/test2020) ===\n  WHERE pos vs random-neg: {a:.4f} (n={n})  [v10b was 0.81; did crossover-augmented training change it?]')
