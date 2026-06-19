"""
The operational WHERE x WHEN product. Two specialist models:
  WHERE model  : fire vs random-LOCATION negatives   -> P(this place is fire-prone)
  WHEN model   : fire vs same-place-other-DAY negs    -> P(this day is dangerous | place)
Risk(place, day) = P(where) x P(when). Test whether the product beats either specialist
on the full task (rank fires above BOTH wrong-place and wrong-time negatives).
Uses the 10k v10b points + v11cross crossover negatives (same points, consistent).
"""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

v10 = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True)
cross = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11cross/*.csv')], ignore_index=True)
df = pd.concat([v10, cross], ignore_index=True).dropna().reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
pos = (df.label == 1).values; rnd = (df.cause == -1).values; cro = (df.cause == -2).values
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
X = np.nan_to_num(df[FEATS].values.astype('float32'))
print(f'pos={pos.sum()} random-neg={rnd.sum()} crossover-neg={cro.sum()}')


def MK():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)


def oof(train_mask):
    """spatial leave-block-out; train only on rows in train_mask; predict all held-out."""
    o = np.full(len(y), np.nan)
    for tg, eg in GroupKFold(5).split(X, y, block):
        tr = tg[train_mask[tg]]
        if len(tr) < 100 or y[tr].sum() < 20 or (y[tr] == 0).sum() < 20:
            continue
        m = MK(); m.fit(X[tr], y[tr]); o[eg] = m.predict_proba(X[eg])[:, 1]
    return o


where_p = oof(pos | rnd)   # trained vs wrong-PLACE
when_p = oof(pos | cro)    # trained vs wrong-TIME
prod = where_p * when_p


def auc_on(score, sub):
    m = (~np.isnan(score)) & sub
    return roc_auc_score(y[m], score[m]), int(m.sum())


print('\nEvaluated on the FULL task (fire vs ALL negatives — wrong place AND wrong time):')
for name, s in [('WHERE model alone', where_p), ('WHEN model alone', when_p), ('WHERE x WHEN product', prod)]:
    a, n = auc_on(s, pos | rnd | cro); print(f'  {name:<22s} {a:.4f}  (n={n})')
print('\nFor reference, each specialist on its OWN task:')
a, _ = auc_on(where_p, pos | rnd); print(f'  WHERE on where-task (vs wrong place) {a:.4f}')
a, _ = auc_on(when_p, pos | cro); print(f'  WHEN  on when-task  (vs wrong time)  {a:.4f}')
print('\n-> If the product wins the full task, the two specialists capture different,')
print('   complementary signals (a place can be fire-prone AND a day can be dangerous).')
