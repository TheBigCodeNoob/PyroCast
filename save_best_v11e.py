"""Save new best model: v11e (25k positives, consolidated winning features).
Confirmed on fresh untuned data: 0.840 as-is / 0.803 pop-matched / 0.764 env-floor."""
import glob, numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11e/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
X = np.nan_to_num(df[FEATS].values.astype('float32'))
model = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(X, y)
joblib.dump({'model': model, 'features': FEATS,
             'data': 'v11e: 25k FPA-FOD SE-US ignitions + consolidated winning features (human-access + neighborhood fuel context + nightlights)',
             'honest_metrics': {'as-is': 0.840, 'CI': [0.831, 0.849], 'pop_matched': 0.803, 'dev500_matched': 0.782,
                                'no_human_access': 0.795, 'env_floor': 0.764, 'florida_only': [0.749, 0.662],
                                'note': 'confirmed on fresh untuned 25k data; blocked space+time'},
             'trained_on': f'{len(df)} rows (fire={int(y.sum())}, neg={int((y==0).sum())})'},
            'best_model_v11e.joblib')
print(f'Saved best_model_v11e.joblib ({len(FEATS)} feats, {len(df)} rows, fire={int(y.sum())}/neg={int((y==0).sum())})')
