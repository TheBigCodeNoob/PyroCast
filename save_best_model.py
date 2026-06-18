"""Save the current best model (v10b FPA-FOD) trained on ALL data, with the exact
feature recipe + validated metrics, so it's reproducible/deployable."""
import glob, json, numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
# --- exact engineered-feature recipe (must match training/eval) ---
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90
df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90
df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
X = np.nan_to_num(df[FEATS].values.astype('float32'))

model = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63,
                                       l2_regularization=2.0, min_samples_leaf=25, random_state=0)
model.fit(X, y)

bundle = {
    'model': model,
    'features': FEATS,
    'engineered': {'pdsi_traj_90': 'pdsi_0 - pdsi_90', 'vpd_trend': 'vpd_7 - vpd_90',
                   'pr_deficit': 'pr_365/4 - pr_90', 'fm100_trend': 'fm100_30 - fm100_90',
                   'dryness': 'vpd_30 + erc_30 - pr_90/50'},
    'data': 'v10b = FPA-FOD SE-US wildfire ignitions 2017-2020 (10k pos / 10k land negs, season-matched)',
    'honest_metrics': {'space+time_AUC': 0.81, 'bootstrap_95CI': [0.796, 0.821],
                       'pop_matched': 0.755, 'DistDev_matched': 0.709,
                       'note': 'blocked leave-spatial-block-out x past->future CV; crutch-free'},
    'trained_on': f'{len(df)} rows, all v10b data',
}
joblib.dump(bundle, 'best_model_v10b.joblib')
print(f'Saved best_model_v10b.joblib ({len(FEATS)} features, trained on {len(df)} rows)')
