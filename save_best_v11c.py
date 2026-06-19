"""Save new best model: v10b features + v11c neighborhood-context/nightlights features.
Verified honest gain (survives pop & DistDev matching)."""
import glob, numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier

d10 = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna()
dx = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v11c_extra/*.csv')], ignore_index=True).dropna()
NEWF = ['NightLights', 'nbhd_dev_500m', 'nbhd_forest_2km', 'nbhd_wetland_2km']
for d in (d10, dx):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
df = d10.merge(dx[['k'] + NEWF].drop_duplicates('k'), on='k', how='inner').reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
X = np.nan_to_num(df[FEATS].values.astype('float32'))
model = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(X, y)
joblib.dump({
    'model': model, 'features': FEATS,
    'new_in_v11c': NEWF,
    'data': 'v10b (FPA-FOD SE-US ignitions 2017-2020) + v11c neighborhood-context & nightlights',
    'honest_metrics': {'space+time_AUC': 0.827, 'pop_matched': 0.778, 'DistDev_matched': 0.730,
                       'note': 'gain over v10b survives & grows under crutch matching -> real fuel/landscape signal'},
    'trained_on': f'{len(df)} rows',
}, 'best_model_v11c.joblib')
print(f'Saved best_model_v11c.joblib ({len(FEATS)} features, {len(df)} rows)')
