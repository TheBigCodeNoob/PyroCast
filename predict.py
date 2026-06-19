"""
Score new place-days with a saved PyroCast model.

Usage:
  python predict.py <features.csv> [model.joblib]

<features.csv> must contain the raw feature columns the model was trained on (same names
as the Dataget_*.py exports — e.g. pr_90, vpd_30, DistDev, canopy_ht, ...). This script
re-creates the engineered features, runs the model, and writes <features>_scored.csv with
a 'risk' column (0-1 predicted ignition probability), sorted high to low.
"""
import sys, joblib, numpy as np, pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else 'example_points.csv'
model_path = sys.argv[2] if len(sys.argv) > 2 else 'best_model_v11h.joblib'

bundle = joblib.load(model_path)
model, FEATS = bundle['model'], bundle['features']
df = pd.read_csv(src)

# re-create the engineered features (must match training)
eng = {
    'pdsi_traj_90': lambda d: d.pdsi_0 - d.pdsi_90,
    'vpd_trend': lambda d: d.vpd_7 - d.vpd_90,
    'pr_deficit': lambda d: d.pr_365 / 4 - d.pr_90,
    'fm100_trend': lambda d: d.fm100_30 - d.fm100_90,
    'dryness': lambda d: d.vpd_30 + d.erc_30 - d.pr_90 / 50,
}
for name, fn in eng.items():
    if name in FEATS and name not in df.columns:
        try:
            df[name] = fn(df)
        except Exception:
            df[name] = 0.0

missing = [f for f in FEATS if f not in df.columns]
if missing:
    print(f'WARNING: {len(missing)} features missing, filling 0: {missing[:8]}{"..." if len(missing)>8 else ""}')
    for f in missing:
        df[f] = 0.0

X = np.nan_to_num(df[FEATS].values.astype('float32'))
df['risk'] = model.predict_proba(X)[:, 1]
out = src.rsplit('.', 1)[0] + '_scored.csv'
cols = [c for c in ['lon', 'lat', 'year', 'doy', 'risk'] if c in df.columns] + [c for c in df.columns if c not in FEATS and c not in ['lon', 'lat', 'year', 'doy', 'risk']]
df.sort_values('risk', ascending=False).to_csv(out, index=False)
print(f'scored {len(df)} points -> {out}')
print(f'risk: min {df.risk.min():.3f}  median {df.risk.median():.3f}  max {df.risk.max():.3f}')
print('model honest metrics:', bundle.get('honest_metrics', {}).get('note', ''))
