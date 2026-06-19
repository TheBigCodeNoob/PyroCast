"""
THE TOOL (part 2): predict ignition risk on the exported grid and render an operational
risk map. Loads best_model_final.joblib, scores every grid cell, draws a smooth heatmap.
  python render_grid.py
"""
import glob, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11, 'figure.autolayout': True})

GRID_DIR = 'Training Data Florida/grid'
df = pd.concat([pd.read_csv(c) for c in glob.glob(f'{GRID_DIR}/*.csv')], ignore_index=True)
print(f'grid points: {len(df)}')
bundle = joblib.load('best_model_final.joblib')
model, FEATS = bundle['model'], bundle['features']
# engineered features (must match training)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
df['et_stress'] = df.et / (df.pet + 1)
for f in FEATS:
    if f not in df.columns:
        df[f] = np.nan
# ocean mask: no MODIS vegetation = water
land = df.NDVI.notna() & df.Elevation.notna() & (df[['LC_Forest', 'LC_Crop', 'LC_Wetland']].notna().any(axis=1))
X = df[FEATS].values.astype('float32')
df['risk'] = model.predict_proba(X)[:, 1]
df.loc[~land, 'risk'] = np.nan
print(f'land cells {int(land.sum())} | risk: med {df.risk.median():.3f} max {df.risk.max():.3f}')


def heatmap(d, title, fname, figsize):
    piv = d.pivot_table(index='lat', columns='lon', values='risk')
    fig, ax = plt.subplots(figsize=figsize)
    pc = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap='YlOrRd', vmin=0, vmax=1, shading='auto')
    ax.set_xlabel('longitude'); ax.set_ylabel('latitude'); ax.set_aspect(1.15)
    ax.set_title(title, fontweight='bold')
    plt.colorbar(pc, ax=ax, label='ignition risk', shrink=0.8)
    fig.savefig(fname); plt.close(fig)
    print('saved', fname)


heatmap(df, 'PyroCast operational ignition-risk map — SE US, 15 Apr 2020', 'figures/12_operational_map.png', (9.5, 6))
fl = df[(df.lat < 31) & (df.lon > -87.6)]
heatmap(fl, 'PyroCast risk map — Florida, 15 Apr 2020', 'figures/12b_operational_florida.png', (6.6, 7))
df[['lon', 'lat', 'risk']].dropna().sort_values('risk', ascending=False).to_csv('operational_risk_grid.csv', index=False)
print('saved operational_risk_grid.csv (the scored grid)')
