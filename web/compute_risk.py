"""Score a grid of FL points and write the demo's data files. Run by the scheduler after a
fresh GEE export, or on any existing grid CSV dir.
  python compute_risk.py [grid_csv_dir]
Outputs: web/data/fl_risk.json (map metadata + tier legend), fl_risk.png (ignition tiers),
fl_priority.png (vulnerable-places tiers), fl_grid_full.csv (full features + calibrated columns,
for on-demand explanations)."""
import sys, glob, pathlib, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pyrocast_core as core
import calibrate

# discrete operational tiers (Minimal..Extreme), colored by score percentile boundaries
_TIER_CMAP = ListedColormap(calibrate.TIER_COLORS)
_TIER_NORM = BoundaryNorm([0.0] + calibrate.TIER_PCT + [1.0001], _TIER_CMAP.N)


def render_tiers(df, col, path):
    """Interpolate the 0-1 percentile field `col` onto a smooth high-res raster, then color it by
    discrete tier (Minimal..Extreme) so the map reads as operational danger bands with clean
    contours (not blocky). Returns map bounds [[lat0,lon0],[lat1,lon1]]."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
    pts = df[['lon', 'lat']].values
    lon0, lon1 = float(df.lon.min()), float(df.lon.max())
    lat0, lat1 = float(df.lat.min()), float(df.lat.max())
    W = 1400
    H = int(round(W * (lat1 - lat0) / (lon1 - lon0)))
    gx, gy = np.meshgrid(np.linspace(lon0, lon1, W), np.linspace(lat0, lat1, H))
    gz = griddata(pts, df[col].values, (gx, gy), method='linear').ravel()
    dist, _ = cKDTree(pts).query(np.column_stack([gx.ravel(), gy.ravel()]))
    gz[dist > 0.04] = np.nan                                       # >~one cell from data = ocean
    gz = gz.reshape(H, W)
    valid = ~np.isnan(gz)
    land = binary_opening(binary_closing(valid, np.ones((3, 3)), iterations=2), np.ones((3, 3)), iterations=2)
    sig = 3
    wsm = gaussian_filter(valid.astype(float), sigma=sig)
    sm = gaussian_filter(np.where(valid, gz, 0.0), sigma=sig) / np.where(wsm > 1e-3, wsm, 1.0)
    sm[~land] = np.nan
    sm = sm[::-1, :]                                               # row 0 = north
    rgba = _TIER_CMAP(_TIER_NORM(np.clip(np.nan_to_num(sm, nan=0.0), 0, 1)))
    rgba[..., 3] = np.where(np.isnan(sm), 0.0, 0.88)
    plt.imsave(str(path), rgba)
    return [[lat0, lon0], [lat1, lon1]]

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / 'data'
ROOT = HERE.parent
DEFAULT_DIRS = [ROOT / 'Training Data Florida' / 'FLgrid', ROOT / 'Training Data Florida' / 'grid']
FPAFOD = ROOT / 'fpafod_se.csv'


def load_grid(grid_dir):
    files = glob.glob(str(grid_dir) + '/*.csv')
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(c) for c in files], ignore_index=True)
    df = df[(df.lat < 31.05) & (df.lat > 24.4) & (df.lon > -87.7) & (df.lon < -79.8)]
    if 'NDVI' in df.columns:
        df = df[df.NDVI.notna()]
    df = df.drop_duplicates(subset=['lon', 'lat']).reset_index(drop=True)
    if 'pet' in df.columns:
        for f in ['et', 'pet']:
            if f in df.columns:
                df.loc[df.pet <= 0, f] = np.nan
    return df


def main(grid_dir=None):
    if grid_dir:
        dirs = [pathlib.Path(grid_dir)]
    else:
        dirs = [d for d in DEFAULT_DIRS if glob.glob(str(d) + '/*.csv')]
    df = pd.DataFrame()
    for d in dirs:
        df = load_grid(d)
        if len(df) > 50:
            print(f'using grid: {d} ({len(df)} FL land points)'); break
    if len(df) < 10:
        print('not enough grid data yet'); return
    DATA.mkdir(parents=True, exist_ok=True)
    df['risk_raw'] = core.score(df)
    df['risk'] = df['risk_raw'].rank(pct=True).round(4)             # score percentile (tier basis)
    core.set_medians(df)

    # grid spacing + calibration against real ignition history (FPA-FOD)
    lons = np.sort(df.lon.unique()); gaps = np.diff(lons); gaps = gaps[gaps > 1e-6]
    cell = float(round(float(np.min(gaps)), 3)) if len(gaps) else 0.04
    df, calib = calibrate.calibrate(df, FPAFOD, cell)
    df['prisk'] = df['priority_score'].rank(pct=True).round(4)      # priority percentile (vuln. tiers)

    bounds = render_tiers(df, 'risk', DATA / 'fl_risk.png')         # ignition-likelihood tiers
    render_tiers(df, 'prisk', DATA / 'fl_priority.png')             # vulnerable-places tiers

    from datetime import datetime
    meta = {'model': core.load().get('_path'), 'n_points': int(len(df)), 'bounds': bounds,
            'cell': cell, 'computed': datetime.now().strftime('%Y-%m-%d'),
            'layers': {
                'ignition': {'png': 'fl_risk.png', 'title': 'Ignition likelihood',
                             'unit': 'expected ignitions per 100 km2 per year'},
                'priority': {'png': 'fl_priority.png', 'title': 'People & property at risk',
                             'unit': 'ignition risk x people & property exposed'}},
            'calib': calib}
    json.dump({'meta': meta}, open(DATA / 'fl_risk.json', 'w'))
    df.to_csv(DATA / 'fl_grid_full.csv', index=False)
    print('wrote fl_risk.json + fl_risk.png + fl_priority.png + fl_grid_full.csv')
    print('  baseline', calib['baseline_rate_100km2_yr'], 'ign/100km2/yr | Extreme cells:',
          calib['tiers'][5]['n_cells'])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
