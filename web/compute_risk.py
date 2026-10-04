"""Score a grid of FL points and write the demo's data files. Run by the scheduler after a
fresh GEE export, or on any existing grid CSV dir.
  python compute_risk.py [grid_csv_dir]
Outputs: web/data/fl_risk.json (compact, for the map) + web/data/fl_grid_full.csv (full
features, for on-demand explanations)."""
import sys, glob, pathlib, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pyrocast_core as core

# psychology-tuned scale: green dominates the low/middle so average areas read CALM;
# amber/orange/red reserved for the genuine top of the distribution.
PSYCH = LinearSegmentedColormap.from_list('psych', [
    (0.00, '#1a7d3a'), (0.50, '#74bd57'), (0.66, '#cdd451'),
    (0.79, '#f3c63f'), (0.90, '#ee7e2b'), (1.00, '#d8342a')])


def render_raster(df, path):
    """Interpolate the scattered grid points onto a clean high-res raster (no pivot holes, no coast
    warp), mask out anything farther than ~one cell from real data (ocean), lightly smooth, then
    re-rank to percentile for the psychology-tuned green-heavy colors. Returns map bounds."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
    pts = df[['lon', 'lat']].values
    lon0, lon1 = float(df.lon.min()), float(df.lon.max())
    lat0, lat1 = float(df.lat.min()), float(df.lat.max())
    W = 1000
    H = int(round(W * (lat1 - lat0) / (lon1 - lon0)))
    gx, gy = np.meshgrid(np.linspace(lon0, lon1, W), np.linspace(lat0, lat1, H))
    gz = griddata(pts, df.risk_raw.values, (gx, gy), method='linear').ravel()
    dist, _ = cKDTree(pts).query(np.column_stack([gx.ravel(), gy.ravel()]))
    gz[dist > 0.04] = np.nan                                       # >~one cell from data = ocean
    gz = gz.reshape(H, W)
    valid = ~np.isnan(gz)                                          # cells with real data
    # display mask: fill pinholes, then drop isolated edge specks -> clean coastline
    land = binary_opening(binary_closing(valid, np.ones((3, 3)), iterations=2), np.ones((3, 3)), iterations=2)
    sig = 6
    wsm = gaussian_filter(valid.astype(float), sigma=sig)          # smooth using only real data
    sm = gaussian_filter(np.where(valid, gz, 0.0), sigma=sig) / np.where(wsm > 1e-3, wsm, 1.0)
    sm[~land] = np.nan
    m = ~np.isnan(sm)
    disp = np.full(sm.shape, np.nan)
    disp[m] = sm[m].argsort().argsort() / max(1, int(m.sum()) - 1)  # smooth -> percentile
    disp = disp[::-1, :]                                            # row 0 = north
    rgba = PSYCH(np.nan_to_num(disp, nan=0.0))
    rgba[..., 3] = np.where(np.isnan(disp), 0.0, 0.88)
    plt.imsave(str(path), rgba)
    return [[lat0, lon0], [lat1, lon1]]

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / 'data'
ROOT = HERE.parent
DEFAULT_DIRS = [ROOT / 'Training Data Florida' / 'FLgrid', ROOT / 'Training Data Florida' / 'grid']


def load_grid(grid_dir):
    files = glob.glob(str(grid_dir) + '/*.csv')
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(c) for c in files], ignore_index=True)
    # FL bbox
    df = df[(df.lat < 31.05) & (df.lat > 24.4) & (df.lon > -87.7) & (df.lon < -79.8)]
    # drop ocean (no MODIS vegetation) and de-dup
    if 'NDVI' in df.columns:
        df = df[df.NDVI.notna()]
    df = df.drop_duplicates(subset=['lon', 'lat']).reset_index(drop=True)
    # current-date satellite lags: ET/PET sentinel 0 -> NaN so the model treats them as missing
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
    # display as PERCENTILE within Florida (prioritization tool: "how does this cell rank?").
    # Robust to the balanced-training scale and to lagging satellite layers; spreads the map.
    df['risk'] = df['risk_raw'].rank(pct=True).round(4)
    core.set_medians(df)
    bounds = render_raster(df, DATA / 'fl_risk.png')   # PNG kept as a static fallback only
    # native grid spacing, so the frontend can draw each cell as a crisp rectangle
    lons = np.sort(df.lon.unique()); gaps = np.diff(lons); gaps = gaps[gaps > 1e-6]
    cell = float(round(float(np.min(gaps)), 3)) if len(gaps) else 0.04
    # per-point percentile — the SAME value /api/explain returns on click, so cell color == click
    points = df[['lon', 'lat', 'risk']].round({'lon': 3, 'lat': 3, 'risk': 3}).values.tolist()
    from datetime import datetime
    meta = {'model': core.load().get('_path'), 'n_points': int(len(df)), 'display': 'percentile',
            'raw_median': round(float(df.risk_raw.median()), 3), 'bounds': bounds,
            'cell': cell, 'computed': datetime.now().strftime('%Y-%m-%d')}
    json.dump({'meta': meta, 'points': points}, open(DATA / 'fl_risk.json', 'w'))
    df.to_csv(DATA / 'fl_grid_full.csv', index=False)   # raw + percentile + features, for explain
    print('wrote fl_risk.json + fl_grid_full.csv:', meta)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
