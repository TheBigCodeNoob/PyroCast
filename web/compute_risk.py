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
    """Smooth, readable surface: missing-data-aware Gaussian smooth of the raw risk (removes per-
    cell speckle without touching the model), then re-rank to percentile so the colors keep the
    psychology-tuned green-heavy spread. Ocean/gaps stay transparent. Returns map bounds."""
    from scipy.ndimage import gaussian_filter, zoom
    piv = df.pivot_table(index='lat', columns='lon', values='risk_raw')
    arr = piv.values
    mask = ~np.isnan(arr)
    sig = 1.3                                                       # light: kill speckle, keep definition
    w = gaussian_filter(mask.astype(float), sigma=sig)
    sm = gaussian_filter(np.where(mask, arr, 0.0), sigma=sig) / np.where(w > 1e-3, w, 1.0)
    sm[~mask] = np.nan
    land = sm[mask]
    disp = np.full_like(sm, np.nan)
    disp[mask] = land.argsort().argsort() / max(1, len(land) - 1)   # smooth -> percentile
    disp = disp[::-1, :]                                            # row 0 = north
    Z = 6                                                           # high-res output so it renders crisp
    big = zoom(np.nan_to_num(disp, nan=0.0), Z, order=1)
    bigm = zoom(mask[::-1, :].astype(float), Z, order=1)
    big[bigm < 0.5] = np.nan
    rgba = PSYCH(np.nan_to_num(big, nan=0.0))
    rgba[..., 3] = np.where(np.isnan(big), 0.0, 0.88)
    plt.imsave(str(path), rgba)
    return [[float(piv.index.min()), float(piv.columns.min())], [float(piv.index.max()), float(piv.columns.max())]]

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
    bounds = render_raster(df, DATA / 'fl_risk.png')
    meta = {'model': core.load().get('_path'), 'n_points': int(len(df)), 'display': 'percentile',
            'raw_median': round(float(df.risk_raw.median()), 3), 'bounds': bounds}
    json.dump({'meta': meta}, open(DATA / 'fl_risk.json', 'w'))
    df.to_csv(DATA / 'fl_grid_full.csv', index=False)   # raw + percentile + features, for explain
    print('wrote fl_risk.json + fl_grid_full.csv:', meta)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
