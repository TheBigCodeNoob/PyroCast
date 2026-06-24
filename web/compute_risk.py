"""Score a grid of FL points and write the demo's data files. Run by the scheduler after a
fresh GEE export, or on any existing grid CSV dir.
  python compute_risk.py [grid_csv_dir]
Outputs: web/data/fl_risk.json (compact, for the map) + web/data/fl_grid_full.csv (full
features, for on-demand explanations)."""
import sys, glob, pathlib, json, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pyrocast_core as core

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
    df['risk'] = core.score(df)
    core.set_medians(df)
    # compact JSON for the map
    out = df[['lon', 'lat', 'risk']].round({'lon': 4, 'lat': 4, 'risk': 4})
    meta = {'model': core.load().get('_path'), 'n_points': int(len(out)),
            'risk_min': round(float(out.risk.min()), 3), 'risk_max': round(float(out.risk.max()), 3),
            'risk_median': round(float(out.risk.median()), 3)}
    json.dump({'meta': meta, 'points': out.to_dict('records')}, open(DATA / 'fl_risk.json', 'w'))
    # full features for on-demand explanations
    df.to_csv(DATA / 'fl_grid_full.csv', index=False)
    print('wrote fl_risk.json + fl_grid_full.csv:', meta)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
