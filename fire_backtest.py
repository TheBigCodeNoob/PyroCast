"""Prospective backtest: for real Florida wildfires that started AFTER the 2017-2020 training
window, reconstruct the conditions ~1 week before ignition, score the whole state, and check
whether the fire's location stood out as high risk. Saves one annotated risk map per fire.

  GOOGLE_APPLICATION_CREDENTIALS=<service-account key.json> \
  conda run -n base python fire_backtest.py [--limit N] [--step 0.08] [--lead 7]
"""
import os, sys, glob, json, argparse, datetime, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'web'))
_REAL_ARGV = sys.argv[1:]                       # keep CLI flags before we mask argv for the import
sys.argv = [sys.argv[0], '2022-01-01']        # let Dataget_fl_grid import cleanly; we override dates per fire
import numpy as np, pandas as pd, joblib, ee
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
import Dataget_fl_grid as G
import calibrate

TIER_PCT = calibrate.TIER_PCT; TIER_COLORS = calibrate.TIER_COLORS; TIER_NAMES = calibrate.TIER_NAMES
ENG = {'pdsi_traj_90': lambda d: d.pdsi_0 - d.pdsi_90, 'vpd_trend': lambda d: d.vpd_7 - d.vpd_90,
       'pr_deficit': lambda d: d.pr_365 / 4 - d.pr_90, 'fm100_trend': lambda d: d.fm100_30 - d.fm100_90,
       'dryness': lambda d: d.vpd_30 + d.erc_30 - d.pr_90 / 50, 'et_stress': lambda d: d.et / (d.pet + 1)}
BUNDLE = joblib.load(os.path.join(ROOT, 'best_model_fl_ensemble.joblib'))
FEATURES = BUNDLE['features']
OUTDIR = os.path.join(ROOT, 'figures', 'fire_backtest'); os.makedirs(OUTDIR, exist_ok=True)

# Real FL wildfires AFTER training (2023, 2025). (name, county, start date, acres, cause)
FIRES = [
    ('Mile 31', 'Broward', '2023-03-12', 13500, 'unknown'),
    ('Cypress Camp Trail', 'Collier', '2023-04-04', 9749, 'unknown'),
    ('County Road 121', 'Nassau', '2023-04-20', 1053, 'human (vehicle)'),
    ('Major', 'Marion', '2023-04-27', 4952, 'unknown'),
    ('Sandy 2023', 'Collier', '2023-05-01', 19814, 'lightning'),
    ('Two Barrel', 'Liberty', '2025-02-26', 1943, 'unknown'),
    ('Turkey Hammock', 'Brevard', '2025-02-28', 1171, 'unknown'),
    ('The 344', 'Miami-Dade', '2025-03-12', 26719, 'unknown'),
    ('Johnson', 'Miami-Dade', '2025-03-18', 8340, 'unknown'),
    ('East Marsh', 'Volusia', '2025-03-31', 3377, 'unknown'),
    ('Juniper Swamp', 'Liberty', '2025-04-12', 4986, 'unknown'),
    ('Curry Island', 'Glades', '2025-04-18', 1050, 'unknown'),
    ('Wood', 'Highlands', '2025-05-25', 1272, 'unknown'),
    ('13 Mile', 'Palm Beach', '2025-07-07', 13000, 'unknown'),
    ('Mile Marker 39', 'Broward', '2025-08-18', 48000, 'lightning'),
]


def ms(date_str):
    y, m, d = (int(x) for x in date_str.split('-'))
    return int(datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc).timestamp() * 1000)


def locate_fire(county, start):
    """Centroid of FIRMS fire detections in the county during the first 4 days of the fire."""
    cty = (ee.FeatureCollection('TIGER/2018/Counties')
           .filter(ee.Filter.eq('STATEFP', '12')).filter(ee.Filter.eq('NAME', county)).geometry())
    d0 = datetime.date(*(int(x) for x in start.split('-')))
    firms = (ee.ImageCollection('FIRMS').filterDate(str(d0), str(d0 + datetime.timedelta(days=4)))
             .select('T21').max())
    mask = firms.gt(0)
    ll = ee.Image.pixelLonLat().updateMask(mask)
    c = ll.reduceRegion(ee.Reducer.mean(), cty, 1000, maxPixels=1e9).getInfo()
    if c.get('longitude') is None:
        cc = cty.centroid(1000).coordinates().getInfo()      # fallback: county centroid
        return cc[0], cc[1], False
    return c['longitude'], c['latitude'], True


def _gf(f):
    out = G.grid_feat(f)
    c = f.geometry().coordinates()
    return out.set('lon', c.get(0)).set('lat', c.get(1))


def extract(points, target_ms, chunk=350):
    """Full feature stack for (lon,lat) points at the target time, via chunked getDownloadURL (one
    server-side computation per chunk -> CSV)."""
    cols = G.COLUMNS + ['lon', 'lat']
    frames = []
    for i in range(0, len(points), chunk):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]),
             {'label': 0, 'cause': -1, 'target_time': target_ms}) for lo, la in points[i:i + chunk]])
        feats = fc.map(_gf).filter(ee.Filter.notNull(['NDVI']))
        url = feats.getDownloadURL(filetype='CSV', selectors=cols)
        frames.append(pd.read_csv(url))
        print(f'    chunk {i//chunk+1}/{(len(points)+chunk-1)//chunk}: {len(frames[-1])} land pts', flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score(df):
    df = df.copy()
    for name, fn in ENG.items():
        if name not in df.columns:
            try: df[name] = fn(df)
            except Exception: df[name] = np.nan
    for f in FEATURES:
        if f not in df.columns: df[f] = np.nan
    X = np.nan_to_num(df[FEATURES].to_numpy(dtype='float32'))
    return np.mean([m.predict_proba(X)[:, 1] for m in BUNDLE['models']], axis=0)


def render(df, flon, flat, fire, date, pct, tier, rate, path):
    lon0, lon1, lat0, lat1 = -87.7, -79.8, 24.5, 31.05
    W = 700; H = int(W * (lat1 - lat0) / (lon1 - lon0))
    gx, gy = np.meshgrid(np.linspace(lon0, lon1, W), np.linspace(lat0, lat1, H))
    gz = griddata(df[['lon', 'lat']].values, df['pctile'].values, (gx, gy), method='linear').ravel()
    dist, _ = cKDTree(df[['lon', 'lat']].values).query(np.column_stack([gx.ravel(), gy.ravel()]))
    gz[dist > 0.12] = np.nan; gz = gz.reshape(H, W)
    valid = ~np.isnan(gz)
    land = binary_opening(binary_closing(valid, np.ones((3, 3)), 2), np.ones((3, 3)), 2)
    wsm = gaussian_filter(valid.astype(float), 2.0)
    sm = gaussian_filter(np.where(valid, gz, 0), 2.0) / np.where(wsm > 1e-3, wsm, 1)
    sm[~land] = np.nan
    cmap = ListedColormap(TIER_COLORS); norm = BoundaryNorm([0] + TIER_PCT + [1.0001], cmap.N)
    fig, ax = plt.subplots(figsize=(6.2, 6.6), facecolor='#0e1116')
    ax.imshow(cmap(norm(np.clip(np.nan_to_num(sm), 0, 1))), extent=[lon0, lon1, lat0, lat1],
              origin='lower', interpolation='bilinear')
    ax.imshow(np.where(np.isnan(sm), 1, np.nan), extent=[lon0, lon1, lat0, lat1], origin='lower',
              cmap=ListedColormap(['#0e1116']), vmin=0, vmax=1)
    ax.plot(flon, flat, marker='*', ms=22, mfc='white', mec='black', mew=1.5, zorder=5)
    ax.set_facecolor('#0e1116'); ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{fire}  ({date})\nignition ranked {pct:.0f}th percentile of FL  -  tier: {tier}'
                 f'  ({rate:.1f} fires/100km2/yr)', color='white', fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=130, facecolor='#0e1116'); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--step', type=float, default=0.08)
    ap.add_argument('--lead', type=int, default=7)
    a = ap.parse_args(_REAL_ARGV)
    lons = np.arange(-87.7, -79.8, a.step); lats = np.arange(24.5, 31.05, a.step)
    GRID = [(round(float(lo), 3), round(float(la), 3)) for lo in lons for la in lats]
    print(f'{len(GRID)} grid points at {a.step} deg; lead {a.lead}d')
    fires = FIRES[:a.limit] if a.limit else FIRES
    csv_path = os.path.join(OUTDIR, 'backtest_results.csv')
    done = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try: done = set(pd.read_csv(csv_path).fire)      # resume: skip fires already written
        except Exception: done = set()
    for name, county, start, acres, cause in fires:
        if name in done:
            print(f'  {name}: already done, skip', flush=True); continue
        try:
            flon, flat, located = locate_fire(county, start)
            pre = (datetime.date(*(int(x) for x in start.split('-'))) - datetime.timedelta(days=a.lead))
            tms = ms(str(pre))
            df = extract(GRID, tms)
            if len(df) < 50:
                print(f'  {name}: too few land points ({len(df)}), skip', flush=True); continue
            df['score'] = score(df)
            df['pctile'] = df['score'].rank(pct=True)
            fp = extract([(flon, flat)], tms)
            if len(fp) == 0:
                print(f'  {name}: ignition point on water/no-veg, skip', flush=True); continue
            fsc = score(fp)[0]
            fpct = float((df['score'] < fsc).mean())
            tier = int(np.searchsorted(TIER_PCT, fpct, side='right'))
            cal = json.load(open(os.path.join(ROOT, 'web', 'data', 'fl_risk.json')))['meta']['calib']
            rate = cal['tiers'][tier]['rate']                 # tier percentile -> published rate (same cutpoints)
            render(df, flon, flat, name, pre.strftime('%Y-%m-%d'), fpct * 100, TIER_NAMES[tier], rate,
                   os.path.join(OUTDIR, f"{start}_{name.replace(' ', '_')}.png"))
            row = {'fire': name, 'county': county, 'start': start, 'acres': acres, 'cause': cause,
                   'lon': round(flon, 3), 'lat': round(flat, 3), 'located_by_firms': located,
                   'pre_date': pre.strftime('%Y-%m-%d'), 'ign_pctile': round(fpct * 100, 1),
                   'tier': TIER_NAMES[tier], 'rate': rate}
            hdr = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
            pd.DataFrame([row]).to_csv(csv_path, mode='a', header=hdr, index=False)   # write incrementally
            done.add(name)
            print(f'  {name:20s} {start}  pctile={fpct*100:5.1f}  tier={TIER_NAMES[tier]:9s}  (firms={located})', flush=True)
        except Exception as e:
            print(f'  {name}: ERROR {type(e).__name__}: {e}', flush=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        r = pd.read_csv(csv_path)
        hi = (r.tier.isin(['High', 'Very High', 'Extreme'])).mean()
        md = (r.tier.isin(['Moderate', 'High', 'Very High', 'Extreme'])).mean()
        print(f'\n{len(r)}/{len(fires)} fires done. Median ignition percentile: {r.ign_pctile.median():.0f}. '
              f'{hi*100:.0f}% in High+; {md*100:.0f}% in Moderate+.', flush=True)
        print('saved maps + backtest_results.csv to', OUTDIR, flush=True)


if __name__ == '__main__':
    ee.Initialize(project='gleaming-glass-426122-k0')
    main()
