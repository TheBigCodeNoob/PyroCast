"""Turn the model's raw score into decision-useful quantities, grounded in REAL fire history:

  - exp_ign_100km2_yr : expected ignitions per 100 km^2 per year, calibrated by isotonic
                        regression of the model score against actual FPA-FOD ignitions (2017-2020).
  - rel_risk          : that rate / the statewide-median rate ("3x the FL average").
  - tier (0..5)       : operational band (Minimal..Extreme) by score percentile, the way fire
                        agencies use danger classes; each tier is annotated with its real rate range.
  - exposure (0..1)   : how much is at stake nearby (people + development), from existing grid cols.
  - priority / ptier  : VULNERABLE-PLACES ranking = rel_risk x exposure -> where an ignition would
                        do the most harm (the wildland-urban interface). Tiered the same way.

All of this is precomputed offline (needs fpafod_se.csv) and baked into web/data, so the server
stays dependency-light and instant.
"""
import numpy as np, pandas as pd

TIER_NAMES = ['Minimal', 'Low', 'Moderate', 'High', 'Very High', 'Extreme']
TIER_PCT = [0.25, 0.50, 0.75, 0.90, 0.98]          # upper percentile edges of tiers 0..4
TIER_COLORS = ['#1a7d3a', '#74bd57', '#cdd451', '#f3c63f', '#ee7e2b', '#d8342a']
EXPOSURE_COLS = ['Pop_Density', 'built', 'nbhd_dev_500m']


def _snap(v, axis):
    i = np.clip(np.searchsorted(axis, v), 1, len(axis) - 1)
    return np.where(np.abs(axis[i] - v) < np.abs(axis[i - 1] - v), axis[i], axis[i - 1])


def cell_area_km2(df, cell):
    latm = float(df.lat.mean())
    return (cell * 110.574) * (cell * 111.320 * np.cos(np.radians(latm)))


def _pct(s):
    return pd.Series(np.asarray(s, float)).rank(pct=True).fillna(0.0).values


def _tier(pct_rank):
    return np.searchsorted(TIER_PCT, pct_rank, side='right').astype(int)


def observed_ignitions(df, fpafod_csv, cell):
    """Count actual recorded ignitions snapped to each grid cell; returns (counts, n_years)."""
    f = pd.read_csv(fpafod_csv, usecols=['latitude', 'longitude', 'fire_year'])
    f = f[(f.latitude < 31.05) & (f.latitude > 24.4) & (f.longitude > -87.7) & (f.longitude < -79.8)]
    f = f.dropna(subset=['latitude', 'longitude'])
    n_years = max(1, int(f.fire_year.nunique()))
    glon, glat = np.sort(df.lon.unique()), np.sort(df.lat.unique())
    key = pd.DataFrame({'clon': _snap(f.longitude.values, glon), 'clat': _snap(f.latitude.values, glat)})
    cnt = key.groupby(['clon', 'clat']).size().rename('ign').reset_index()
    m = df[['lon', 'lat']].merge(cnt, left_on=['lon', 'lat'], right_on=['clon', 'clat'], how='left')
    return m['ign'].fillna(0.0).values, n_years


def calibrate(df, fpafod_csv, cell):
    """Attach calibrated rate / rel_risk / tier / exposure / priority columns to df (in place-ish).
    Returns (df, calib_meta) where calib_meta describes the tiers + baseline for the legend/UI."""
    from sklearn.isotonic import IsotonicRegression
    df = df.copy()
    ign, n_years = observed_ignitions(df, fpafod_csv, cell)
    rate = ign / n_years                                            # observed ignitions / cell / yr
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0).fit(df.risk_raw.values, rate)
    exp_cell = np.clip(iso.predict(df.risk_raw.values), 0.0, None)  # smoothed monotonic rate / cell / yr
    area = cell_area_km2(df, cell)
    df['exp_ign_100km2_yr'] = np.round(exp_cell * (100.0 / area), 3)
    med = float(np.median(exp_cell)) or float(np.mean(exp_cell))
    df['rel_risk'] = np.round(exp_cell / med, 2) if med > 0 else 0.0
    pr = _pct(df.risk_raw)
    df['tier'] = _tier(pr)
    # exposure (what's at stake) and vulnerable-places priority
    cols = [c for c in EXPOSURE_COLS if c in df.columns]
    df['exposure'] = np.round(np.mean([_pct(df[c]) for c in cols], axis=0), 3) if cols else 0.0
    df['priority_score'] = df['rel_risk'].values * df['exposure'].values
    df['ptier'] = _tier(_pct(df['priority_score']))
    # legend: real rate range per ignition tier
    tier_defs = []
    for k in range(6):
        m = df.tier == k
        sub = df.loc[m, 'exp_ign_100km2_yr']
        rel = df.loc[m, 'rel_risk']
        tier_defs.append({'name': TIER_NAMES[k], 'color': TIER_COLORS[k], 'k': k,
                          'n_cells': int(m.sum()),
                          'rate_lo': round(float(sub.min()), 2) if m.any() else 0.0,
                          'rate_hi': round(float(sub.max()), 2) if m.any() else 0.0,
                          'rel_lo': round(float(rel.min()), 2) if m.any() else 0.0,
                          'rel_hi': round(float(rel.max()), 2) if m.any() else 0.0})
    calib = {'n_years': n_years, 'cell_area_km2': round(float(area), 1),
             'baseline_rate_100km2_yr': round(med * (100.0 / area), 3),
             'total_actual_ignitions': int(ign.sum()),
             'tiers': tier_defs, 'tier_names': TIER_NAMES, 'tier_colors': TIER_COLORS}
    return df, calib


if __name__ == '__main__':
    import pathlib
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    df = pd.read_csv(ROOT / 'web' / 'data' / 'fl_grid_full.csv')
    df, calib = calibrate(df, ROOT / 'fpafod_se.csv', 0.04)
    print('cell area km2:', calib['cell_area_km2'], '| n_years:', calib['n_years'],
          '| baseline ign/100km2/yr:', calib['baseline_rate_100km2_yr'])
    print('\nIGNITION TIERS (by score percentile):')
    for t in calib['tiers']:
        print(f"  {t['name']:10s} n={t['n_cells']:5d}  rate {t['rate_lo']:.2f}-{t['rate_hi']:.2f} /100km2/yr"
              f"  ({t['rel_lo']:.1f}-{t['rel_hi']:.1f}x median)")
    print('\nexposure quartiles:', np.round(df.exposure.quantile([.25, .5, .75, .95]).values, 3))
    print('priority tier counts:', df.ptier.value_counts().sort_index().to_dict())
    # where are the top vulnerable places?
    top = df.sort_values('priority_score', ascending=False).head(8)
    print('\nTOP-8 VULNERABLE CELLS (lon,lat,rel_risk,exposure,rate/100km2):')
    for r in top.itertuples():
        print(f"  {r.lon:.2f},{r.lat:.2f}  rel={r.rel_risk:.1f}x  expo={r.exposure:.2f}  rate={r.exp_ign_100km2_yr:.2f}")
