"""Turn the model's raw score into decision-useful quantities, grounded in REAL fire history and
reported only to the precision the data supports (validated by web/audit_labels.py).

  - tier (0..5)          : operational band (Minimal..Extreme) by score percentile.
  - exp_ign_100km2_yr    : the tier's OBSERVED ignition rate (FPA-FOD 2017-20), i.e. every cell in a
                           tier shows that tier's empirical rate. No per-cell isotonic tails to
                           over-state the top. rate_lo/rate_hi = Poisson 95% CI.
  - rel_risk             : tier rate / the STATEWIDE AVERAGE rate ("2x the Florida average").
  - exposure (0..1)      : people + development at stake, percentile-ranked from existing grid cols.
  - priority_score/ptier : balanced WUI priority = risk percentile x exposure percentile, so neither
                           dominates (the old rel_risk x exposure was ~73% just population).

Precomputed offline (needs fpafod_se.csv); baked into web/data so the server stays dependency-light.
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


def poisson_ci(count, exposure):
    """95% CI for a Poisson rate count/exposure (Byar's approximation)."""
    if exposure <= 0:
        return (0.0, 0.0)
    if count == 0:
        return (0.0, 3.689 / exposure)
    lo = count * (1 - 1 / (9 * count) - 1.96 / (3 * np.sqrt(count))) ** 3
    hi = (count + 1) * (1 - 1 / (9 * (count + 1)) + 1.96 / (3 * np.sqrt(count + 1))) ** 3
    return (lo / exposure, hi / exposure)


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
    """Attach empirical tier rates / rel_risk / exposure / priority to df. Returns (df, calib_meta)."""
    df = df.copy()
    ign, n_years = observed_ignitions(df, fpafod_csv, cell)
    area = cell_area_km2(df, cell); per100 = 100.0 / area
    pr = _pct(df.risk_raw)
    df['tier'] = _tier(pr)
    avg_rate = ign.sum() / (len(df) * n_years) * per100            # statewide average /100km2/yr
    # OBSERVED rate per tier (+ Poisson 95% CI) — the honest, data-supported number
    rate, ci = {}, {}
    for k in range(6):
        m = df.tier.values == k
        fires = float(ign[m].sum()); exposure = int(m.sum()) * n_years
        rate[k] = fires / exposure * per100 if exposure else 0.0
        lo, hi = poisson_ci(fires, exposure); ci[k] = (lo * per100, hi * per100)
    df['exp_ign_100km2_yr'] = np.round(df.tier.map(rate).astype(float), 2)
    df['rate_lo'] = np.round(df.tier.map(lambda k: ci[k][0]).astype(float), 2)
    df['rate_hi'] = np.round(df.tier.map(lambda k: ci[k][1]).astype(float), 2)
    df['rel_risk'] = np.round(df.tier.map(lambda k: rate[k] / avg_rate if avg_rate else 0.0).astype(float), 1)
    # exposure + balanced priority (risk pct x exposure pct, so neither term dominates)
    cols = [c for c in EXPOSURE_COLS if c in df.columns]
    df['exposure'] = np.round(np.mean([_pct(df[c]) for c in cols], axis=0), 3) if cols else 0.0
    df['priority_score'] = np.round(pr * df['exposure'].values, 4)
    df['ptier'] = _tier(_pct(df['priority_score']))
    tiers = [{'name': TIER_NAMES[k], 'color': TIER_COLORS[k], 'k': k,
              'n_cells': int((df.tier == k).sum()), 'rate': round(rate[k], 2),
              'ci_lo': round(ci[k][0], 2), 'ci_hi': round(ci[k][1], 2),
              'rel': round(rate[k] / avg_rate, 1) if avg_rate else 0.0} for k in range(6)]
    calib = {'n_years': n_years, 'cell_area_km2': round(float(area), 1),
             'avg_rate_100km2_yr': round(avg_rate, 2), 'total_actual_ignitions': int(ign.sum()),
             'tiers': tiers, 'tier_names': TIER_NAMES, 'tier_colors': TIER_COLORS}
    return df, calib


if __name__ == '__main__':
    import pathlib
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    df = pd.read_csv(ROOT / 'web' / 'data' / 'fl_grid_full.csv')
    df, calib = calibrate(df, ROOT / 'fpafod_se.csv', 0.04)
    print('avg', calib['avg_rate_100km2_yr'], 'cell_km2', calib['cell_area_km2'])
    for t in calib['tiers']:
        print(f"  {t['name']:10s} n={t['n_cells']:5d} rate {t['rate']} ({t['ci_lo']}-{t['ci_hi']}) {t['rel']}x")
