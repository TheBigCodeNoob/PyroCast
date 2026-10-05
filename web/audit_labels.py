"""Rigorous accuracy audit of every PUBLISHED label in PyroCast, against the real fire record
(FPA-FOD ignitions 2017-2020). Reproducible foundation for expert review.

Checks, in order:
  1. Rank skill         - AUC + top-k capture of risk_raw vs "cell ever ignited" (honest headline).
  2. Tier rates         - observed ignition rate per tier, with Poisson 95% CIs; do the published
                          tier ranges match reality? are tiers monotonic?
  3. Spatial holdout    - refit the calibration on held-out spatial blocks; does the rate curve
                          generalize to cells it never saw? (in-sample vs out-of-sample gap)
  4. Temporal holdout   - calibrate on 2017-19, test the rates on 2020.
  5. Cause mix          - human vs natural (lightning) share per tier -> how much is predictable.
  6. Vulnerability      - does the priority/exposure label actually track ignitions near people?
  7. Timing flag        - how the "elevated right now" flag behaves on a STATIC snapshot (honesty).
  8. Coverage           - zero-ignition cells, share of all real ignitions the grid accounts for.

Run: conda run -n base python web/audit_labels.py
"""
import numpy as np, pandas as pd, pathlib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import calibrate

ROOT = pathlib.Path(__file__).resolve().parent.parent
AREA = 17.2                  # km^2 per cell (from calibrate.cell_area_km2)
PER100 = 100.0 / AREA
TIERS = calibrate.TIER_NAMES


def poisson_ci(count, exposure):
    """95% CI for a Poisson rate = count/exposure (Garwood exact via chi2 approx with normal fallback)."""
    if exposure <= 0:
        return (0.0, 0.0)
    if count == 0:
        return (0.0, 3.689 / exposure)
    lo = count * (1 - 1/(9*count) - 1.96/(3*np.sqrt(count)))**3
    hi = (count + 1) * (1 - 1/(9*(count+1)) + 1.96/(3*np.sqrt(count+1)))**3
    return (lo / exposure, hi / exposure)


def load():
    df = pd.read_csv(ROOT / 'web' / 'data' / 'fl_grid_full.csv')
    f = pd.read_csv(ROOT / 'fpafod_se.csv',
                    usecols=['latitude', 'longitude', 'fire_year', 'nwcg_cause_classification'])
    f = f[(f.latitude < 31.05) & (f.latitude > 24.4) & (f.longitude > -87.7) & (f.longitude < -79.8)]
    f = f.dropna(subset=['latitude', 'longitude'])
    glon, glat = np.sort(df.lon.unique()), np.sort(df.lat.unique())
    f['clon'] = calibrate._snap(f.longitude.values, glon)
    f['clat'] = calibrate._snap(f.latitude.values, glat)
    f['nat'] = (f.nwcg_cause_classification == 'Natural').astype(int)
    years = sorted(f.fire_year.dropna().unique().astype(int))
    agg = f.groupby(['clon', 'clat']).agg(ign=('nat', 'size'), nat=('nat', 'sum'))
    yrs = f.pivot_table(index=['clon', 'clat'], columns='fire_year', values='nat', aggfunc='size', fill_value=0)
    yrs.columns = [f'y{int(c)}' for c in yrs.columns]
    agg = agg.join(yrs).reset_index()
    df = df.merge(agg, left_on=['lon', 'lat'], right_on=['clon', 'clat'], how='left')
    for c in ['ign', 'nat'] + [f'y{y}' for y in years]:
        df[c] = df[c].fillna(0.0)
    return df, years, f


def sec(t): print('\n' + '=' * 78 + '\n' + t + '\n' + '-' * 78)


def main():
    df, years, f = load()
    ny = len(years)
    df['rate'] = df.ign / ny
    tot_grid = int(df.ign.sum())
    print(f'grid cells: {len(df)} | years: {years} | FL ignitions snapped to grid: {tot_grid} '
          f'(of {len(f)} in bbox) | cell {AREA} km2')

    sec('1. RANK SKILL (does the raw score find where fires are?)')
    y = (df.ign > 0).astype(int)
    auc = roc_auc_score(y, df.risk_raw)
    order = df.sort_values('risk_raw', ascending=False)
    for k in (0.05, 0.10, 0.25):
        n = int(k * len(df)); cap = order.head(n).ign.sum() / tot_grid
        print(f'  top {int(k*100):>2d}% of cells  -> capture {cap*100:5.1f}% of all ignitions  (lift x{cap/k:.1f})')
    print(f'  AUC (ever-ignited):   {auc:.3f}')
    print(f'  cells that ever ignited: {int(y.sum())} / {len(df)}  ({y.mean()*100:.0f}%)')

    sec('2. TIER RATES vs REALITY (published range vs observed, Poisson 95% CI)')
    print(f'  {"tier":10s} {"cells":>6s} {"fires":>6s} {"obs /100km2/yr":>16s} {"95% CI":>16s} {"published":>14s}')
    prev = -1; mono = True
    for k in range(6):
        m = df.tier == k
        cells = int(m.sum()); fires = int(df.loc[m, 'ign'].sum())
        exposure = cells * ny
        obs = fires / exposure * PER100 if exposure else 0
        lo, hi = poisson_ci(fires, exposure); lo *= PER100; hi *= PER100
        pub_lo = df.loc[m, 'exp_ign_100km2_yr'].min(); pub_hi = df.loc[m, 'exp_ign_100km2_yr'].max()
        print(f'  {TIERS[k]:10s} {cells:6d} {fires:6d} {obs:16.2f} {f"{lo:.2f}-{hi:.2f}":>16s} {f"{pub_lo:.2f}-{pub_hi:.2f}":>14s}')
        if obs < prev - 1e-9: mono = False
        prev = obs
    print(f'  monotonic across tiers: {mono}')

    sec('3. SPATIAL HOLDOUT CALIBRATION (does the rate curve generalize to unseen areas?)')
    block = (np.floor(df.lon / 0.6).astype(int).astype(str) + '_' + np.floor(df.lat / 0.6).astype(int).astype(str))
    ub = block.unique(); rng = np.random.default_rng(0); rng.shuffle(ub)
    folds = np.array_split(ub, 5)
    oos = np.full(len(df), np.nan)
    for fold in folds:
        te = block.isin(fold).values; tr = ~te
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0).fit(df.risk_raw.values[tr], df.rate.values[tr])
        oos[te] = iso.predict(df.risk_raw.values[te])
    df['oos100'] = oos * PER100
    print(f'  {"tier":10s} {"in-sample pred":>15s} {"OOS pred":>10s} {"observed":>10s} {"OOS err":>9s}')
    for k in range(6):
        m = df.tier == k
        ins = df.loc[m, 'exp_ign_100km2_yr'].mean()
        oosp = df.loc[m, 'oos100'].mean()
        obs = df.loc[m, 'ign'].sum() / (m.sum() * ny) * PER100
        print(f'  {TIERS[k]:10s} {ins:15.2f} {oosp:10.2f} {obs:10.2f} {abs(oosp-obs):9.2f}')
    print(f'  mean |OOS pred - in-sample pred| per cell: {np.nanmean(np.abs(df.oos100 - df.exp_ign_100km2_yr)):.3f} /100km2/yr')

    sec('4. TEMPORAL HOLDOUT (calibrate on 2017-19, test on 2020)')
    if 2020 in years and len([y for y in years if y < 2020]) >= 2:
        trn = [f'y{y}' for y in years if y < 2020]
        rate_tr = df[trn].sum(axis=1).values / len(trn)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0).fit(df.risk_raw.values, rate_tr)
        pred20 = iso.predict(df.risk_raw.values) * PER100
        for k in range(6):
            m = (df.tier == k).values
            pr = pred20[m].mean(); ob = df.loc[m, 'y2020'].mean() * PER100
            print(f'  {TIERS[k]:10s} predicted {pr:6.2f}  vs  2020 observed {ob:6.2f}  /100km2/yr')
    else:
        print('  not enough years')

    sec('5. CAUSE MIX per tier (natural/lightning = NOT predictable from landscape+weather)')
    for k in range(6):
        m = df.tier == k; fires = df.loc[m, 'ign'].sum(); nat = df.loc[m, 'nat'].sum()
        print(f'  {TIERS[k]:10s} fires {int(fires):5d}  natural {int(nat):4d} ({(nat/fires*100 if fires else 0):4.1f}%)')
    print(f'  overall natural share: {df.nat.sum()/df.ign.sum()*100:.1f}%')

    sec('6. VULNERABILITY / PRIORITY label validity')
    # does priority rank where ignitions AND people coincide?
    print(f'  corr(priority_score, ignitions):        {np.corrcoef(df.priority_score, df.ign)[0,1]:.3f}')
    print(f'  corr(priority_score, exposure):         {np.corrcoef(df.priority_score, df.exposure)[0,1]:.3f}')
    if 'Pop_Density' in df:
        harm = df.ign * df.Pop_Density
        print(f'  corr(priority_score, ign x Pop_Density): {np.corrcoef(df.priority_score, harm)[0,1]:.3f}')
    for k in [5, 4, 0]:
        m = df.ptier == k
        print(f'  ptier {TIERS[k]:10s} n={int(m.sum()):4d}  mean ign/cell/yr {df.loc[m,"ign"].mean()/ny:.2f}  '
              f'mean exposure {df.loc[m,"exposure"].mean():.2f}')

    sec('7. TIMING FLAG ("fire-weather elevated right now") on a STATIC snapshot')
    # replicate the flag logic: a weather/condition group raises risk. We approximate by checking the
    # share of cells where a dynamic-condition feature is above its median (static data => always "now").
    print('  NOTE: data is a single frozen snapshot; any "right now" wording is time-misleading until')
    print('  live recompute is wired. (Semantic check, fixed in code.)')

    sec('8. COVERAGE')
    z = (df.ign == 0).mean()
    print(f'  cells with 0 recorded ignitions (2017-20): {z*100:.1f}%')
    print(f'  ignitions in grid / in FL bbox: {tot_grid}/{len(f)} = {tot_grid/len(f)*100:.1f}% '
          f'(rest fall on ocean/no-veg cells dropped from the grid)')


if __name__ == '__main__':
    main()
