"""PyroCast core: load the model, score grid points, and explain WHY a point is high/low
risk via grouped occlusion (no SHAP needed). Shared by the API and the compute job."""
import json, pathlib, numpy as np, pandas as pd, joblib

ROOT = pathlib.Path(__file__).resolve().parent.parent
# prefer FL ensemble (lowest-variance) > FL single > general best
_MODEL_PATHS = [ROOT / 'best_model_fl_ensemble.joblib', ROOT / 'best_model_fl.joblib',
                ROOT / 'best_model_v13.joblib', ROOT / 'best_model_final.joblib']

# human-readable factor groups (feature -> plain English bucket)
FACTOR_GROUPS = {
    'Nearness to people & development': ['DistDev', 'Pop_Density', 'nbhd_dev_500m', 'LC_Developed', 'NightLights', 'built'],
    'Drought & rainfall': ['pdsi_0', 'pdsi_30', 'pdsi_90', 'pdsi_180', 'pdsi_traj_90', 'dryness', 'pr_deficit',
                           'pr_7', 'pr_14', 'pr_30', 'pr_60', 'pr_90', 'pr_180', 'pr_365', 'vpd_7', 'vpd_30', 'vpd_90',
                           'vpd_trend', 'erc_7', 'erc_30', 'erc_90', 'fm100_30', 'fm100_90', 'fm100_trend', 'et_stress'],
    'Vegetation moisture': ['ndmi', 'smap_root', 'lst_day', 'NDVI', 'EVI'],
    'Forest & canopy fuel': ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km', 'LC_Forest', 'LC_Shrub', 'nbhd_forest_2km'],
    'Grass & cropland fuel': ['LC_Grass', 'LC_Crop', 'LC_Pasture', 'nbhd_crop_2km', 'nbhd_pasture_2km'],
    'Wetland & marsh (sawgrass)': ['LC_Wetland', 'nbhd_wetland_2km', 'nbhd_wetland_5km'],
    'Temperature & humidity': ['tmmx_7', 'tmmx_30', 'tmmx_90', 'rmin_30', 'rmin_90'],
    'Terrain': ['Elevation'],
}
# landscape ("why this place") vs the rest, which are weather/moisture ("conditions right now")
STATIC_FACTORS = {'Nearness to people & development', 'Forest & canopy fuel', 'Grass & cropland fuel',
                  'Wetland & marsh (sawgrass)', 'Terrain'}

TIER_NAMES = ['Minimal', 'Low', 'Moderate', 'High', 'Very High', 'Extreme']
# modifiable landscape drivers -> the concrete action an agency can schedule/fund to lower risk
MITIGATION = {
    'Forest & canopy fuel': 'Fuel reduction -mechanical thinning or prescribed fire to cut canopy & surface fuel load.',
    'Grass & cropland fuel': 'Prescribed burns, mowing, or grazed firebreaks to break up fine-fuel continuity.',
    'Wetland & marsh (sawgrass)': 'Hydrologic management + targeted prescribed burns in sawgrass/marsh fuels.',
    'Nearness to people & development': 'Ignition prevention near homes -defensible space, debris-burn/campfire limits, WUI hardening.',
}
TIMING_GROUPS = {'Drought & rainfall', 'Temperature & humidity', 'Vegetation moisture'}

ENGINEERED = {
    'pdsi_traj_90': lambda d: d.pdsi_0 - d.pdsi_90, 'vpd_trend': lambda d: d.vpd_7 - d.vpd_90,
    'pr_deficit': lambda d: d.pr_365 / 4 - d.pr_90, 'fm100_trend': lambda d: d.fm100_30 - d.fm100_90,
    'dryness': lambda d: d.vpd_30 + d.erc_30 - d.pr_90 / 50, 'et_stress': lambda d: d.et / (d.pet + 1),
}

_BUNDLE = None
_MEDIANS = None


def load():
    global _BUNDLE
    if _BUNDLE is None:
        for p in _MODEL_PATHS:
            if p.exists():
                _BUNDLE = joblib.load(p); _BUNDLE['_path'] = p.name; break
        if _BUNDLE is None:
            raise FileNotFoundError('no model found')
    return _BUNDLE


def _predict(X):
    """probability of ignition; averages the ensemble members if present (lower variance)."""
    b = load()
    if b.get('models'):
        return np.mean([m.predict_proba(X)[:, 1] for m in b['models']], axis=0)
    return b['model'].predict_proba(X)[:, 1]


def _engineer(df):
    df = df.copy()
    for name, fn in ENGINEERED.items():
        if name not in df.columns:
            try:
                df[name] = fn(df)
            except Exception:
                df[name] = np.nan
    return df


def set_medians(df):
    """call once on a representative scored grid so explanations have a baseline."""
    global _MEDIANS
    b = load(); _MEDIANS = {f: float(np.nanmedian(df[f])) if f in df.columns else 0.0 for f in b['features']}


def score(df):
    """df with raw feature columns -> array of risk probabilities (0-1)."""
    b = load(); df = _engineer(df)
    for f in b['features']:
        if f not in df.columns:
            df[f] = np.nan
    X = df[b['features']].values.astype('float32')
    return _predict(X)


def explain(row, n=4):
    """row: dict/Series of raw features for ONE point. Returns top risk-driving factor groups
    via occlusion: replace each group with the median and measure how much risk drops."""
    b = load()
    if _MEDIANS is None:
        raise RuntimeError('call set_medians first')
    df = _engineer(pd.DataFrame([dict(row)]))
    for f in b['features']:
        if f not in df.columns:
            df[f] = np.nan
    base = float(_predict(df[b['features']].values.astype('float32'))[0])
    contribs = []
    for group, feats in FACTOR_GROUPS.items():
        present = [f for f in feats if f in b['features']]
        if not present:
            continue
        d2 = df.copy()
        for f in present:
            d2[f] = _MEDIANS.get(f, 0.0)
        r2 = float(_predict(d2[b['features']].values.astype('float32'))[0])
        contribs.append({'factor': group, 'effect': round(base - r2, 4)})  # +ve = raises risk
    contribs.sort(key=lambda c: -c['effect'])
    sig = [c for c in contribs if abs(c['effect']) > 0.004]
    place = [c for c in sig if c['factor'] in STATIC_FACTORS][:4]
    conditions = [c for c in sig if c['factor'] not in STATIC_FACTORS][:4]
    # precomputed calibrated fields for this cell (added by calibrate.py; 0/defaults if absent)
    rd = dict(row)

    def _num(k, d=0.0):
        v = rd.get(k, d)
        try:
            v = float(v)
        except Exception:
            return d
        return d if np.isnan(v) else v
    tier = int(_num('tier', 0)); ptier = int(_num('ptier', 0))
    # actionable mitigation: modifiable landscape drivers that RAISE risk here
    actions = [{'driver': c['factor'], 'do': MITIGATION[c['factor']]}
               for c in place if c['factor'] in MITIGATION and c['effect'] > 0][:3]
    timing = any(c['factor'] in TIMING_GROUPS and c['effect'] > 0 for c in conditions)
    # keep legacy keys too (area aggregation / older clients)
    return {'risk': round(base, 4), 'place': place, 'conditions': conditions,
            'raises_risk': [c for c in sig if c['effect'] > 0][:n], 'lowers_risk': [c for c in sig if c['effect'] < 0][-2:],
            'tier': tier, 'tier_name': TIER_NAMES[min(tier, 5)],
            'rate': round(_num('exp_ign_100km2_yr'), 2), 'rel_risk': round(_num('rel_risk'), 1),
            'exposure': round(_num('exposure'), 2),
            'ptier': ptier, 'ptier_name': TIER_NAMES[min(ptier, 5)],
            'actions': actions,
            'timing': ('Fire-weather is currently elevated here -a short-term window for burn bans / pre-positioning crews.'
                       if timing else None)}


def save_grid(df, path):
    """score df, attach risk, write the compact JSON the frontend fetches."""
    df = df.copy(); df['risk'] = score(df)
    set_medians(df)
    land = df.get('NDVI', pd.Series(1, index=df.index)).notna() & df.get('Elevation', pd.Series(1, index=df.index)).notna()
    out = df[land][['lon', 'lat', 'risk']].round({'lon': 4, 'lat': 4, 'risk': 4})
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = {'model': load().get('_path'), 'n_points': len(out), 'risk_median': round(float(out.risk.median()), 3)}
    json.dump({'meta': meta, 'points': out.to_dict('records')}, open(path, 'w'))
    return meta
