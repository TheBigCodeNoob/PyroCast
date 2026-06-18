"""
v10: the cleanest positives yet — real WILDFIRE ignitions from FPA-FOD (no prescribed
fire), SE-US 2017-2020, sampled from fpafod_se.csv. Features = v8 stack (spatial +
human-access DistDev/Pop + temporal weather/drought/fuel). Negatives = random burnable,
season-matched to the FPA-FOD ignition month distribution. Keeps 'cause' (1=human,
0=lightning, -1=neg) so we can split later. Features evaluated 2 days BEFORE discovery.

  python Dataget_fpafod_v10.py --test | python Dataget_fpafod_v10.py
"""
import ee, sys, csv, random, datetime
ee.Initialize(project='gleaming-glass-426122-k0')
SE_BBOX = ee.Geometry.Rectangle([-94.5, 24.5, -75.0, 37.5])
N_POS = 10000
N_NEG = 10000
LEAD_DAYS = 2
POS_BATCHES = 7
NEG_BATCHES = 7
EXPORT_FOLDER = 'Fire_Ignition_Florida_v10b'
GM = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
DROUGHT = ee.ImageCollection("GRIDMET/DROUGHT")
SRTM = ee.Image('USGS/SRTMGL1_003').unmask(0)
NLCD = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq('system:index', '2021')).first()
POP = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01').mosaic()
LC = NLCD.select('landcover')
# Negatives are drawn from ALL non-water land (incl. developed), to MATCH the FPA-FOD
# positives' landscape (94% human fires occur near development). Sampling negatives only
# from burnable veg would let the model cheat on "developed=fire". Exclude water(11)/ice(12).
LAND = LC.neq(11).And(LC.neq(12))
DEV = LC.eq(21).Or(LC.eq(22)).Or(LC.eq(23)).Or(LC.eq(24))
DEV_DIST = DEV.fastDistanceTransform(2048, 'pixels').sqrt().multiply(30).divide(1000.0).rename('DistDev')

COLUMNS = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy',
           'Elevation', 'Pop_Density', 'DistDev', 'LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Pasture',
           'LC_Wetland', 'LC_Crop', 'LC_Developed', 'NDVI', 'EVI',
           'pr_7', 'pr_14', 'pr_30', 'pr_60', 'pr_90', 'pr_180', 'pr_365',
           'vpd_7', 'vpd_30', 'vpd_90', 'erc_7', 'erc_30', 'erc_90', 'fm100_30', 'fm100_90',
           'tmmx_7', 'tmmx_30', 'tmmx_90', 'rmin_30', 'rmin_90', 'pdsi_0', 'pdsi_30', 'pdsi_90', 'pdsi_180']


def features(feature):
    pt = feature.geometry(); td = ee.Date(feature.get('target_time'))

    def at(img, band, scale=4000):
        return img.reduceRegion(ee.Reducer.first(), pt, scale).get(band)

    def win(days, band, how):
        col = GM.filterDate(td.advance(-days, 'day'), td).select(band)
        return at(col.sum() if how == 'sum' else col.mean(), band)

    def pdsi_at(off):
        c = td.advance(-off, 'day'); col = DROUGHT.filterDate(c.advance(-12, 'day'), c.advance(1, 'day')).select('pdsi')
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.sort('system:time_start', False).first(), ee.Image.constant(0).rename('pdsi')))
        return at(img, 'pdsi')

    # MODIS NDVI/EVI (precomputed 16-day 250m) — keeps the vegetation/fuel signal at a
    # tiny fraction of the cost of an on-the-fly Sentinel-2 median composite.
    mod = ee.ImageCollection('MODIS/061/MOD13Q1').filterBounds(pt).filterDate(td.advance(-40, 'day'), td)
    modimg = ee.Image(ee.Algorithms.If(mod.size().gt(0), mod.sort('system:time_start', False).first(),
             ee.Image.constant([0, 0]).rename(['NDVI', 'EVI'])))
    coords = pt.coordinates()
    props = {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'), 'cause': feature.get('cause'),
        'year': td.get('year'), 'month': td.get('month'), 'doy': td.getRelative('day', 'year'),
        'Elevation': at(SRTM.select('elevation'), 'elevation', 90),
        'Pop_Density': at(POP.select('population').unmask(0).rename('pop'), 'pop', 100),
        'DistDev': at(DEV_DIST, 'DistDev', 100),
        'LC_Forest': at(LC.eq(41).Or(LC.eq(42)).Or(LC.eq(43)).rename('lc'), 'lc', 30),
        'LC_Shrub': at(LC.eq(52).rename('lc'), 'lc', 30), 'LC_Grass': at(LC.eq(71).rename('lc'), 'lc', 30),
        'LC_Pasture': at(LC.eq(81).rename('lc'), 'lc', 30), 'LC_Wetland': at(LC.eq(90).Or(LC.eq(95)).rename('lc'), 'lc', 30),
        'LC_Crop': at(LC.eq(82).rename('lc'), 'lc', 30), 'LC_Developed': at(DEV.rename('lc'), 'lc', 30),
        'NDVI': at(modimg.select('NDVI').multiply(0.0001), 'NDVI', 250),
        'EVI': at(modimg.select('EVI').multiply(0.0001), 'EVI', 250),
        'pr_7': win(7, 'pr', 'sum'), 'pr_14': win(14, 'pr', 'sum'), 'pr_30': win(30, 'pr', 'sum'),
        'pr_60': win(60, 'pr', 'sum'), 'pr_90': win(90, 'pr', 'sum'), 'pr_180': win(180, 'pr', 'sum'), 'pr_365': win(365, 'pr', 'sum'),
        'vpd_7': win(7, 'vpd', 'mean'), 'vpd_30': win(30, 'vpd', 'mean'), 'vpd_90': win(90, 'vpd', 'mean'),
        'erc_7': win(7, 'erc', 'mean'), 'erc_30': win(30, 'erc', 'mean'), 'erc_90': win(90, 'erc', 'mean'),
        'fm100_30': win(30, 'fm100', 'mean'), 'fm100_90': win(90, 'fm100', 'mean'),
        'tmmx_7': win(7, 'tmmx', 'mean'), 'tmmx_30': win(30, 'tmmx', 'mean'), 'tmmx_90': win(90, 'tmmx', 'mean'),
        'rmin_30': win(30, 'rmin', 'mean'), 'rmin_90': win(90, 'rmin', 'mean'),
        'pdsi_0': pdsi_at(0), 'pdsi_30': pdsi_at(30), 'pdsi_90': pdsi_at(90), 'pdsi_180': pdsi_at(180),
    }
    return ee.Feature(pt, props)


def _load_positives():
    rows = list(csv.DictReader(open('fpafod_se.csv')))
    random.seed(42); random.shuffle(rows)
    pts, months = [], []
    for r in rows:
        try:
            lon, lat = float(r['longitude']), float(r['latitude'])
            yr, doy = int(float(r['fire_year'])), int(float(r['discovery_doy']))
        except Exception:
            continue
        if not (24.5 < lat < 37.5 and -94.5 < lon < -75.0) or doy < 1 or doy > 366:
            continue
        d = datetime.datetime(yr, 1, 1) + datetime.timedelta(days=doy - 1 - LEAD_DAYS)
        tm = int(d.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
        cause = 1 if r['nwcg_cause_classification'] == 'Human' else (0 if r['nwcg_cause_classification'] == 'Natural' else -1)
        pts.append((lon, lat, tm, cause))
        months.append((datetime.datetime(yr, 1, 1) + datetime.timedelta(days=doy - 1)).month)
        if len(pts) >= N_POS:
            break
    return pts, months


POS, MONTHS = _load_positives()
_POOL = []
for _m in range(1, 13):
    _POOL += [_m] * max(1, round(MONTHS.count(_m) / max(1, len(MONTHS)) * 1000))
MONTH_POOL = ee.List(_POOL)


def pos_fc(lo, hi):
    return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p[0], p[1]]), {'label': 1, 'cause': p[3], 'target_time': p[2]}) for p in POS[lo:hi]])


def negatives(seed):
    pts = LAND.selfMask().rename('b').stratifiedSample(numPoints=N_NEG, classBand='b', region=SE_BBOX, scale=1000, seed=seed, geometries=True, tileScale=8)
    pts = pts.randomColumn('d', seed + 1).randomColumn('dd', seed + 2).randomColumn('yr', seed + 3).randomColumn('part', seed + 4)
    n = MONTH_POOL.size()

    def setd(f):
        m = ee.Number(MONTH_POOL.get(ee.Number(f.get('d')).multiply(n).floor().min(n.subtract(1))))
        day = ee.Number(f.get('dd')).multiply(27).floor().add(1)
        year = ee.Number(2017).add(ee.Number(f.get('yr')).multiply(4).floor()).min(2020)
        return f.set({'label': 0, 'cause': -1, 'target_time': ee.Date.fromYMD(year, m, day).millis(), 'part': f.get('part')})
    return pts.map(setd)


def run_test():
    one = ee.Feature(pos_fc(0, 5).map(features).first()).toDictionary().getInfo()
    print(f'loaded {len(POS)} positives; sample:', {k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in list(one.items())[:14]})
    print('missing:', [c for c in COLUMNS if c not in one])
    on = ee.Feature(negatives(303).map(features).first()).toDictionary().getInfo()
    print('neg month/year/cause:', on.get('month'), on.get('year'), on.get('cause'))


def run_export():
    print(f'v10 FPA-FOD export: pos={len(POS)} neg={N_NEG} cols={len(COLUMNS)}')
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        fc = pos_fc(i * chunk, (i + 1) * chunk)
        ee.batch.Export.table.toDrive(collection=fc.map(features), description=f'Fire_Ignition_v10b_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_Ignition_v10b_pos_{i+1}')
    neg = negatives(303)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(features), description=f'Fire_Ignition_v10b_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_Ignition_v10b_neg_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
