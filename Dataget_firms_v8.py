"""
v8: BETTER POSITIVES via FIRMS active fire (real fire occurrence: small+large+near-
people), fixing the MTBS large-fire/remoteness bias. Adds human-access feature
(distance-to-developed). Negatives season-matched to the FIRMS fire month distribution.

GEE-friendly: positives batched by DAY-RANGE (each export task samples ~48 days, so no
"too many concurrent aggregations"); negatives' dates drawn from a precomputed month
pool (no heavy interactive aggregation). Features 3 days BEFORE detection (short-lead).

  python Dataget_firms_v8.py --test
  python Dataget_firms_v8.py
"""
import ee, sys
PROJECT_ID = 'gleaming-glass-426122-k0'
ee.Initialize(project=PROJECT_ID)
SE_BBOX = ee.Geometry.Rectangle([-94.5, 24.5, -75.0, 37.5])
FIRE_START = '2017-01-01'
DAY_SPAN = 2900
DAY_STEP = 6           # sample a fire day every 6 days within each batch's range
PIX_PER_DAY = 18
LEAD_DAYS = 3
EXPORT_FOLDER = 'Fire_Ignition_Florida_v8'
N_NEG = 9000
POS_BATCHES = 10
NEG_BATCHES = 6

# FIRMS SE fire-pixel month weights (Jan..Dec), measured 2019-2022:
MW = [0.0695, 0.0907, 0.1811, 0.0904, 0.0468, 0.0457, 0.0433, 0.0511, 0.0908, 0.1167, 0.1131, 0.0608]
# client-side month pool (~1000 entries) for season-matched negative dates
_POOL = []
for _m, _w in enumerate(MW, 1):
    _POOL += [_m] * max(1, round(_w * 1000))
MONTH_POOL = ee.List(_POOL)

GM = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
DROUGHT = ee.ImageCollection("GRIDMET/DROUGHT")
SRTM = ee.Image('USGS/SRTMGL1_003').unmask(0)
NLCD = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq('system:index', '2021')).first()
POP = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01').mosaic()
LC = NLCD.select('landcover')
BURNABLE = (LC.eq(41).Or(LC.eq(42)).Or(LC.eq(43)).Or(LC.eq(52)).Or(LC.eq(71)).Or(LC.eq(81)).Or(LC.eq(90)).Or(LC.eq(95)))
DEV = LC.eq(21).Or(LC.eq(22)).Or(LC.eq(23)).Or(LC.eq(24))
DEV_DIST = DEV.fastDistanceTransform(2048, 'pixels').sqrt().multiply(30).divide(1000.0).rename('DistDev')

COLUMNS = ['lon', 'lat', 'label', 'month', 'year', 'doy',
           'Elevation', 'Pop_Density', 'DistDev', 'LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Pasture',
           'LC_Wetland', 'LC_Crop', 'LC_Developed', 'NDVI', 'NDMI',
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
        c = td.advance(-off, 'day')
        col = DROUGHT.filterDate(c.advance(-10, 'day'), c.advance(1, 'day')).select('pdsi')
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.sort('system:time_start', False).first(), ee.Image.constant(0).rename('pdsi')))
        return at(img, 'pdsi')

    s2c = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(pt).filterDate(td.advance(-45, 'day'), td).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
    s2 = ee.Image(ee.Algorithms.If(s2c.size().gt(0), s2c.median(), ee.Image.constant([0, 0, 0]).rename(['B8', 'B4', 'B11'])))
    ndvi = at(s2.normalizedDifference(['B8', 'B4']).rename('NDVI'), 'NDVI', 20)
    ndmi = at(s2.normalizedDifference(['B8', 'B11']).rename('NDMI'), 'NDMI', 20)
    coords = pt.coordinates()
    props = {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'),
        'year': td.get('year'), 'month': td.get('month'), 'doy': td.getRelative('day', 'year'),
        'Elevation': at(SRTM.select('elevation'), 'elevation', 90),
        'Pop_Density': at(POP.select('population').unmask(0).rename('pop'), 'pop', 100),
        'DistDev': at(DEV_DIST, 'DistDev', 100),
        'LC_Forest': at(LC.eq(41).Or(LC.eq(42)).Or(LC.eq(43)).rename('lc'), 'lc', 30),
        'LC_Shrub': at(LC.eq(52).rename('lc'), 'lc', 30), 'LC_Grass': at(LC.eq(71).rename('lc'), 'lc', 30),
        'LC_Pasture': at(LC.eq(81).rename('lc'), 'lc', 30), 'LC_Wetland': at(LC.eq(90).Or(LC.eq(95)).rename('lc'), 'lc', 30),
        'LC_Crop': at(LC.eq(82).rename('lc'), 'lc', 30), 'LC_Developed': at(DEV.rename('lc'), 'lc', 30),
        'NDVI': ndvi, 'NDMI': ndmi,
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


def positives_range(lo, hi):
    offs = ee.List.sequence(lo, hi - 1, DAY_STEP)

    def per_day(off):
        off = ee.Number(off); d = ee.Date(FIRE_START).advance(off, 'day')
        col = ee.ImageCollection('FIRMS').filterDate(d, d.advance(1, 'day'))
        fire = ee.Image(ee.Algorithms.If(col.size().gt(0), col.first().select('T21'),
                ee.Image.constant(0).updateMask(ee.Image.constant(0)).rename('T21'))).gt(0).selfMask().rename('f')
        pts = fire.stratifiedSample(numPoints=PIX_PER_DAY, classBand='f', region=SE_BBOX, scale=1000, seed=off.int(), geometries=True, tileScale=8)
        tt = d.advance(-LEAD_DAYS, 'day').millis()
        return pts.map(lambda f: ee.Feature(f.geometry()).set({'label': 1, 'target_time': tt}))
    return ee.FeatureCollection(offs.map(per_day)).flatten()


def negatives(seed):
    pts = BURNABLE.selfMask().rename('b').stratifiedSample(numPoints=N_NEG, classBand='b', region=SE_BBOX, scale=1000, seed=seed, geometries=True, tileScale=8)
    pts = pts.randomColumn('d', seed + 1).randomColumn('dd', seed + 2).randomColumn('yr', seed + 3).randomColumn('part', seed + 4)
    n = MONTH_POOL.size()

    def setd(f):
        m = ee.Number(MONTH_POOL.get(ee.Number(f.get('d')).multiply(n).floor().min(n.subtract(1))))
        day = ee.Number(f.get('dd')).multiply(27).floor().add(1)
        year = ee.Number(2017).add(ee.Number(f.get('yr')).multiply(8).floor()).min(2024)
        return f.set({'label': 0, 'target_time': ee.Date.fromYMD(year, m, day).millis(), 'part': f.get('part')})
    return pts.map(setd)


def run_test():
    pos = positives_range(0, 60)  # ~10 days
    one = ee.Feature(pos.map(features).first()).toDictionary().getInfo()
    print("pos sample month/doy:", one.get('month'), one.get('doy'), "| DistDev:", one.get('DistDev'), "| missing:", [c for c in COLUMNS if c not in one])
    neg = negatives(202)
    on = ee.Feature(neg.map(features).first()).toDictionary().getInfo()
    print("neg sample month/year:", on.get('month'), on.get('year'), "| label:", on.get('label'))
    print("OK")


def run_export():
    step = DAY_SPAN // POS_BATCHES
    for i in range(POS_BATCHES):
        b = positives_range(i * step, (i + 1) * step)
        ee.batch.Export.table.toDrive(collection=b.map(features), description=f'Fire_Ignition_v8_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f"  submitted Fire_Ignition_v8_pos_{i+1}")
    neg = negatives(202)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(features), description=f'Fire_Ignition_v8_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f"  submitted Fire_Ignition_v8_neg_{i+1}")
    print(f"Submitted {POS_BATCHES} pos + {NEG_BATCHES} neg tasks -> '{EXPORT_FOLDER}'.")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
