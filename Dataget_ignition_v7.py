"""
v7: controlled season-matched re-export to test whether the v6 0.93 AUC was
inflated by a calendar-season giveaway.

CONTROLLED CHANGE vs v6 (everything else identical, same seeds):
  - Positives: IDENTICAL to v6 (same fires seed 42/777, same pre-fire dates).
  - Negatives: SAME LOCATIONS as v6 (same stratifiedSample seed=101, same 'd'/'part'
    random columns -> byte-identical points), but their DATES are re-drawn from the
    POSITIVES' day-of-year distribution (random year 2017-2024) instead of uniform
    random dates. So season can no longer separate fire vs non-fire by the calendar.
  - Adds month/year/doy columns so we can also do a temporal holdout (train past,
    test future).

If the AUC drops materially vs v6's 0.93, the headline was partly a season artifact.

  python Dataget_ignition_v7.py --test
  python Dataget_ignition_v7.py
"""
import ee
import sys

PROJECT_ID = 'gleaming-glass-426122-k0'
ee.Initialize(project=PROJECT_ID)
SE_BBOX = ee.Geometry.Rectangle([-94.5, 24.5, -75.0, 37.5])
MTBS = "USFS/GTAC/MTBS/burned_area_boundaries/v1"
FIRE_START = '2017-01-01'
EXPORT_FOLDER = 'Fire_Ignition_Florida_v7'
N_NEG = 6400
POS_BATCHES = 4
NEG_BATCHES = 6

GM = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
DROUGHT = ee.ImageCollection("GRIDMET/DROUGHT")
SRTM = ee.Image('USGS/SRTMGL1_003').unmask(0)
NLCD = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq('system:index', '2021')).first()
POP = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01').mosaic()
LC = NLCD.select('landcover')
BURNABLE = (LC.eq(41).Or(LC.eq(42)).Or(LC.eq(43)).Or(LC.eq(52))
            .Or(LC.eq(71)).Or(LC.eq(81)).Or(LC.eq(90)).Or(LC.eq(95)))

COLUMNS = [
    'lon', 'lat', 'label', 'month', 'year', 'doy',
    'Elevation', 'Pop_Density', 'LC_Forest', 'LC_Shrub', 'LC_Grass', 'LC_Pasture',
    'LC_Wetland', 'LC_Crop', 'LC_Developed', 'NDVI', 'NDMI',
    'pr_7', 'pr_14', 'pr_30', 'pr_60', 'pr_90', 'pr_180', 'pr_365',
    'vpd_7', 'vpd_30', 'vpd_90', 'erc_7', 'erc_30', 'erc_90', 'fm100_30', 'fm100_90',
    'tmmx_7', 'tmmx_30', 'tmmx_90', 'rmin_30', 'rmin_90',
    'pdsi_0', 'pdsi_30', 'pdsi_90', 'pdsi_180',
]


def features(feature):
    pt = feature.geometry()
    td = ee.Date(feature.get('target_time'))

    def at(img, band, scale=4000):
        return img.reduceRegion(ee.Reducer.first(), pt, scale).get(band)

    def win(days, band, how):
        col = GM.filterDate(td.advance(-days, 'day'), td).select(band)
        return at(col.sum() if how == 'sum' else col.mean(), band)

    def pdsi_at(off):
        c = td.advance(-off, 'day')
        col = DROUGHT.filterDate(c.advance(-10, 'day'), c.advance(1, 'day')).select('pdsi')
        img = ee.Image(ee.Algorithms.If(col.size().gt(0),
              col.sort('system:time_start', False).first(), ee.Image.constant(0).rename('pdsi')))
        return at(img, 'pdsi')

    s2c = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(pt)
           .filterDate(td.advance(-45, 'day'), td).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
    s2 = ee.Image(ee.Algorithms.If(s2c.size().gt(0), s2c.median(),
         ee.Image.constant([0, 0, 0]).rename(['B8', 'B4', 'B11'])))
    ndvi = at(s2.normalizedDifference(['B8', 'B4']).rename('NDVI'), 'NDVI', 20)
    ndmi = at(s2.normalizedDifference(['B8', 'B11']).rename('NDMI'), 'NDMI', 20)

    pop = at(POP.select('population').unmask(0).rename('pop'), 'pop', 100)
    coords = pt.coordinates()
    props = {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'),
        'year': td.get('year'), 'month': td.get('month'), 'doy': td.getRelative('day', 'year'),
        'Elevation': at(SRTM.select('elevation'), 'elevation', 90),
        'Pop_Density': pop,
        'LC_Forest': at(LC.eq(41).Or(LC.eq(42)).Or(LC.eq(43)).rename('lc'), 'lc', 30),
        'LC_Shrub': at(LC.eq(52).rename('lc'), 'lc', 30),
        'LC_Grass': at(LC.eq(71).rename('lc'), 'lc', 30),
        'LC_Pasture': at(LC.eq(81).rename('lc'), 'lc', 30),
        'LC_Wetland': at(LC.eq(90).Or(LC.eq(95)).rename('lc'), 'lc', 30),
        'LC_Crop': at(LC.eq(82).rename('lc'), 'lc', 30),
        'LC_Developed': at(LC.eq(21).Or(LC.eq(22)).Or(LC.eq(23)).Or(LC.eq(24)).rename('lc'), 'lc', 30),
        'NDVI': ndvi, 'NDMI': ndmi,
        'pr_7': win(7, 'pr', 'sum'), 'pr_14': win(14, 'pr', 'sum'), 'pr_30': win(30, 'pr', 'sum'),
        'pr_60': win(60, 'pr', 'sum'), 'pr_90': win(90, 'pr', 'sum'), 'pr_180': win(180, 'pr', 'sum'),
        'pr_365': win(365, 'pr', 'sum'),
        'vpd_7': win(7, 'vpd', 'mean'), 'vpd_30': win(30, 'vpd', 'mean'), 'vpd_90': win(90, 'vpd', 'mean'),
        'erc_7': win(7, 'erc', 'mean'), 'erc_30': win(30, 'erc', 'mean'), 'erc_90': win(90, 'erc', 'mean'),
        'fm100_30': win(30, 'fm100', 'mean'), 'fm100_90': win(90, 'fm100', 'mean'),
        'tmmx_7': win(7, 'tmmx', 'mean'), 'tmmx_30': win(30, 'tmmx', 'mean'), 'tmmx_90': win(90, 'tmmx', 'mean'),
        'rmin_30': win(30, 'rmin', 'mean'), 'rmin_90': win(90, 'rmin', 'mean'),
        'pdsi_0': pdsi_at(0), 'pdsi_30': pdsi_at(30), 'pdsi_90': pdsi_at(90), 'pdsi_180': pdsi_at(180),
    }
    return ee.Feature(pt, props)


def positives():
    fires = (ee.FeatureCollection(MTBS)
             .filter(ee.Filter.gte('Ig_Date', ee.Date(FIRE_START).millis()))
             .filterBounds(SE_BBOX).randomColumn('roff', 777).randomColumn('part', 42))

    def mk(f):
        ig = ee.Number(f.get('Ig_Date')); r = ee.Number(f.get('roff'))
        t = ee.Date(ig).advance(r.multiply(29).add(1).round().multiply(-1), 'day').millis()
        return ee.Feature(f.geometry().centroid(1)).set({'label': 1, 'target_time': t, 'part': f.get('part')})
    return fires.map(mk)


def positive_doys():
    return positives().map(lambda f: f.set('doy', ee.Date(f.get('target_time')).getRelative('day', 'year'))).aggregate_array('doy')


def negatives(seed, doys):
    # SAME locations as v6 (seed 101, same random cols). ONLY dates change: season-matched.
    pts = BURNABLE.selfMask().rename('b').stratifiedSample(
        numPoints=N_NEG, classBand='b', region=SE_BBOX, scale=1000, seed=seed, geometries=True, tileScale=8)
    pts = pts.randomColumn('d', seed + 1).randomColumn('part', seed + 2).randomColumn('yr', seed + 3)
    n = doys.size()

    def setd(f):
        idx = ee.Number(f.get('d')).multiply(n).floor().min(n.subtract(1))
        doy = ee.Number(doys.get(idx))
        year = ee.Number(2017).add(ee.Number(f.get('yr')).multiply(8).floor()).min(2024)
        t = ee.Date.fromYMD(year, 1, 1).advance(doy, 'day').millis()
        return f.set({'label': 0, 'target_time': t, 'part': f.get('part')})
    return pts.map(setd)


def export_batches(fc, kind, nbatches):
    for i in range(nbatches):
        lo, hi = i / nbatches, (i + 1) / nbatches
        batch = fc.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=batch.map(features), description=f'Fire_Ignition_v7_{kind}_{i+1}',
                                      folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f"  submitted Fire_Ignition_v7_{kind}_{i+1}")


def run_test():
    doys = positive_doys()
    pos = positives(); neg = negatives(101, doys)
    print("positives:", pos.size().getInfo(), "| negatives:", neg.size().getInfo())
    one = ee.Feature(neg.map(features).first()).toDictionary().getInfo()
    print("neg sample month/year/doy:", one.get('month'), one.get('year'), one.get('doy'), "| label:", one.get('label'))
    print("missing:", [c for c in COLUMNS if c not in one])
    # quick season check: positive vs negative month means
    pm = ee.Number(positives().map(lambda f: ee.Feature(None, {'m': ee.Date(f.get('target_time')).get('month')})).aggregate_mean('m')).getInfo()
    nm = ee.Number(neg.map(lambda f: ee.Feature(None, {'m': ee.Date(f.get('target_time')).get('month')})).aggregate_mean('m')).getInfo()
    print(f"mean month  pos={pm:.2f}  neg={nm:.2f}  (should now be CLOSE)")
    print("OK")


def run_export():
    doys = positive_doys()
    pos, neg = positives(), negatives(101, doys)
    print(f"v7 season-matched export: pos={pos.size().getInfo()} neg={neg.size().getInfo()} cols={len(COLUMNS)}")
    export_batches(pos, 'pos', POS_BATCHES)
    export_batches(neg, 'neg', NEG_BATCHES)
    print(f"Done. Drive folder '{EXPORT_FOLDER}'.")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
