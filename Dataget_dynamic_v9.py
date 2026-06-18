"""
#1 / v9: add the strongest untried DYNAMIC fire-danger features to the EXACT v7 MTBS
samples (same seeds -> identical points/dates), then merge with v7 locally. Goal: see if
genuine (pop+season-matched) skill can rise toward 0.8 from real dynamics, not crutches.

New features (GRIDMET): Burning Index, 1000-hr fuel moisture, wind (multi-window),
dry-day counts, longer precip + drought windows, heat/VPD extremes.
Exports only the new features + merge keys (lon,lat,label,year,doy).

  python Dataget_dynamic_v9.py --test | python Dataget_dynamic_v9.py
"""
import ee, sys
ee.Initialize(project='gleaming-glass-426122-k0')
SE_BBOX = ee.Geometry.Rectangle([-94.5, 24.5, -75.0, 37.5])
MTBS = "USFS/GTAC/MTBS/burned_area_boundaries/v1"
FIRE_START = '2017-01-01'
EXPORT_FOLDER = 'Fire_Dynamic_Florida_v9'
N_NEG = 6400
POS_BATCHES = 4
NEG_BATCHES = 6
GM = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
DROUGHT = ee.ImageCollection("GRIDMET/DROUGHT")
LC = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq('system:index', '2021')).first().select('landcover')
BURNABLE = (LC.eq(41).Or(LC.eq(42)).Or(LC.eq(43)).Or(LC.eq(52)).Or(LC.eq(71)).Or(LC.eq(81)).Or(LC.eq(90)).Or(LC.eq(95)))

COLUMNS = ['lon', 'lat', 'label', 'year', 'doy',
           'bi_30', 'bi_90', 'fm1000_30', 'fm1000_90', 'vs_7', 'vs_30',
           'dry_days_14', 'dry_days_30', 'pr_270', 'pr_540', 'pdsi_270', 'pdsi_365',
           'vpd_max_30', 'tmmx_max_7']


def features(feature):
    pt = feature.geometry(); td = ee.Date(feature.get('target_time'))

    def at(img, band, scale=4000):
        return img.reduceRegion(ee.Reducer.first(), pt, scale).get(band)

    def win(days, band, how):
        col = GM.filterDate(td.advance(-days, 'day'), td).select(band)
        img = {'mean': col.mean(), 'sum': col.sum(), 'max': col.max()}[how]
        return at(img, band)

    def dry_days(days):
        col = GM.filterDate(td.advance(-days, 'day'), td).select('pr')
        return at(col.map(lambda i: i.lt(1.0)).sum().rename('pr'), 'pr')

    def pdsi_at(off):
        c = td.advance(-off, 'day')
        col = DROUGHT.filterDate(c.advance(-12, 'day'), c.advance(1, 'day')).select('pdsi')
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.sort('system:time_start', False).first(), ee.Image.constant(0).rename('pdsi')))
        return at(img, 'pdsi')

    coords = pt.coordinates()
    props = {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'),
        'year': td.get('year'), 'doy': td.getRelative('day', 'year'),
        'bi_30': win(30, 'bi', 'mean'), 'bi_90': win(90, 'bi', 'mean'),
        'fm1000_30': win(30, 'fm1000', 'mean'), 'fm1000_90': win(90, 'fm1000', 'mean'),
        'vs_7': win(7, 'vs', 'mean'), 'vs_30': win(30, 'vs', 'mean'),
        'dry_days_14': dry_days(14), 'dry_days_30': dry_days(30),
        'pr_270': win(270, 'pr', 'sum'), 'pr_540': win(540, 'pr', 'sum'),
        'pdsi_270': pdsi_at(270), 'pdsi_365': pdsi_at(365),
        'vpd_max_30': win(30, 'vpd', 'max'), 'tmmx_max_7': win(7, 'tmmx', 'max'),
    }
    return ee.Feature(pt, props)


def positives():
    fires = (ee.FeatureCollection(MTBS).filter(ee.Filter.gte('Ig_Date', ee.Date(FIRE_START).millis()))
             .filterBounds(SE_BBOX).randomColumn('roff', 777).randomColumn('part', 42))

    def mk(f):
        ig = ee.Number(f.get('Ig_Date')); r = ee.Number(f.get('roff'))
        t = ee.Date(ig).advance(r.multiply(29).add(1).round().multiply(-1), 'day').millis()
        return ee.Feature(f.geometry().centroid(1)).set({'label': 1, 'target_time': t, 'part': f.get('part')})
    return fires.map(mk)


def pos_doys():
    return positives().map(lambda f: f.set('doy', ee.Date(f.get('target_time')).getRelative('day', 'year'))).aggregate_array('doy')


def negatives(seed, doys):
    pts = BURNABLE.selfMask().rename('b').stratifiedSample(numPoints=N_NEG, classBand='b', region=SE_BBOX, scale=1000, seed=seed, geometries=True, tileScale=8)
    pts = pts.randomColumn('d', seed + 1).randomColumn('part', seed + 2).randomColumn('yr', seed + 3)
    n = doys.size()

    def setd(f):
        idx = ee.Number(f.get('d')).multiply(n).floor().min(n.subtract(1))
        doy = ee.Number(doys.get(idx)); year = ee.Number(2017).add(ee.Number(f.get('yr')).multiply(8).floor()).min(2024)
        return f.set({'label': 0, 'target_time': ee.Date.fromYMD(year, 1, 1).advance(doy, 'day').millis(), 'part': f.get('part')})
    return pts.map(setd)


def export_batches(fc, kind, nb):
    for i in range(nb):
        lo, hi = i / nb, (i + 1) / nb
        b = fc.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=b.map(features), description=f'Fire_Dynamic_v9_{kind}_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f"  submitted Fire_Dynamic_v9_{kind}_{i+1}")


def run_test():
    one = ee.Feature(positives().map(features).first()).toDictionary().getInfo()
    print("pos sample:", {k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in one.items()})
    print("missing:", [c for c in COLUMNS if c not in one])


def run_export():
    doys = pos_doys()
    export_batches(positives(), 'pos', POS_BATCHES)
    export_batches(negatives(101, doys), 'neg', NEG_BATCHES)
    print(f"Submitted -> '{EXPORT_FOLDER}'.")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
