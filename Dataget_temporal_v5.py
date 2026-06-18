"""
Lever 3: rich temporal weather/drought feature export (scalars, not patches).

The v4 audit showed fire signal is dominated by weather/drought, but v4 captured it
only coarsely (single-day + 30d/90d). This pulls a fine-grained temporal profile at
each fire centroid: multi-window precip/VPD/ERC/FM100/Tmax/RHmin aggregates + a PDSI
trajectory (0/-30/-90/-180d). Scalars per (point,date) -> tiny, fast CSV export.

Same SE-US / MTBS-2017+ fires and matched-temporal-negative scheme as v4, so it stays
on the honest task. Output: CSV in Drive folder 'Fire_Temporal_Florida_v5'.

  python Dataget_temporal_v5.py --test
  python Dataget_temporal_v5.py
"""
import ee
import sys

PROJECT_ID = 'gleaming-glass-426122-k0'
ee.Initialize(project=PROJECT_ID)
SE_BBOX = ee.Geometry.Rectangle([-94.5, 24.5, -75.0, 37.5])
MTBS = "USFS/GTAC/MTBS/burned_area_boundaries/v1"
FIRE_START = '2017-01-01'
EXPORT_FOLDER = 'Fire_Temporal_Florida_v5'
NUM_BATCHES = 6

GM = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
DROUGHT = ee.ImageCollection("GRIDMET/DROUGHT")
SRTM = ee.Image('USGS/SRTMGL1_003').unmask(0)
NLCD = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq('system:index', '2021')).first()

COLUMNS = [
    'lon', 'lat', 'label', 'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open',
    'pr_7', 'pr_14', 'pr_30', 'pr_60', 'pr_90', 'pr_180', 'pr_365',
    'vpd_7', 'vpd_30', 'vpd_90', 'erc_7', 'erc_30', 'erc_90',
    'fm100_30', 'fm100_90', 'tmmx_7', 'tmmx_30', 'tmmx_90', 'rmin_30', 'rmin_90',
    'pdsi_0', 'pdsi_30', 'pdsi_90', 'pdsi_180',
]


def temporal_features(feature):
    pt = feature.geometry()
    td = ee.Date(feature.get('target_time'))

    def at(img, band):
        return img.reduceRegion(ee.Reducer.first(), pt, 4000).get(band)

    def win(days, band, how):
        col = GM.filterDate(td.advance(-days, 'day'), td).select(band)
        img = col.sum() if how == 'sum' else col.mean()
        return at(img, band)

    def pdsi_at(offset):
        c = td.advance(-offset, 'day')
        col = DROUGHT.filterDate(c.advance(-10, 'day'), c.advance(1, 'day')).select('pdsi')
        img = ee.Image(ee.Algorithms.If(col.size().gt(0),
                                        col.sort('system:time_start', False).first(),
                                        ee.Image.constant(0).rename('pdsi')))
        return at(img, 'pdsi')

    elev = at(SRTM.select('elevation'), 'elevation')
    lc = NLCD.select('landcover')
    lc_forest = at(lc.eq(41).Or(lc.eq(42)).Or(lc.eq(43)).rename('landcover'), 'landcover')
    lc_wet = at(lc.eq(90).Or(lc.eq(95)).rename('landcover'), 'landcover')
    lc_open = at(lc.eq(52).Or(lc.eq(71)).Or(lc.eq(81)).Or(lc.eq(82)).rename('landcover'), 'landcover')

    coords = pt.coordinates()
    props = {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'),
        'Elevation': elev, 'LC_Forest': lc_forest, 'LC_Wetland': lc_wet, 'LC_Open': lc_open,
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


def fires():
    return (ee.FeatureCollection(MTBS)
            .filter(ee.Filter.gte('Ig_Date', ee.Date(FIRE_START).millis()))
            .filterBounds(SE_BBOX)
            .randomColumn('part', 42).randomColumn('roff', 777))


def make_pairs(fc):
    def mk(f):
        ig = ee.Number(f.get('Ig_Date'))
        r = ee.Number(f.get('roff'))
        pt = f.geometry().centroid(1)
        pos = ee.Feature(pt).set({'label': 1, 'target_time': ee.Date(ig).advance(r.multiply(29).add(1).round().multiply(-1), 'day').millis()})
        neg = ee.Feature(pt).set({'label': 0, 'target_time': ee.Date(ig).advance(r.multiply(60).add(335).round().multiply(-1), 'day').millis()})
        return ee.FeatureCollection([pos, neg])
    return ee.FeatureCollection(fc.map(mk)).flatten()


def run_test():
    fc = fires()
    print("fires:", fc.size().getInfo())
    s = make_pairs(fc.limit(2)).map(temporal_features)
    one = ee.Feature(s.first()).toDictionary().getInfo()
    miss = [c for c in COLUMNS if c not in one]
    print("sample keys:", sorted(one.keys()))
    print("missing:", miss)
    print("values:", {k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in list(one.items())[:12]})
    print("OK" if not miss else "PROBLEM")


def run_export():
    fc = fires()
    total = fc.size().getInfo()
    print(f"v5 temporal export: {total} fires -> {total*2} samples, {len(COLUMNS)} cols")
    for i in range(NUM_BATCHES):
        lo, hi = i / NUM_BATCHES, (i + 1) / NUM_BATCHES
        batch = fc.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        tbl = make_pairs(batch).map(temporal_features)
        desc = f'Fire_Temporal_v5_Part_{i+1}'
        ee.batch.Export.table.toDrive(collection=tbl, description=desc, folder=EXPORT_FOLDER,
                                      fileFormat='CSV', selectors=COLUMNS).start()
        print(f"  submitted {desc}")
    print(f"Done. Drive folder '{EXPORT_FOLDER}'.")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
