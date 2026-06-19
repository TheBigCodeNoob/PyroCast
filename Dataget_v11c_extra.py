"""
v11c (WHERE push): compute NEW candidate where-features at the EXACT v10b points,
to A/B test against the locked metric without a full re-export.
New features:
  NightLights     : VIIRS DNB avg radiance (human-activity intensity)
  nbhd_dev_2km    : fraction developed within 2km   (landscape WUI context)
  nbhd_dev_500m   : fraction developed within 500m  (fine WUI)
  nbhd_forest_2km : fraction forest within 2km      (fuel continuity)
  nbhd_wetland_2km: fraction wetland within 2km
Reuses POS / negatives(303) from Dataget_fpafod_v10 -> identical points/seeds.
Merge back on (lon,lat). Exports to 'Fire_v11c_extra'.

  python Dataget_v11c_extra.py --test | python Dataget_v11c_extra.py
"""
import ee, sys
from Dataget_fpafod_v10 import POS, pos_fc, negatives, DEV, LC, POS_BATCHES, NEG_BATCHES

EXPORT_FOLDER = 'Fire_v11c_extra'
COLUMNS = ['lon', 'lat', 'label', 'cause', 'year', 'doy',
           'NightLights', 'nbhd_dev_2km', 'nbhd_dev_500m', 'nbhd_forest_2km', 'nbhd_wetland_2km']
FOREST = LC.eq(41).Or(LC.eq(42)).Or(LC.eq(43))
WETLAND = LC.eq(90).Or(LC.eq(95))


def extra(feature):
    pt = feature.geometry(); td = ee.Date(feature.get('target_time'))

    def frac(mask, radius):
        return mask.rename('m').reduceRegion(ee.Reducer.mean(), pt.buffer(radius), 100).get('m')

    ntl = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filterBounds(pt).filterDate(td.advance(-120, 'day'), td).select('avg_rad')
    img = ee.Image(ee.Algorithms.If(ntl.size().gt(0), ntl.sort('system:time_start', False).first(), ee.Image.constant(0).rename('avg_rad'))).unmask(0)
    coords = pt.coordinates()
    return ee.Feature(pt, {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'), 'cause': feature.get('cause'),
        'year': td.get('year'), 'doy': td.getRelative('day', 'year'),
        'NightLights': img.reduceRegion(ee.Reducer.first(), pt, 500).get('avg_rad'),
        'nbhd_dev_2km': frac(DEV, 2000), 'nbhd_dev_500m': frac(DEV, 500),
        'nbhd_forest_2km': frac(FOREST, 2000), 'nbhd_wetland_2km': frac(WETLAND, 2000),
    })


def run_test():
    one = ee.Feature(pos_fc(0, 5).map(extra).first()).toDictionary().getInfo()
    print('sample:', {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in one.items()})
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        ee.batch.Export.table.toDrive(collection=pos_fc(i * chunk, (i + 1) * chunk).map(extra), description=f'Fire_v11c_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v11c_pos_{i+1}')
    neg = negatives(303)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(extra), description=f'Fire_v11c_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v11c_neg_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
