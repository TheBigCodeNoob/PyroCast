"""
v11d (WHERE push, round 2): expand the winning neighborhood-context lever.
More radii for the winners (dev/forest/wetland) + other fuel fractions (grass/crop/
shrub/pasture) + distance-to-water. Computed at the exact v10b points, merged.
  python Dataget_v11d_extra.py --test | python Dataget_v11d_extra.py
"""
import ee, sys
from Dataget_fpafod_v10 import POS, pos_fc, negatives, DEV, LC, POS_BATCHES, NEG_BATCHES

EXPORT_FOLDER = 'Fire_v11d_extra'
COLUMNS = ['lon', 'lat', 'label', 'cause', 'year', 'doy',
           'nbhd_dev_1km', 'nbhd_forest_1km', 'nbhd_forest_5km', 'nbhd_wetland_5km',
           'nbhd_grass_2km', 'nbhd_crop_2km', 'nbhd_shrub_2km', 'nbhd_pasture_2km', 'dist_water']
FOREST = LC.eq(41).Or(LC.eq(42)).Or(LC.eq(43))
WETLAND = LC.eq(90).Or(LC.eq(95))
GRASS = LC.eq(71); CROP = LC.eq(82); SHRUB = LC.eq(52); PASTURE = LC.eq(81)
WATER_DIST = LC.eq(11).fastDistanceTransform(2048, 'pixels').sqrt().multiply(30).divide(1000.0).rename('dw')


def extra(feature):
    pt = feature.geometry(); td = ee.Date(feature.get('target_time'))

    def frac(mask, radius):
        return mask.rename('m').reduceRegion(ee.Reducer.mean(), pt.buffer(radius), 100).get('m')
    coords = pt.coordinates()
    return ee.Feature(pt, {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'), 'cause': feature.get('cause'),
        'year': td.get('year'), 'doy': td.getRelative('day', 'year'),
        'nbhd_dev_1km': frac(DEV, 1000), 'nbhd_forest_1km': frac(FOREST, 1000), 'nbhd_forest_5km': frac(FOREST, 5000),
        'nbhd_wetland_5km': frac(WETLAND, 5000), 'nbhd_grass_2km': frac(GRASS, 2000), 'nbhd_crop_2km': frac(CROP, 2000),
        'nbhd_shrub_2km': frac(SHRUB, 2000), 'nbhd_pasture_2km': frac(PASTURE, 2000),
        'dist_water': WATER_DIST.reduceRegion(ee.Reducer.first(), pt, 100).get('dw'),
    })


def run_test():
    one = ee.Feature(pos_fc(0, 5).map(extra).first()).toDictionary().getInfo()
    print('sample:', {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in one.items()})
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        ee.batch.Export.table.toDrive(collection=pos_fc(i * chunk, (i + 1) * chunk).map(extra), description=f'Fire_v11d_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v11d_pos_{i+1}')
    neg = negatives(303)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(extra), description=f'Fire_v11d_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v11d_neg_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
