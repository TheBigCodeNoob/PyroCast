"""
v11g (HONEST lever, round 2): vertical FUEL STRUCTURE — canopy height (ETH 10m 2020) +
tree cover (Hansen). Captures tall-flammable-pine vs short-scrub vs marsh, which NLCD
land-cover classes and NDVI greenness miss. Pure vegetation structure -> not a human-
access crutch. Computed at the exact v10b points, merged.
  python Dataget_v11g_extra.py --test | python Dataget_v11g_extra.py
"""
import ee, sys
from Dataget_fpafod_v10 import POS, pos_fc, negatives, POS_BATCHES, NEG_BATCHES

EXPORT_FOLDER = 'Fire_v11g_extra'
COLUMNS = ['lon', 'lat', 'label', 'cause', 'year', 'doy',
           'canopy_ht', 'canopy_ht_2km', 'treecover', 'treecover_2km']
CANOPY = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').select('b1').rename('ch').unmask(0)
TREECOVER = ee.Image('UMD/hansen/global_forest_change_2023_v1_11').select('treecover2000').rename('tc').unmask(0)


def extra(feature):
    pt = feature.geometry(); td = ee.Date(feature.get('target_time'))
    coords = pt.coordinates()
    return ee.Feature(pt, {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'), 'cause': feature.get('cause'),
        'year': td.get('year'), 'doy': td.getRelative('day', 'year'),
        'canopy_ht': CANOPY.reduceRegion(ee.Reducer.first(), pt, 10).get('ch'),
        'canopy_ht_2km': CANOPY.reduceRegion(ee.Reducer.mean(), pt.buffer(2000), 100).get('ch'),
        'treecover': TREECOVER.reduceRegion(ee.Reducer.first(), pt, 30).get('tc'),
        'treecover_2km': TREECOVER.reduceRegion(ee.Reducer.mean(), pt.buffer(2000), 100).get('tc'),
    })


def run_test():
    one = ee.Feature(pos_fc(0, 5).map(extra).first()).toDictionary().getInfo()
    print('sample:', {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in one.items()})
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        ee.batch.Export.table.toDrive(collection=pos_fc(i * chunk, (i + 1) * chunk).map(extra), description=f'Fire_v11g_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v11g_pos_{i+1}')
    neg = negatives(303)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(extra), description=f'Fire_v11g_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v11g_neg_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
