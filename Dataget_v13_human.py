"""
v13a: HUMAN-PRESSURE features that capture the ignition causes our current features miss.
DistDev only knows distance-to-buildings; it's blind to ROADS and POWER LINES, which cause a
huge share of real wildfires. The Global Human Modification index (gHM) integrates roads,
power, rail, built-up and agriculture into one "human pressure" surface. Plus finer built-up
(GHSL) and settlement footprint (WSF).
  ghm, ghm_2km          : Global Human Modification (CSP), point + 2km mean
  built, built_2km      : GHSL built-surface (m^2/cell), point + 2km mean
  wsf_2km               : World Settlement Footprint fraction within 2km
Computed at the 25k v11e points, merged.
  python Dataget_v13_human.py --test | python Dataget_v13_human.py
"""
import ee, sys
from Dataget_v11e import POS, pos_fc, negatives, POS_BATCHES, NEG_BATCHES

EXPORT_FOLDER = 'Fire_v13_human'
COLUMNS = ['lon', 'lat', 'label', 'cause', 'year', 'doy', 'ghm', 'ghm_2km', 'built', 'built_2km', 'wsf_2km']
GHM = ee.Image(ee.ImageCollection('CSP/HM/GlobalHumanModification').first()).select('gHM').rename('ghm').unmask(0)
BUILT = ee.Image(ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S').filterDate('2019-01-01', '2021-06-01').first()).select('built_surface').rename('built').unmask(0)
WSF = ee.Image('DLR/WSF/WSF2015/v1').eq(255).rename('wsf').unmask(0)


def extra(feature):
    pt = feature.geometry(); td = ee.Date(feature.get('target_time'))

    def at(img, band, scale):
        return img.reduceRegion(ee.Reducer.first(), pt, scale).get(band)

    def mean(img, band, radius, scale=100):
        return img.rename(band).reduceRegion(ee.Reducer.mean(), pt.buffer(radius), scale).get(band)
    coords = pt.coordinates()
    return ee.Feature(pt, {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'), 'cause': feature.get('cause'),
        'year': td.get('year'), 'doy': td.getRelative('day', 'year'),
        'ghm': at(GHM, 'ghm', 1000), 'ghm_2km': mean(GHM, 'ghm', 2000, 300),
        'built': at(BUILT, 'built', 100), 'built_2km': mean(BUILT, 'built', 2000, 100),
        'wsf_2km': mean(WSF, 'wsf', 2000, 100),
    })


def run_test():
    one = ee.Feature(pos_fc(0, 5).map(extra).first()).toDictionary().getInfo()
    print('sample:', {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in one.items()})
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        ee.batch.Export.table.toDrive(collection=pos_fc(i * chunk, (i + 1) * chunk).map(extra), description=f'Fire_v13_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v13_pos_{i+1}')
    neg = negatives(303)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(extra), description=f'Fire_v13_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v13_neg_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
