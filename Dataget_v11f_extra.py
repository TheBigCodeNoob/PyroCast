"""
v11f (overnight, HONEST lever): raise the ENVIRONMENTAL floor (0.72) with features that
CANNOT be a human-access/reporting crutch — pure terrain + fuel continuity:
  slope, northness, eastness (aspect), ruggedness (TRI), tpi (topo position),
  nbhd_wildland_2km (total burnable-fuel fraction).
Computed at the exact v10b points, merged. Terrain is pure topography -> any gain here
is unimpeachable (no reporting bias possible).
  python Dataget_v11f_extra.py --test | python Dataget_v11f_extra.py
"""
import ee, sys, math
from Dataget_fpafod_v10 import POS, pos_fc, negatives, SRTM, LC, POS_BATCHES, NEG_BATCHES

EXPORT_FOLDER = 'Fire_v11f_extra'
COLUMNS = ['lon', 'lat', 'label', 'cause', 'year', 'doy',
           'slope', 'northness', 'eastness', 'ruggedness', 'tpi', 'nbhd_wildland_2km']
SLOPE = ee.Terrain.slope(SRTM)
ASP = ee.Terrain.aspect(SRTM).multiply(math.pi / 180.0)
NORTHNESS = ASP.cos().rename('northness')
EASTNESS = ASP.sin().rename('eastness')
TRI = SRTM.reduceNeighborhood(ee.Reducer.stdDev(), ee.Kernel.square(2)).rename('tri')
TPI = SRTM.subtract(SRTM.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(5))).rename('tpi')
WILDLAND = LC.eq(41).Or(LC.eq(42)).Or(LC.eq(43)).Or(LC.eq(52)).Or(LC.eq(71)).Or(LC.eq(90)).Or(LC.eq(95))


def extra(feature):
    pt = feature.geometry(); td = ee.Date(feature.get('target_time'))

    def at(img, band, scale=90):
        return img.reduceRegion(ee.Reducer.first(), pt, scale).get(band)
    coords = pt.coordinates()
    return ee.Feature(pt, {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'), 'cause': feature.get('cause'),
        'year': td.get('year'), 'doy': td.getRelative('day', 'year'),
        'slope': at(SLOPE, 'slope'), 'northness': at(NORTHNESS, 'northness'), 'eastness': at(EASTNESS, 'eastness'),
        'ruggedness': at(TRI, 'tri'), 'tpi': at(TPI, 'tpi'),
        'nbhd_wildland_2km': WILDLAND.rename('m').reduceRegion(ee.Reducer.mean(), pt.buffer(2000), 100).get('m'),
    })


def run_test():
    one = ee.Feature(pos_fc(0, 5).map(extra).first()).toDictionary().getInfo()
    print('sample:', {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in one.items()})
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        ee.batch.Export.table.toDrive(collection=pos_fc(i * chunk, (i + 1) * chunk).map(extra), description=f'Fire_v11f_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v11f_pos_{i+1}')
    neg = negatives(303)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(extra), description=f'Fire_v11f_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v11f_neg_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
