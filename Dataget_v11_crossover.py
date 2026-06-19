"""
v11b: CASE-CROSSOVER negatives to attack the weak 'when' signal.
For each FPA-FOD fire (location L, pre-fire date D), add a negative at the SAME
location L but 1 year earlier (D-365, a non-fire day). Combined with v10b's random-
location negatives, the model must beat BOTH wrong-place AND wrong-time negatives.
Reuses the EXACT v10b fire points + the same (MODIS) feature recipe. cause=-2 marks
crossover negatives. Exports to 'Fire_v11_crossover'.

  python Dataget_v11_crossover.py --test | python Dataget_v11_crossover.py
"""
import ee, sys
from Dataget_fpafod_v10 import POS, features, COLUMNS  # same points/seeds + MODIS feature recipe; ee already initialized

EXPORT_FOLDER = 'Fire_v11_crossover'
BATCHES = 8
YEAR_MS = 365 * 24 * 3600 * 1000


def cross_fc(lo, hi):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([p[0], p[1]]), {'label': 0, 'cause': -2, 'target_time': p[2] - YEAR_MS})
        for p in POS[lo:hi]])


def run_test():
    one = ee.Feature(cross_fc(0, 5).map(features).first()).toDictionary().getInfo()
    print('crossover sample: year', one.get('year'), 'doy', one.get('doy'), 'label', one.get('label'), 'cause', one.get('cause'))
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    chunk = (len(POS) + BATCHES - 1) // BATCHES
    for i in range(BATCHES):
        ee.batch.Export.table.toDrive(collection=cross_fc(i * chunk, (i + 1) * chunk).map(features),
                                      description=f'Fire_v11cross_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v11cross_{i+1}')
    print(f"{len(POS)} crossover negatives -> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
