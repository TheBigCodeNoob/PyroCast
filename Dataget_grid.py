"""
THE TOOL (part 1): export the full final feature set over a regular grid for a fixed date,
so we can render an operational ignition-risk map for any day. Reuses every feature function.
Grid: fire-prone SE (Florida + Gulf/Atlantic coast), 0.1 deg, for a peak-season day.
  python Dataget_grid.py --test | python Dataget_grid.py
"""
import ee, sys, datetime, numpy as np
from Dataget_v11e import features
from Dataget_v11h_canopy import extra as canopy_extra
from Dataget_v12_moisture import extra as moist_extra
from Dataget_v11e import COLUMNS as C_BASE

DATE = '2020-04-15'                       # peak SE spring fire season
TARGET_MS = int(datetime.datetime(2020, 4, 15, tzinfo=datetime.timezone.utc).timestamp() * 1000)
LAT0, LAT1, LON0, LON1, STEP = 24.6, 35.0, -88.0, -79.0, 0.1
EXPORT_FOLDER = 'Fire_grid'
BATCHES = 16
COLUMNS = C_BASE + ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km',
                    'ndmi', 'smap_surface', 'smap_root', 'lst_day', 'lst_night', 'et', 'pet']

_lons = np.arange(LON0, LON1, STEP)
_lats = np.arange(LAT0, LAT1, STEP)
GRID = [(round(float(lo), 3), round(float(la), 3)) for lo in _lons for la in _lats]


def grid_fc(lo, hi):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([p[0], p[1]]), {'label': 0, 'cause': -1, 'target_time': TARGET_MS})
        for p in GRID[lo:hi]])


def grid_feat(f):
    d = features(f).toDictionary().combine(canopy_extra(f).toDictionary()).combine(moist_extra(f).toDictionary())
    return ee.Feature(f.geometry(), d)


def run_test():
    one = ee.Feature(grid_fc(0, 5).map(grid_feat).first()).toDictionary().getInfo()
    print(f'{len(GRID)} grid points; sample cols present: {len([c for c in COLUMNS if c in one])}/{len(COLUMNS)}')
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    print(f'grid export: {len(GRID)} points @ {DATE}, {len(COLUMNS)} cols, {BATCHES} batches')
    chunk = (len(GRID) + BATCHES - 1) // BATCHES
    for i in range(BATCHES):
        ee.batch.Export.table.toDrive(collection=grid_fc(i * chunk, (i + 1) * chunk).map(grid_feat),
                                      description=f'Fire_grid_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_grid_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
