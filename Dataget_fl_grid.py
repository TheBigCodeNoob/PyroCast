"""
Dense Florida risk grid for the web demo / scheduled job. Exports the full feature stack
over a fine FL grid for a given date (default: ~10 days ago, for data availability).
  python Dataget_fl_grid.py [YYYY-MM-DD]
"""
import ee, sys, datetime, numpy as np
ee.Initialize(project='gleaming-glass-426122-k0')
from Dataget_v11e import features
from Dataget_v11h_canopy import extra as canopy_extra
from Dataget_v12_moisture import extra as moist_extra
from Dataget_v11e import COLUMNS as C_BASE

_arg = next((a for a in sys.argv[1:] if a[0] != '-' and '-' in a), None)
if _arg:
    DATE = _arg
else:
    _d = datetime.datetime(2026, 6, 19) - datetime.timedelta(days=10)
    DATE = _d.strftime('%Y-%m-%d')
_y, _m, _dd = (int(x) for x in DATE.split('-'))
TARGET_MS = int(datetime.datetime(_y, _m, _dd, tzinfo=datetime.timezone.utc).timestamp() * 1000)
LAT0, LAT1, LON0, LON1, STEP = 24.5, 31.05, -87.7, -79.8, 0.04
EXPORT_FOLDER = 'Fire_FLgrid'
BATCHES = 18
BUILT = ee.Image(ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S').filterDate('2019-01-01', '2021-06-01').first()).select('built_surface').rename('built').unmask(0)
NLCD = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq('system:index', '2021')).first().select('landcover')
LANDMASK = NLCD.neq(11)  # exclude open water
COLUMNS = C_BASE + ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km',
                    'ndmi', 'smap_surface', 'smap_root', 'lst_day', 'lst_night', 'et', 'pet', 'built']

_lons = np.arange(LON0, LON1, STEP); _lats = np.arange(LAT0, LAT1, STEP)
GRID = [(round(float(lo), 3), round(float(la), 3)) for lo in _lons for la in _lats]


def grid_fc(lo, hi):
    return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p[0], p[1]]), {'label': 0, 'cause': -1, 'target_time': TARGET_MS}) for p in GRID[lo:hi]])


def grid_feat(f):
    d = features(f).toDictionary().combine(canopy_extra(f).toDictionary()).combine(moist_extra(f).toDictionary())
    d = d.set('built', BUILT.reduceRegion(ee.Reducer.first(), f.geometry(), 100).get('built'))
    return ee.Feature(f.geometry(), d)


def run_test():
    one = ee.Feature(grid_fc(0, 50).map(grid_feat).filter(ee.Filter.notNull(['NDVI'])).first()).toDictionary().getInfo()
    print(f'{len(GRID)} grid points @ {DATE}; sample land point cols {len([c for c in COLUMNS if c in one])}/{len(COLUMNS)}')
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    print(f'FL grid: {len(GRID)} points @ {DATE}, {BATCHES} batches')
    chunk = (len(GRID) + BATCHES - 1) // BATCHES
    for i in range(BATCHES):
        fc = grid_fc(i * chunk, (i + 1) * chunk).filterBounds(LANDMASK.geometry() if False else ee.Geometry.Rectangle([LON0, LAT0, LON1, LAT1]))
        ee.batch.Export.table.toDrive(collection=fc.map(grid_feat), description=f'Fire_FLgrid_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_FLgrid_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'  (date {DATE})")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
