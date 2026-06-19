"""
v12 (last-session honest lever): MOISTURE & WATER-STRESS — the gap in our feature set.
We have greenness (NDVI/EVI), structure (canopy/treecover), drought index (PDSI), and air
weather, but no direct measure of how WET the fuel/soil is. Flammability depends on that.
  NDMI         : MODIS NBAR veg moisture (NIR-SWIR)        -> live fuel moisture
  smap_surface : SMAP surface soil moisture
  smap_root    : SMAP root-zone soil moisture
  lst_day/night: MODIS land-surface temperature (skin temp, not air)
  et / pet     : MODIS evapotranspiration & potential ET   -> water stress (et/pet)
Computed at the exact 25k v11e points, merged.
  python Dataget_v12_moisture.py --test | python Dataget_v12_moisture.py
"""
import ee, sys
from Dataget_v11e import POS, pos_fc, negatives, POS_BATCHES, NEG_BATCHES

EXPORT_FOLDER = 'Fire_v12_moisture'
COLUMNS = ['lon', 'lat', 'label', 'cause', 'year', 'doy',
           'ndmi', 'smap_surface', 'smap_root', 'lst_day', 'lst_night', 'et', 'pet']
NBAR = ee.ImageCollection('MODIS/061/MCD43A4')
SMAP = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
LST = ee.ImageCollection('MODIS/061/MOD11A1')
MET = ee.ImageCollection('MODIS/006/MOD16A2')


def extra(feature):
    pt = feature.geometry(); td = ee.Date(feature.get('target_time'))

    def at(img, band, scale):
        return img.reduceRegion(ee.Reducer.first(), pt, scale).get(band)

    nb = NBAR.filterDate(td.advance(-16, 'day'), td)
    nbimg = ee.Image(ee.Algorithms.If(nb.size().gt(0), nb.sort('system:time_start', False).first(),
            ee.Image.constant([0, 0]).rename(['Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band6'])))
    ndmi = nbimg.normalizedDifference(['Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band6']).rename('ndmi')
    sm = SMAP.filterDate(td.advance(-3, 'day'), td)
    smimg = ee.Image(ee.Algorithms.If(sm.size().gt(0), sm.sort('system:time_start', False).first(),
            ee.Image.constant([0, 0]).rename(['sm_surface', 'sm_rootzone'])))
    lst = LST.filterDate(td.advance(-8, 'day'), td)
    lstimg = ee.Image(ee.Algorithms.If(lst.size().gt(0), lst.mean(),
             ee.Image.constant([0, 0]).rename(['LST_Day_1km', 'LST_Night_1km'])))
    et = MET.filterDate(td.advance(-16, 'day'), td)
    etimg = ee.Image(ee.Algorithms.If(et.size().gt(0), et.sort('system:time_start', False).first(),
            ee.Image.constant([0, 0]).rename(['ET', 'PET'])))
    coords = pt.coordinates()
    return ee.Feature(pt, {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'), 'cause': feature.get('cause'),
        'year': td.get('year'), 'doy': td.getRelative('day', 'year'),
        'ndmi': at(ndmi, 'ndmi', 500),
        'smap_surface': at(smimg.select('sm_surface'), 'sm_surface', 10000),
        'smap_root': at(smimg.select('sm_rootzone'), 'sm_rootzone', 10000),
        'lst_day': at(lstimg.select('LST_Day_1km').multiply(0.02), 'LST_Day_1km', 1000),
        'lst_night': at(lstimg.select('LST_Night_1km').multiply(0.02), 'LST_Night_1km', 1000),
        'et': at(etimg.select('ET').multiply(0.1), 'ET', 500),
        'pet': at(etimg.select('PET').multiply(0.1), 'PET', 500),
    })


def run_test():
    one = ee.Feature(pos_fc(0, 5).map(extra).first()).toDictionary().getInfo()
    print('sample:', {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in one.items()})
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        ee.batch.Export.table.toDrive(collection=pos_fc(i * chunk, (i + 1) * chunk).map(extra), description=f'Fire_v12_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v12_pos_{i+1}')
    neg = negatives(303)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(extra), description=f'Fire_v12_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v12_neg_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
