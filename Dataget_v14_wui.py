"""
v14: WILDLAND-URBAN INTERFACE (WUI) — the single feature the winning human-ignition models
(California 0.84, Europe 0.829) are built on, and the one PyroCast is missing. DistDev only
knows distance-to-development in the aggregate; it is blind to the INTERMIX: scattered rural
housing embedded inside wildland vegetation, which is exactly where most human ignitions start
(escaped debris burns, equipment, vehicles at the house-meets-woods edge).

We reconstruct the WUI following the Radeloff et al. / USFS SILVIS definition — housing presence
intersected with wildland vegetation cover — from GEE-native layers (NLCD land cover, WorldPop
population, GHSL built-surface):
  WILD           : NLCD wildland veg (forest/shrub/grass/wetland — FL sawgrass counts)
  HAS_HOUSING    : WorldPop >= ~15 persons/km^2  OR  any GHSL built surface
  WUI (intermix) : housing present AND >50% wildland veg within 500 m
Features at each of the 25k v11e points:
  dist_wui       : km to nearest WUI cell            (the literature's headline predictor)
  wui            : is this point itself WUI (0/1)
  wui_2km        : fraction of the 2 km neighborhood that is WUI
  wild_1km       : wildland-veg fraction within 1 km (fuel availability at the interface)
  intermix_int   : intermix intensity = log(1+pop/km^2) x local wildland fraction
Computed at the same points as v11e, merged on lon_lat.
  python Dataget_v14_wui.py --test | python Dataget_v14_wui.py
"""
import ee, sys
from Dataget_v11e import POS, pos_fc, negatives, POS_BATCHES, NEG_BATCHES

EXPORT_FOLDER = 'Fire_v14_wui'
COLUMNS = ['lon', 'lat', 'label', 'cause', 'year', 'doy',
           'dist_wui', 'wui', 'wui_2km', 'wild_1km', 'intermix_int']

# Everything is built on a 300 m CONUS-Albers metric grid. The 30 m NLCD distance-transform
# (whose input needs a 500 m focal-mean) OOM'd GEE at native res and was too slow at 100 m when
# recomputed per point; 300 m makes each point's transform ~50x cheaper and 300 m precision is
# irrelevant for a distance-to-interface feature measured in km.
BASE = ee.Projection('EPSG:5070').atScale(300)               # CONUS Albers Equal Area, 300 m
NLCD = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq('system:index', '2021')).first().select('landcover')
WILD = (NLCD.eq(41).Or(NLCD.eq(42)).Or(NLCD.eq(43)).Or(NLCD.eq(52))
        .Or(NLCD.eq(71)).Or(NLCD.eq(90)).Or(NLCD.eq(95))).rename('wild')
WILD_FRAC = WILD.reduceResolution(ee.Reducer.mean(), maxPixels=200).reproject(BASE)  # wildland fraction @300 m
POP = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01').mosaic().select('population').unmask(0)
BUILT = ee.Image(ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S').filterDate('2019-01-01', '2021-06-01').first()).select('built_surface').unmask(0)
POP_KM2 = POP.multiply(100)                                   # persons per 100 m pixel -> per km^2
HAS_HOUSING = POP_KM2.gte(15).Or(BUILT.gt(0)).reduceResolution(ee.Reducer.max(), maxPixels=200).reproject(BASE)  # any housing in cell
WILD_500 = WILD_FRAC.focal_mean(radius=600, kernelType='circle', units='meters')     # ~2 px kernel @300 m
WUI = HAS_HOUSING.And(WILD_500.gt(0.5)).reproject(BASE).rename('wui')  # intermix WUI, 0/1 everywhere
DIST_WUI = WUI.fastDistanceTransform(100, 'pixels').sqrt().multiply(300).divide(1000.0).rename('dist_wui')  # km, up to ~30 km
INTERMIX = WILD_FRAC.focal_mean(radius=1000, kernelType='circle', units='meters').multiply(POP_KM2.add(1).log()).rename('ii')


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
        'dist_wui': at(DIST_WUI, 'dist_wui', 100),
        'wui': at(WUI, 'wui', 30),
        'wui_2km': mean(WUI, 'wui', 2000, 100),
        'wild_1km': mean(WILD, 'wild', 1000, 100),
        'intermix_int': at(INTERMIX, 'ii', 100),
    })


def run_test():
    fc = pos_fc(0, 8).map(extra)
    rows = fc.getInfo()['features']
    print(f'WUI sample ({len(rows)} points):')
    for r in rows[:8]:
        p = r['properties']
        print('  ', {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in p.items()
                     if k in ('lon', 'lat', 'dist_wui', 'wui', 'wui_2km', 'wild_1km', 'intermix_int')})
    miss = [c for c in COLUMNS if c not in rows[0]['properties']]
    print('missing:', miss)


def run_export():
    print(f'v14 WUI export: pos={len(POS)} + negs, cols={len(COLUMNS)}')
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        ee.batch.Export.table.toDrive(collection=pos_fc(i * chunk, (i + 1) * chunk).map(extra), description=f'Fire_v14_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v14_pos_{i+1}')
    neg = negatives(303)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(extra), description=f'Fire_v14_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v14_neg_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
