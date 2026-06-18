"""
v4 Florida/Southeast fire dataset generator.

Goal: break past the v3 honest-AUC ceiling (~0.70) by fixing the two data limits
the v3 audit + tabular search exposed:
  1) Too few unique locations (v3 had ~474) -> poor spatial generalization.
     FIX: Southeast US region, MTBS 2017+ -> ~3,200 unique fires (6x more), all in
     the Sentinel-2 era. Each fire is used ONCE (v3 re-sampled the same ~533 fires
     10x); batches are disjoint partitions by a random column.
  2) Single-snapshot weather -> no accumulated-drying / trend signal.
     FIX: add temporal channels (30/90-day precip, 30-day ERC/FM100/VPD, PDSI 90d
     ago for drought trajectory, NDVI 90d ago for fuel curing).

Keeps v3's honest matched-temporal-negative design (positive 1-30d pre-fire,
negative SAME location ~1yr earlier) so the biome shortcut stays eliminated.
Drops the dead all-zero Pop_Density channel. 26 channels total.

Run:
  python Dataget_Florida_v4.py --test     # sanity-check feature stack on a few fires (getInfo)
  python Dataget_Florida_v4.py            # submit the full export to Drive
"""
import ee
import sys

PROJECT_ID = 'gleaming-glass-426122-k0'
NUM_BATCHES = 12                 # disjoint random partitions of the fire set
KERNEL_RADIUS = 128
SCALE = 20
EXPORT_FOLDER = 'Fire_Prediction_Dataset_Florida_v4'
MTBS = "USFS/GTAC/MTBS/burned_area_boundaries/v1"
FIRE_START = '2017-01-01'        # S2 era; matched negatives land in 2016+ (S2 fallback handles thin coverage)

try:
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print("Initialization failed. Run 'earthengine authenticate' first.")
    print(e)
    sys.exit(1)

# Southeast US (FL, GA, AL, SC, NC, MS, LA, TN, parts of AR/VA).
SE_BBOX = ee.Geometry.Rectangle([-94.5, 24.5, -75.0, 37.5])

COLUMNS = [
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',          # 0-7  optical
    'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip', 'ERC', 'FM100', 'PDSI',  # 8-14 same-day wx/drought
    'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open',                        # 15-18 static
    'Precip_30d', 'Precip_90d', 'ERC_30d', 'FM100_30d', 'VPD', 'PDSI_90dago', 'NDVI_90d',  # 19-25 temporal
    'label',
]


def get_feature_stack(feature):
    geom = feature.geometry()
    target_date = ee.Date(feature.get('target_time'))

    s2_bands_raw = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    s2_bands_renamed = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

    def s2_median(d_start, d_end):
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(geom).filterDate(d_start, d_end)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        fb = ee.Image.constant([0] * len(s2_bands_raw)).rename(s2_bands_raw)
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median().select(s2_bands_raw), fb)).unmask(0)
        return img.divide(10000.0).float().rename(s2_bands_renamed)

    s2 = s2_median(target_date.advance(-45, 'day'), target_date)
    ndvi = s2.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndmi = s2.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')

    # Fuel curing: NDVI ~90 days earlier (a composite ending 60d before target).
    s2_old = s2_median(target_date.advance(-105, 'day'), target_date.advance(-60, 'day'))
    ndvi_90d = s2_old.normalizedDifference(['NIR', 'Red']).rename('NDVI_90d')

    # ---- Same-day GRIDMET weather/fire-danger ----
    gm = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(geom)
    wx_bands = ['tmmx', 'rmin', 'vs', 'pr', 'erc', 'fm100', 'vpd']
    wx_fb = ee.Image.constant([0] * len(wx_bands)).rename(wx_bands)
    day = gm.filterDate(target_date, target_date.advance(1, 'day'))
    weather = ee.Image(ee.Algorithms.If(day.size().gt(0), day.first(), wx_fb))
    tmmx = weather.select('tmmx').subtract(253.15).divide(50.0).clamp(0, 1).rename('Temp_Max')
    rmin = weather.select('rmin').divide(100.0).clamp(0, 1).rename('Humidity_Min')
    vs = weather.select('vs').divide(20.0).clamp(0, 1).rename('Wind_Speed')
    pr = weather.select('pr').divide(50.0).clamp(0, 1).rename('Precip')
    erc = weather.select('erc').divide(130.0).clamp(0, 1).rename('ERC')
    fm100 = weather.select('fm100').divide(40.0).clamp(0, 1).rename('FM100')

    # ---- Temporal aggregates (GRIDMET is daily & dense for 2017+) ----
    def window(days):
        return gm.filterDate(target_date.advance(-days, 'day'), target_date)
    g30 = window(30)
    g90 = window(90)
    g30n = g30.size().gt(0)
    g90n = g90.size().gt(0)
    precip_30d = ee.Image(ee.Algorithms.If(g30n, g30.select('pr').sum(), ee.Image.constant(0))) \
        .divide(300.0).clamp(0, 1).rename('Precip_30d')
    precip_90d = ee.Image(ee.Algorithms.If(g90n, g90.select('pr').sum(), ee.Image.constant(0))) \
        .divide(800.0).clamp(0, 1).rename('Precip_90d')
    erc_30d = ee.Image(ee.Algorithms.If(g30n, g30.select('erc').mean(), ee.Image.constant(0))) \
        .divide(130.0).clamp(0, 1).rename('ERC_30d')
    fm100_30d = ee.Image(ee.Algorithms.If(g30n, g30.select('fm100').mean(), ee.Image.constant(0))) \
        .divide(40.0).clamp(0, 1).rename('FM100_30d')
    vpd_30d = ee.Image(ee.Algorithms.If(g30n, g30.select('vpd').mean(), ee.Image.constant(0))) \
        .divide(5.0).clamp(0, 1).rename('VPD')

    # ---- Drought (GRIDMET/DROUGHT pentad) now and ~90 days ago ----
    def pdsi_at(center, name):
        col = ee.ImageCollection("GRIDMET/DROUGHT").filterDate(
            center.advance(-10, 'day'), center.advance(1, 'day'))
        img = ee.Image(ee.Algorithms.If(
            col.size().gt(0), col.sort('system:time_start', False).first().select('pdsi'),
            ee.Image.constant(0)))
        return img.add(10.0).divide(20.0).clamp(0, 1).rename(name)
    pdsi = pdsi_at(target_date, 'PDSI')
    pdsi_90 = pdsi_at(target_date.advance(-90, 'day'), 'PDSI_90dago')

    # ---- Static ----
    elevation = ee.Image('USGS/SRTMGL1_003').unmask(0).select('elevation') \
        .divide(4000.0).clamp(0, 1).rename('Elevation').float()
    nlcd_col = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq('system:index', '2021'))
    nlcd = ee.Image(ee.Algorithms.If(nlcd_col.size().gt(0), nlcd_col.first().select('landcover'),
                                     ee.Image.constant(0).rename('landcover'))).unmask(0)
    lc_forest = nlcd.eq(41).Or(nlcd.eq(42)).Or(nlcd.eq(43)).rename('LC_Forest').float()
    lc_wetland = nlcd.eq(90).Or(nlcd.eq(95)).rename('LC_Wetland').float()
    lc_open = nlcd.eq(52).Or(nlcd.eq(71)).Or(nlcd.eq(81)).Or(nlcd.eq(82)).rename('LC_Open').float()

    full_stack = ee.Image.cat([
        s2, ndvi, ndmi,
        tmmx, rmin, vs, pr, erc, fm100, pdsi,
        elevation, lc_forest, lc_wetland, lc_open,
        precip_30d, precip_90d, erc_30d, fm100_30d, vpd_30d, pdsi_90, ndvi_90d,
    ])

    patch = full_stack.neighborhoodToArray(
        kernel=ee.Kernel.rectangle(KERNEL_RADIUS, KERNEL_RADIUS, 'pixels')
    ).sample(region=geom, scale=SCALE, projection='EPSG:3857', factor=1, tileScale=16, dropNulls=False).first()

    return ee.Algorithms.If(
        patch,
        feature.copyProperties(patch).set('label', feature.get('label')),
        None,
    )


def fires_in_region():
    return (ee.FeatureCollection(MTBS)
            .filter(ee.Filter.gte('Ig_Date', ee.Date(FIRE_START).millis()))
            .filterBounds(SE_BBOX)
            .randomColumn('part', 42)      # for disjoint batch partitioning
            .randomColumn('roff', 777))    # decoupled randomness for temporal offsets


def make_pairs(fires):
    def make_pair(feature):
        ig_date = ee.Number(feature.get('Ig_Date'))
        r = ee.Number(feature.get('roff'))
        point = feature.geometry().centroid(1)
        pos_offset = r.multiply(29).add(1).round()
        pos_time = ee.Date(ig_date).advance(pos_offset.multiply(-1), 'day').millis()
        pos = ee.Feature(point).set({'label': 1, 'target_time': pos_time})
        neg_offset = r.multiply(60).add(335).round()
        neg_time = ee.Date(ig_date).advance(neg_offset.multiply(-1), 'day').millis()
        neg = ee.Feature(point).set({'label': 0, 'target_time': neg_time})
        return ee.FeatureCollection([pos, neg])
    return ee.FeatureCollection(fires.map(make_pair)).flatten()


def run_test():
    print("TEST MODE: building feature stack for 3 fires, fetching one sample via getInfo...")
    fires = fires_in_region()
    total = fires.size().getInfo()
    print(f"  Total fires in region/period: {total}")
    sample = make_pairs(fires.limit(3))
    print(f"  Pairs from 3 fires: {sample.size().getInfo()} (expect 6)")
    stacked = sample.map(get_feature_stack, True)
    one = ee.Feature(stacked.first())
    props = one.toDictionary().getInfo()
    keys = sorted(props.keys())
    print(f"  Sample feature has {len(keys)} properties.")
    missing = [c for c in COLUMNS if c not in props]
    print(f"  Expected columns present? missing={missing}")
    arr = props.get('NDVI_90d')
    print(f"  NDVI_90d type: {type(arr).__name__}; label={props.get('label')}")
    print("  OK" if not missing else "  PROBLEM: missing columns")


def run_export():
    print("=" * 60)
    print("FLORIDA/SE FIRE DATASET v4 (temporal trends + many more unique fires)")
    print("=" * 60)
    fires = fires_in_region()
    total = fires.size().getInfo()
    print(f"Region: Southeast US | MTBS {FIRE_START}+ | {total} unique fires")
    print(f"Schema: 1 positive + 1 matched-1yr-prior negative per fire -> {total*2} samples")
    print(f"Channels: {len(COLUMNS)-1} (v3 was 20; dropped Pop_Density, added 7 temporal)")
    print("=" * 60)

    for i in range(NUM_BATCHES):
        lo = i / NUM_BATCHES
        hi = (i + 1) / NUM_BATCHES
        batch_fires = fires.filter(ee.Filter.And(
            ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        dataset = make_pairs(batch_fires).map(get_feature_stack, True)
        description = f'Export_Florida_Fire_Dataset_v4_Part_{i+1}'
        print(f"[Batch {i+1}/{NUM_BATCHES}] part in [{lo:.3f},{hi:.3f}) -> submitting {description}")
        task = ee.batch.Export.table.toDrive(
            collection=dataset, description=description, folder=EXPORT_FOLDER,
            fileFormat='TFRecord', selectors=COLUMNS)
        task.start()
        print(f"  Task ID: {task.id} (Submitted)")

    print("\n" + "=" * 60)
    print("All v4 tasks submitted. Monitor at https://code.earthengine.google.com/tasks")
    print(f"Download from Drive folder: '{EXPORT_FOLDER}' into 'Training Data Florida/v4/'")
    print("=" * 60)


if __name__ == '__main__':
    if '--test' in sys.argv:
        run_test()
    else:
        run_export()
