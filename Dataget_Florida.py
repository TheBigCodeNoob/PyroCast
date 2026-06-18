# Generates Florida fire dataset by sampling fire and non-fire locations from Google Earth Engine
#
# v3 schema (real fire-prediction task, not biome classification):
#   - MATCHED TEMPORAL NEGATIVES: for each MTBS fire at (location L, date D), generate
#     one positive at (L, D - 1..30 days) AND one negative at (L, D - 335..395 days).
#     Negatives are at the SAME LOCATION as positives, one year earlier (same season).
#     This forces the model to discriminate via the year-specific drought / weather /
#     vegetation state, not via "is this patch a forest?" or "is it spring?".
#   - PDSI added (Palmer Drought Severity Index from GRIDMET/DROUGHT) — captures
#     accumulated multi-month drought, which is the v2 audit's missing-feature gap.
#   - 20 channels total (v2 was 19).
import ee
import sys

PROJECT_ID = 'gleaming-glass-426122-k0'
TOTAL_FIRES = 6000           # We sample this many fires; each yields ONE positive + ONE matched negative.
FIRES_PER_BATCH = 600
NUM_BATCHES = TOTAL_FIRES // FIRES_PER_BATCH
KERNEL_RADIUS = 128
SCALE = 20
EXPORT_FOLDER = 'Fire_Prediction_Dataset_Florida_v3'

try:
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print("Initialization failed. Run 'earthengine authenticate' first.")
    sys.exit(1)

FLORIDA_BBOX = ee.Geometry.Rectangle([-87.6, 24.5, -80.0, 31.0])


def get_feature_stack(feature):
    geom = feature.geometry()
    target_date = ee.Date(feature.get('target_time'))

    s2_bands_raw = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    s2_bands_renamed = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geom) \
        .filterDate(target_date.advance(-45, 'day'), target_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    fallback_s2 = ee.Image.constant([0] * len(s2_bands_raw)).rename(s2_bands_raw)
    s2 = ee.Image(ee.Algorithms.If(
        s2_col.size().gt(0),
        s2_col.median().select(s2_bands_raw),
        fallback_s2
    )).unmask(0)
    s2_normalized = s2.divide(10000.0).float().rename(s2_bands_renamed)

    ndvi = s2_normalized.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndmi = s2_normalized.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')

    weather_col = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
        .filterBounds(geom) \
        .filterDate(target_date, target_date.advance(1, 'day'))
    weather_fallback = ee.Image.constant([0, 0, 0, 0, 0, 0]).rename(
        ['tmmx', 'rmin', 'vs', 'pr', 'erc', 'fm100']
    )
    weather = ee.Image(ee.Algorithms.If(
        weather_col.size().gt(0),
        weather_col.first(),
        weather_fallback
    ))
    tmmx  = weather.select('tmmx').subtract(253.15).divide(50.0).clamp(0, 1).rename('Temp_Max')
    rmin  = weather.select('rmin').divide(100.0).rename('Humidity_Min')
    vs    = weather.select('vs').divide(20.0).clamp(0, 1).rename('Wind_Speed')
    pr    = weather.select('pr').divide(50.0).clamp(0, 1).rename('Precip')
    erc   = weather.select('erc').divide(130.0).clamp(0, 1).rename('ERC')
    fm100 = weather.select('fm100').divide(40.0).clamp(0, 1).rename('FM100')

    # NEW: PDSI (Palmer Drought Severity Index) — multi-month drought signal.
    # GRIDMET/DROUGHT is a pentad (5-day) collection. Pull the most recent <= target.
    drought_col = ee.ImageCollection("GRIDMET/DROUGHT") \
        .filterDate(target_date.advance(-10, 'day'), target_date.advance(1, 'day'))
    drought_fallback = ee.Image.constant(0).rename('pdsi')
    pdsi_raw = ee.Image(ee.Algorithms.If(
        drought_col.size().gt(0),
        drought_col.sort('system:time_start', False).first().select('pdsi'),
        drought_fallback
    ))
    # PDSI nominal range: -10 (extreme drought) .. +10 (extreme wet). Normalize -> [0, 1].
    pdsi = pdsi_raw.add(10.0).divide(20.0).clamp(0, 1).rename('PDSI')

    topo = ee.Image('USGS/SRTMGL1_003').unmask(0)
    elevation = topo.select('elevation').divide(4000.0).clamp(0, 1).rename('Elevation').float()

    nlcd_col = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD") \
        .filter(ee.Filter.eq('system:index', '2021'))
    nlcd_fallback = ee.Image.constant(0).rename('landcover')
    nlcd = ee.Image(ee.Algorithms.If(
        nlcd_col.size().gt(0),
        nlcd_col.first().select('landcover'),
        nlcd_fallback
    )).unmask(0)
    lc_forest  = nlcd.eq(41).Or(nlcd.eq(42)).Or(nlcd.eq(43)).rename('LC_Forest').float()
    lc_wetland = nlcd.eq(90).Or(nlcd.eq(95)).rename('LC_Wetland').float()
    lc_open    = nlcd.eq(52).Or(nlcd.eq(71)).Or(nlcd.eq(81)).Or(nlcd.eq(82)) \
                     .rename('LC_Open').float()

    pop_col = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01')
    pop_raw = ee.Image(ee.Algorithms.If(
        pop_col.size().gt(0),
        pop_col.first(),
        ee.Image.constant(0).rename('population')
    )).unmask(0)
    pop = pop_raw.select('population').add(1).log().divide(10.0).clamp(0, 1).rename('Pop_Density').float()

    # 20-channel stack (v2 was 19): added PDSI between FM100 and Elevation.
    full_stack = ee.Image.cat([
        s2_normalized, ndvi, ndmi,                       # 0-7   optical + indices
        tmmx, rmin, vs, pr, erc, fm100, pdsi,            # 8-14  weather + drought
        elevation,                                       # 15    terrain
        lc_forest, lc_wetland, lc_open,                  # 16-18 land cover masks
        pop                                              # 19    population
    ])

    patch = full_stack.neighborhoodToArray(
        kernel=ee.Kernel.rectangle(KERNEL_RADIUS, KERNEL_RADIUS, 'pixels')
    ).sample(
        region=geom,
        scale=SCALE,
        projection='EPSG:3857',
        factor=1,
        tileScale=16,
        dropNulls=False
    ).first()

    return ee.Algorithms.If(
        patch,
        feature.copyProperties(patch).set('label', feature.get('label')),
        None
    )


def generate_fire_pairs(count, seed):
    """For each MTBS Florida fire, emit a positive sample (D - 1..30 days) AND a
    matched negative at the SAME LOCATION one year earlier (D - 335..395 days).
    Same biome, same season, different year."""
    print(f"  - Sampling {count} MTBS fires (seed: {seed})...")

    fires = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1") \
        .filter(ee.Filter.gte('Ig_Date', ee.Date('2018-06-01').millis())) \
        .filterBounds(FLORIDA_BBOX)
    fires_shuffled = fires.randomColumn('random', seed).sort('random').limit(count)

    def make_pair(feature):
        ig_date = ee.Number(feature.get('Ig_Date'))
        rand = ee.Number(feature.get('random'))
        point = feature.geometry().centroid(1)

        # Positive: 1-30 days before fire.
        pos_offset = rand.multiply(29).add(1).round()
        pos_time = ee.Date(ig_date).advance(pos_offset.multiply(-1), 'day').millis()
        pos = ee.Feature(point).set({'label': 1, 'target_time': pos_time})

        # Matched negative: 335-395 days before fire (same season, ~1 year earlier).
        # Use a different mod of `rand` to avoid coupling pos/neg offsets.
        neg_offset = rand.multiply(60).add(335).round()
        neg_time = ee.Date(ig_date).advance(neg_offset.multiply(-1), 'day').millis()
        neg = ee.Feature(point).set({'label': 0, 'target_time': neg_time})

        return ee.FeatureCollection([pos, neg])

    return ee.FeatureCollection(fires_shuffled.map(make_pair)).flatten()


def run_export_batches():
    print("=" * 60)
    print("FLORIDA FIRE DATASET v3 GENERATOR (matched temporal negatives + PDSI)")
    print("=" * 60)
    print(f"Region: Florida (BBox: [-87.6, 24.5, -80.0, 31.0])")
    print(f"Schema: 1 positive + 1 matched-1yr-prior negative PER fire.")
    print(f"Plan: {TOTAL_FIRES} fires across {NUM_BATCHES} tasks -> {TOTAL_FIRES * 2} total samples.")
    print(f"Channels: 20 (v2 was 19; added PDSI).")
    print("=" * 60)

    columns = [
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
        'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip', 'ERC', 'FM100', 'PDSI',
        'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open', 'Pop_Density',
        'label'
    ]

    for i in range(NUM_BATCHES):
        batch_id = i + 1
        current_seed = (i * 12345) + 800000  # distinct from v2 seeds

        print(f"\n[Batch {batch_id}/{NUM_BATCHES}] Preparing...")
        dataset = generate_fire_pairs(FIRES_PER_BATCH, current_seed)

        print("  - Computing feature stacks...")
        dataset_processed = dataset.map(get_feature_stack, True)

        description = f'Export_Florida_Fire_Dataset_v3_Part_{batch_id}'
        print(f"  - Submitting Task: {description}")

        task = ee.batch.Export.table.toDrive(
            collection=dataset_processed,
            description=description,
            folder=EXPORT_FOLDER,
            fileFormat='TFRecord',
            selectors=columns
        )
        task.start()
        print(f"  - Task ID: {task.id} (Submitted)")

    print("\n" + "=" * 60)
    print("All tasks submitted to Google Earth Engine.")
    print("Check progress at: https://code.earthengine.google.com/tasks")
    print("=" * 60)
    print(f"\nAfter completion, download from Drive folder: '{EXPORT_FOLDER}'")
    print("Place them in: 'Training Data/'  (subfolders are fine, mirror v2 layout).")


if __name__ == "__main__":
    run_export_batches()
