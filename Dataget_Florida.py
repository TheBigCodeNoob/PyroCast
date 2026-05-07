# Generates Florida fire dataset by sampling fire and non-fire locations from Google Earth Engine
# v2 feature set:
#   - Drops Slope (uninformative across flat Florida terrain)
#   - Adds ERC (Energy Release Component) and FM100 (100hr dead fuel moisture) from GRIDMET
#       -> Capture accumulated drought stress / fuel dryness, the dominant FL fire driver
#   - Adds Land Cover masks (Forest / Wetland / Open) from NLCD
#       -> Lets the model differentiate flatwoods vs swamp vs scrub vs prairie
import ee
import sys

PROJECT_ID = 'gleaming-glass-426122-k0'
TOTAL_SAMPLES = 12000
BATCH_SIZE = 1200
NUM_BATCHES = int(TOTAL_SAMPLES / BATCH_SIZE)
KERNEL_RADIUS = 128
SCALE = 20
EXPORT_FOLDER = 'Fire_Prediction_Dataset_Florida_v2'

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

    s2 = ee.Algorithms.If(
        s2_col.size().gt(0),
        s2_col.median().select(s2_bands_raw),
        fallback_s2
    )
    s2 = ee.Image(s2).unmask(0)
    s2_normalized = s2.divide(10000.0).float().rename(s2_bands_renamed)

    ndvi = s2_normalized.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndmi = s2_normalized.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')

    # GRIDMET now also gives us ERC (Energy Release Component) and FM100 (100hr fuel moisture).
    # These are the two NFDRS indices most predictive of large-fire potential in Florida.
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

    tmmx = weather.select('tmmx').subtract(253.15).divide(50.0).clamp(0, 1).rename('Temp_Max')
    rmin = weather.select('rmin').divide(100.0).rename('Humidity_Min')
    vs = weather.select('vs').divide(20.0).clamp(0, 1).rename('Wind_Speed')
    pr = weather.select('pr').divide(50.0).clamp(0, 1).rename('Precip')
    # ERC: GRIDMET range ~0-130, divide by 130 to keep [0,1]
    erc = weather.select('erc').divide(130.0).clamp(0, 1).rename('ERC')
    # FM100: percent moisture, GRIDMET range ~0-40, divide by 40
    fm100 = weather.select('fm100').divide(40.0).clamp(0, 1).rename('FM100')

    # Topography: keep elevation, drop slope (Florida is flat enough that slope is noise).
    topo = ee.Image('USGS/SRTMGL1_003').unmask(0)
    elevation = topo.select('elevation').divide(4000.0).clamp(0, 1).rename('Elevation').float()

    # Land cover via NLCD CONUS — much more informative than slope in Florida.
    # Use latest NLCD release; fall back to constant 0 if missing.
    nlcd_col = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD") \
        .filter(ee.Filter.eq('system:index', '2021'))
    nlcd_fallback = ee.Image.constant(0).rename('landcover')
    nlcd = ee.Image(ee.Algorithms.If(
        nlcd_col.size().gt(0),
        nlcd_col.first().select('landcover'),
        nlcd_fallback
    )).unmask(0)

    # Forest classes: 41 deciduous, 42 evergreen, 43 mixed
    lc_forest = nlcd.eq(41).Or(nlcd.eq(42)).Or(nlcd.eq(43)) \
        .rename('LC_Forest').float()
    # Wetland classes: 90 woody wetland, 95 emergent herbaceous wetland
    lc_wetland = nlcd.eq(90).Or(nlcd.eq(95)) \
        .rename('LC_Wetland').float()
    # Open / fire-prone non-forest: 52 shrub, 71 grass, 81 pasture, 82 cultivated
    lc_open = nlcd.eq(52).Or(nlcd.eq(71)).Or(nlcd.eq(81)).Or(nlcd.eq(82)) \
        .rename('LC_Open').float()

    pop_col = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01')

    pop = ee.Image(ee.Algorithms.If(
        pop_col.size().gt(0),
        pop_col.first(),
        ee.Image.constant(0).rename('population')
    )).unmask(0)

    pop = pop.select('population').add(1).log().divide(10.0).clamp(0, 1).rename('Pop_Density').float()

    # 19-channel stack (was 15): dropped Slope, added ERC, FM100, LC_Forest, LC_Wetland, LC_Open.
    full_stack = ee.Image.cat([
        s2_normalized, ndvi, ndmi,                       # 0-7  optical + indices
        tmmx, rmin, vs, pr, erc, fm100,                  # 8-13 weather + drought
        elevation,                                       # 14   terrain
        lc_forest, lc_wetland, lc_open,                  # 15-17 land cover masks
        pop                                              # 18   population
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

def generate_positive_samples(count, seed):
    print(f"  - Finding FLORIDA fires (Seed: {seed})...")
    
    fires = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1") \
        .filter(ee.Filter.gte('Ig_Date', ee.Date('2018-01-01').millis())) \
        .filterBounds(FLORIDA_BBOX)
    
    fires_shuffled = fires.randomColumn('random', seed).sort('random').limit(count)

    def setup_fire_feature(feature):
        ig_date = ee.Number(feature.get('Ig_Date'))
        days_before = ee.Number(feature.get('random')).multiply(29).add(1).round()
        target_time = ee.Date(ig_date).advance(days_before.multiply(-1), 'day').millis()
        point = feature.geometry().centroid(1)
        return ee.Feature(point).set({'label': 1, 'target_time': target_time})

    return fires_shuffled.map(setup_fire_feature)

def generate_negative_samples(count, seed):
    print(f"  - Generating TARGETED vegetation negatives (Seed: {seed})...")
    
    lc = ee.Image("ESA/WorldCover/v100/2020").select('Map')
    
    candidates = ee.FeatureCollection.randomPoints(FLORIDA_BBOX, count * 3, seed)
    
    candidates = lc.sampleRegions(
        collection=candidates, 
        scale=10, 
        geometries=True
    )
    
    burnable = candidates.filter(ee.Filter.inList('Map', [10, 20, 30, 40, 90, 95]))
    
    final_points = burnable.limit(count)

    def setup_random_feature(feature):
        geo_seed = ee.Number(feature.geometry().coordinates().get(0)) \
            .add(feature.geometry().coordinates().get(1)) \
            .add(seed) 
            
        start = ee.Date('2019-01-01').millis()
        end = ee.Date('2022-01-01').millis()
        diff = ee.Number(end).subtract(start)
        random_time = ee.Number(start).add(diff.multiply(geo_seed.sin().abs()))
        
        return feature.set({'label': 0, 'target_time': random_time}).select(['label', 'target_time'])

    return final_points.map(setup_random_feature)

def run_export_batches():
    print("="*60)
    print("FLORIDA FIRE DATASET GENERATOR")
    print("="*60)
    print(f"Region: Florida Only (Bounding Box: [-87.6, 24.5, -80.0, 31.0])")
    print(f"Plan: Splitting {TOTAL_SAMPLES} samples into {NUM_BATCHES} tasks of {BATCH_SIZE} each.")
    print("="*60)
    
    columns = [
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
        'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip', 'ERC', 'FM100',
        'Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open', 'Pop_Density',
        'label'
    ]

    for i in range(NUM_BATCHES):
        # Clean numbering for v2 folder (v1 had a +10 offset for re-run continuity).
        batch_id = i + 1
        current_seed = (i * 12345) + 700000
        
        print(f"\n[Batch {batch_id}/{NUM_BATCHES}] Preparing...")
        
        n_pos = int(BATCH_SIZE / 2)
        n_neg = int(BATCH_SIZE / 2)
        
        pos_ds = generate_positive_samples(n_pos, current_seed)
        neg_ds = generate_negative_samples(n_neg, current_seed)
        dataset = pos_ds.merge(neg_ds)
        
        print("  - Computing feature stacks...")
        dataset_processed = dataset.map(get_feature_stack, True)
        
        description = f'Export_Florida_Fire_Dataset_Part_{batch_id}'
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

    print("\n" + "="*60)
    print("All tasks submitted to Google Earth Engine.")
    print("Check progress at: https://code.earthengine.google.com/tasks")
    print("="*60)
    print(f"\nAfter completion, download files from Google Drive folder: '{EXPORT_FOLDER}'")
    print("Then place them in: 'Training Data Florida/' folder")

if __name__ == "__main__":
    run_export_batches()
