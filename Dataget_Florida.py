# Generates Florida fire dataset by sampling fire and non-fire locations from Google Earth Engine
import ee
import sys

PROJECT_ID = 'gleaming-glass-426122-k0'
TOTAL_SAMPLES = 12000
BATCH_SIZE = 1200
NUM_BATCHES = int(TOTAL_SAMPLES / BATCH_SIZE)
KERNEL_RADIUS = 128
SCALE = 20
EXPORT_FOLDER = 'Fire_Prediction_Dataset_Florida_v1'

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

    weather_col = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
        .filterBounds(geom) \
        .filterDate(target_date, target_date.advance(1, 'day'))
    
    weather_fallback = ee.Image.constant([0, 0, 0, 0]).rename(['tmmx', 'rmin', 'vs', 'pr'])
    
    weather = ee.Image(ee.Algorithms.If(
        weather_col.size().gt(0),
        weather_col.first(),
        weather_fallback
    ))

    tmmx = weather.select('tmmx').subtract(253.15).divide(50.0).clamp(0, 1).rename('Temp_Max')
    rmin = weather.select('rmin').divide(100.0).rename('Humidity_Min')
    vs = weather.select('vs').divide(20.0).clamp(0, 1).rename('Wind_Speed')
    pr = weather.select('pr').divide(50.0).clamp(0, 1).rename('Precip')

    topo = ee.Image('USGS/SRTMGL1_003').unmask(0)
    elevation = topo.select('elevation').divide(4000.0).clamp(0, 1).rename('Elevation').float()
    slope = ee.Terrain.slope(topo.select('elevation')).divide(45.0).clamp(0, 1).rename('Slope').float()

    pop_col = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01')
    
    pop = ee.Image(ee.Algorithms.If(
        pop_col.size().gt(0), 
        pop_col.first(), 
        ee.Image.constant(0).rename('population')
    )).unmask(0)

    pop = pop.select('population').add(1).log().divide(10.0).clamp(0, 1).rename('Pop_Density').float()

    full_stack = ee.Image.cat([
        s2_normalized, ndvi, ndmi,
        tmmx, rmin, vs, pr,
        elevation, slope,
        pop
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
        'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
        'Elevation', 'Slope', 'Pop_Density', 'label'
    ]

    for i in range(NUM_BATCHES):
        batch_id = (i + 1) + 10
        current_seed = (i * 12345) + 400000
        
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
