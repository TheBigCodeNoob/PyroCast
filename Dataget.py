import ee
import sys

# ================= CONFIGURATION =================
# INSERT YOUR PROJECT ID HERE
PROJECT_ID = 'gleaming-glass-426122-k0' 

# Dataset Configuration
# We will split this total into smaller "bite-sized" tasks
TOTAL_SAMPLES = 12000 
BATCH_SIZE = 1200      # 2000 samples is safe for one task
NUM_BATCHES = int(TOTAL_SAMPLES / BATCH_SIZE)

# Patch Configuration
# Kernel Radius 128 = 257x257 pixels
KERNEL_RADIUS = 128 
SCALE = 20         
EXPORT_FOLDER = 'Fire_Prediction_Dataset_v3'

# ================= INITIALIZATION =================
try:
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print("Initialization failed. Run 'earthengine authenticate' first.")
    sys.exit(1)

# ================= CORE FUNCTIONS =================

def get_feature_stack(feature):
    """
    Given a feature (point + target_time), returns a stacked image of all data layers.
    Includes normalization and memory-safe casting.
    """
    geom = feature.geometry()
    target_date = ee.Date(feature.get('target_time'))
    
    # --- 1. OPTICAL (Sentinel-2) ---
    s2_bands_raw = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    s2_bands_renamed = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geom) \
        .filterDate(target_date.advance(-45, 'day'), target_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    # SAFEGUARD: Fallback image if collection is empty
    fallback_s2 = ee.Image.constant([0] * len(s2_bands_raw)).rename(s2_bands_raw)

    s2 = ee.Algorithms.If(
        s2_col.size().gt(0),
        s2_col.median().select(s2_bands_raw),
        fallback_s2
    )
    s2 = ee.Image(s2).unmask(0)
    s2_normalized = s2.divide(10000.0).float().rename(s2_bands_renamed)

    # Indices
    ndvi = s2_normalized.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndmi = s2_normalized.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')

    # --- 2. WEATHER (GRIDMET) ---
    weather_col = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
        .filterBounds(geom) \
        .filterDate(target_date, target_date.advance(1, 'day'))
    
    weather_fallback = ee.Image.constant([0, 0, 0, 0]).rename(['tmmx', 'rmin', 'vs', 'pr'])
    
    weather = ee.Image(ee.Algorithms.If(
        weather_col.size().gt(0),
        weather_col.first(),
        weather_fallback
    ))

    # Normalize Weather
    tmmx = weather.select('tmmx').subtract(253.15).divide(50.0).clamp(0, 1).rename('Temp_Max')
    rmin = weather.select('rmin').divide(100.0).rename('Humidity_Min')
    vs = weather.select('vs').divide(20.0).clamp(0, 1).rename('Wind_Speed')
    pr = weather.select('pr').divide(50.0).clamp(0, 1).rename('Precip')

    # --- 3. TOPOGRAPHY ---
    topo = ee.Image('USGS/SRTMGL1_003').unmask(0)
    elevation = topo.select('elevation').divide(4000.0).clamp(0, 1).rename('Elevation').float()
    slope = ee.Terrain.slope(topo.select('elevation')).divide(45.0).clamp(0, 1).rename('Slope').float()

    # --- 4. POPULATION ---
    pop_col = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01')
    
    pop = ee.Image(ee.Algorithms.If(
        pop_col.size().gt(0), 
        pop_col.first(), 
        ee.Image.constant(0).rename('population')
    )).unmask(0)

    # Log1p implementation: log(x + 1)
    pop = pop.select('population').add(1).log().divide(10.0).clamp(0, 1).rename('Pop_Density').float()

    # --- 5. STACK ALL ---
    full_stack = ee.Image.cat([
        s2_normalized, ndvi, ndmi,  # 8 Bands
        tmmx, rmin, vs, pr,         # 4 Bands
        elevation, slope,           # 2 Bands
        pop                         # 1 Band
    ])
    
    # 6. EXTRACT PATCH AS ARRAY
    # TILESCALE 16 IS CRITICAL TO PREVENT MEMORY CRASHES
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
    
    # Return result or None (which will be dropped later)
    return ee.Algorithms.If(
        patch,
        feature.copyProperties(patch).set('label', feature.get('label')),
        None
    )

# ================= SAMPLING LOGIC (WITH SEEDS) =================

def generate_positive_samples(count, seed):
    print(f"  - Finding fires (Seed: {seed})...")
    fires = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1") \
        .filter(ee.Filter.gte('Ig_Date', ee.Date('2018-01-01').millis())) \
        .filterBounds(ee.Geometry.Rectangle([-124, 24, -67, 49]))
    
    # Use the seed for the random column to get different fires per batch
    fires_shuffled = fires.randomColumn('random', seed).sort('random').limit(count)

    def setup_fire_feature(feature):
        ig_date = ee.Number(feature.get('Ig_Date'))
        # Use the existing random column for the day offset too
        days_before = ee.Number(feature.get('random')).multiply(29).add(1).round()
        target_time = ee.Date(ig_date).advance(days_before.multiply(-1), 'day').millis()
        point = feature.geometry().centroid(1)
        return ee.Feature(point).set({'label': 1, 'target_time': target_time})

    return fires_shuffled.map(setup_fire_feature)

def generate_negative_samples(count, seed):
    print(f"  - Generating non-fires (Seed: {seed})...")
    roi = ee.Geometry.Rectangle([-124, 24, -67, 49])
    
    # randomPoints allows a seed
    points = ee.FeatureCollection.randomPoints(roi, count, seed)

    def setup_random_feature(feature):
        # Create pseudo-random date using coordinates + batch seed
        geo_seed = ee.Number(feature.geometry().coordinates().get(0)) \
            .add(feature.geometry().coordinates().get(1)) \
            .add(seed) 
            
        start = ee.Date('2019-01-01').millis()
        end = ee.Date('2022-01-01').millis()
        diff = ee.Number(end).subtract(start)
        random_time = ee.Number(start).add(diff.multiply(geo_seed.sin().abs()))
        return feature.set({'label': 0, 'target_time': random_time})

    return points.map(setup_random_feature)

# ================= BATCH EXECUTION =================

def run_export_batches():
    print(f"Plan: Splitting {TOTAL_SAMPLES} samples into {NUM_BATCHES} tasks of {BATCH_SIZE} each.")
    
    columns = [
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
        'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
        'Elevation', 'Slope', 'Pop_Density', 'label'
    ]

    for i in range(NUM_BATCHES):
        batch_id = i + 1
        current_seed = (i * 12345) + 271590  # Deterministic but different seed per batch
        
        print(f"\n[Batch {batch_id}/{NUM_BATCHES}] Preparing...")
        
        n_pos = int(BATCH_SIZE / 2)
        n_neg = int(BATCH_SIZE / 2)
        
        # Generate unique samples for this batch
        pos_ds = generate_positive_samples(n_pos, current_seed)
        neg_ds = generate_negative_samples(n_neg, current_seed)
        dataset = pos_ds.merge(neg_ds)
        
        print("  - Computing feature stacks...")
        # dataset.map(..., True) removes any nulls automatically
        dataset_processed = dataset.map(get_feature_stack, True)
        
        description = f'Export_Fire_Dataset_Part_{batch_id}'
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

    print("\nAll tasks submitted. Check progress at: https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    run_export_batches()