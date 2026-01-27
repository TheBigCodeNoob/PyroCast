import ee
import numpy as np
import concurrent.futures
import urllib3
import os
import json

# Increase urllib3 connection pool size to match our parallel workers
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class GEEService:
    def __init__(self):
        try:
            # Check for service account credentials (for deployment)
            if os.getenv('GEE_SERVICE_ACCOUNT'):
                credentials_json = os.getenv('GEE_SERVICE_ACCOUNT')
                credentials = json.loads(credentials_json)
                service_account = credentials['client_email']
                credentials_obj = ee.ServiceAccountCredentials(service_account, key_data=credentials_json)
                ee.Initialize(credentials=credentials_obj, project='gleaming-glass-426122-k0')
                print(f"Google Earth Engine initialized with service account: {service_account}")
            else:
                # Local development - use default authentication
                ee.Initialize(project='gleaming-glass-426122-k0')
                print("Google Earth Engine initialized with default credentials.")
        except Exception as e:
            print(f"GEE Initialization failed: {e}")
            print("Try running `earthengine authenticate` or set GEE_SERVICE_ACCOUNT environment variable.")
            # We don't raise here to allow the app to start, but predictions will fail.

        self.cache = {}
        # Target image size for the model
        self.IMG_SIZE = 256
        # Scale from Dataget.py
        self.SCALE = 20 
        
        # Bands expected by the model (15 channels)
        self.BANDS = [
            'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDMI',
            'Temp_Max', 'Humidity_Min', 'Wind_Speed', 'Precip',
            'Elevation', 'Slope', 'Pop_Density'
        ]

    def _get_gfs_daily(self, geom, date_obj):
        """
        Fetches the most recent valid GFS snapshot for the target date.
        Uses 'Latest Available' logic rather than daily aggregation to ensure valid data.
        """
        # Look for GFS data starting from the target date, going back 2 days (to ensure we find a run)
        # We prefer the closest run to the target time.
        gfs_col = ee.ImageCollection("NOAA/GFS0P25") \
            .filterBounds(geom) \
            .filterDate(date_obj.advance(-2, 'day'), date_obj.advance(1, 'day')) \
            .sort('system:time_start', False) # Newest first
        
        # Check if empty
        if gfs_col.size().getInfo() == 0:
            print("GFS Collection empty for requested window.")
            return None

        # Take the most recent snapshot available
        # This guarantees a coherent image rather than a mix of masked pixels
        snapshot = gfs_col.first()
        
        print(f"Using GFS Snapshot timestamp: {snapshot.date().format().getInfo()}")

        # Helper to calculate Wind Speed
        u = snapshot.select('u_component_of_wind_10m_above_ground')
        v = snapshot.select('v_component_of_wind_10m_above_ground')
        vs = u.hypot(v).rename('vs')

        # Select bands directly from this valid snapshot
        # tmmx (Temperature)
        # CRITICAL FIX: GFS 'temperature_2m_above_ground' is returning Celsius (~0-40) not Kelvin (~273-310).
        # We must convert to Kelvin to match GRIDMET and the model's normalization logic.
        # Logic: If value < 150, assume Celsius and add 273.15.
        tmmx_raw = snapshot.select('temperature_2m_above_ground').selfMask() # Treat 0 as missing/masked
        tmmx = tmmx_raw.where(tmmx_raw.lt(150), tmmx_raw.add(273.15)).rename('tmmx')
        
        rmin = snapshot.select('relative_humidity_2m_above_ground').rename('rmin')
        
        # Precip in GFS is rate (kg/m^2/s). Convert to daily equivalent (mm).
        pr = snapshot.select('precipitation_rate').multiply(86400).rename('pr')

        # Combine
        gfs_img = ee.Image.cat([tmmx, rmin, vs, pr])
        
        # Resample for smoothness
        gfs_img = gfs_img.resample('bilinear')
        
        # Note: We removed the "lazy" clamp. 
        # If this works, tmmx will be ~280-310K. If it fails, it will be 0.
        return gfs_img

    def _get_image_stack(self, geom, date_obj, weather_img=None):
        """
        Replicates logic from Dataget.py to build the 15-band feature stack.
        Accepts optional weather_img to override GRIDMET fetching.
        """
        # --- 1. OPTICAL (Sentinel-2) ---
        s2_bands_raw = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        s2_bands_renamed = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

        s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geom) \
            .filterDate(date_obj.advance(-45, 'day'), date_obj) \
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

        # --- 2. WEATHER ---
        if weather_img is None:
            # Default: Fetch GRIDMET
            weather_col = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
                .filterBounds(geom) \
                .filterDate(date_obj, date_obj.advance(1, 'day'))
            
            weather_fallback = ee.Image.constant([0, 0, 0, 0]).rename(['tmmx', 'rmin', 'vs', 'pr'])
            
            weather = ee.Image(ee.Algorithms.If(
                weather_col.size().gt(0),
                weather_col.first(),
                weather_fallback
            ))
        else:
            # Use provided GFS image
            weather = weather_img

        # Normalize Weather (Standard for both GRIDMET and GFS units)
        # Temp (Kelvin) -> Normalized
        tmmx = weather.select('tmmx').subtract(253.15).divide(50.0).clamp(0, 1).rename('Temp_Max')
        # Humidity (%) -> Normalized (0-1)
        rmin = weather.select('rmin').divide(100.0).rename('Humidity_Min')
        # Wind (m/s) -> Normalized
        vs = weather.select('vs').divide(20.0).clamp(0, 1).rename('Wind_Speed')
        # Precip (mm) -> Normalized
        pr = weather.select('pr').divide(50.0).clamp(0, 1).rename('Precip')
        
        # Unmask weather explicitly just in case
        tmmx = tmmx.unmask(0)
        rmin = rmin.unmask(0)
        vs = vs.unmask(0)
        pr = pr.unmask(0)

        # --- 3. TOPOGRAPHY ---
        topo = ee.Image('USGS/SRTMGL1_003').unmask(0)
        elevation = topo.select('elevation').divide(4000.0).clamp(0, 1).rename('Elevation').float()
        slope = ee.Terrain.slope(topo.select('elevation')).divide(45.0).clamp(0, 1).rename('Slope').float()

        # --- 4. POPULATION ---
        # Note: Dataget.py uses 2020-2021 fixed date.
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
            pop,                        # 1 Band
            weather.select('tmmx').rename('Temp_Raw') # 16th Band: Raw Temp for Debugging
        ]).unmask(0) # GLOBAL UNMASK
        
        return full_stack

    def get_data_from_bounds(self, min_lat, min_lon, max_lat, max_lon, date_str, grid_density):
        """
        Fetches data for a grid within the specified bounding box.
        Batches requests to avoid 'Computed value too large' errors.
        """
        try:
            # 1. Create Grid Points
            # Determine step size based on the longest edge
            lat_diff = max_lat - min_lat
            lon_diff = max_lon - min_lon
            
            # Use grid_density to determine number of points
            # We want roughly grid_density * grid_density points
            
            if lat_diff > lon_diff:
                lat_step = lat_diff / grid_density
                lon_step = lat_step # Keep square pixels roughly
            else:
                lon_step = lon_diff / grid_density
                lat_step = lon_step
                
            points = []
            
            # Generate points
            # Simple iteration
            curr_lat = min_lat + (lat_step / 2)
            while curr_lat < max_lat:
                curr_lon = min_lon + (lon_step / 2)
                while curr_lon < max_lon:
                    points.append([curr_lon, curr_lat])
                    curr_lon += lon_step
                curr_lat += lat_step

            print(f"Generated {len(points)} grid points.")
            
            if len(points) == 0:
                return None, None
                
            # 2. Batch Processing
            # "High" density produces many points. 
            # Updated to 22 per user request.
            BATCH_SIZE = 22
            
            all_patches = []
            all_coords = []
            
            target_date = ee.Date(date_str)
            
            # CRITICAL: Buffer the region by ~5km to ensure 'neighborhoodToArray' (128px ~ 2.5km) has context at edges
            region_geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat]).buffer(5000)
            
            # --- Auto-Detect Valid Date (Smart Fallback) ---
            # GRIDMET (Weather) is the limiting factor for "Today" as it lags by 2-4 days.
            # We prioritize getting *correct* weather for the requested date using GFS if GRIDMET is missing.
            
            weather_img = None
            actual_date_str = date_str
            
            # 1. Try Standard GRIDMET
            weather_check = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
                .filterDate(target_date, target_date.advance(1, 'day')) \
                .limit(1)
                
            if weather_check.size().getInfo() > 0:
                print(f"Using Standard GRIDMET data for {date_str}")
            else:
                print(f"No GRIDMET data for {date_str}. Attempting GFS Real-Time fallback...")
                # 2. Try GFS (Real-Time / Forecast)
                # GFS covers recent days where GRIDMET is missing
                weather_img = self._get_gfs_daily(region_geom, target_date)
                
                if weather_img:
                    print("Successfully loaded GFS Real-Time Weather data.")
                else:
                    print("No GFS data found either. Searching backwards for last available GRIDMET...")
                    # 3. Fallback: Search backwards for GRIDMET
                    past_weather = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
                        .filterDate(target_date.advance(-30, 'day'), target_date) \
                        .sort('system:time_start', False) \
                        .limit(1)
                    
                    if past_weather.size().getInfo() > 0:
                        last_img = past_weather.first()
                        last_millis = last_img.get('system:time_start').getInfo()
                        import datetime
                        new_date = datetime.datetime.fromtimestamp(last_millis / 1000.0)
                        actual_date_str = new_date.strftime('%Y-%m-%d')
                        target_date = ee.Date(actual_date_str)
                        print(f"Fallback: Found Weather data from {actual_date_str}.")
                    else:
                        print("Critical: No weather data found at all.")

            # Check cache first
            if date_str in self.cache:
                print(f"CACHE HIT for date: {date_str}")
                cached_data = self.cache[date_str]
                # We need to filter the cached data for the requested bounds
                all_patches, all_coords, _ = cached_data
                
                filtered_patches = []
                filtered_coords = []
                for i, (lat, lon) in enumerate(all_coords):
                    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                        filtered_patches.append(all_patches[i])
                        filtered_coords.append((lat, lon))
                
                return filtered_patches, filtered_coords, date_str

            print(f"CACHE MISS for date: {date_str}. Fetching from GEE.")
            
            image_stack = self._get_image_stack(region_geom.bounds(), target_date, weather_img)
            
            # Neighborhood kernel
            kernel = ee.Kernel.rectangle(128, 128, 'pixels')
            neighborhood_img = image_stack.neighborhoodToArray(kernel)

            # Define the task function for parallel execution
            def fetch_batch(batch_info):
                index, points_batch = batch_info
                total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE
                # Only log every 10th batch to reduce spam
                if (index + 1) % 10 == 0 or index + 1 == total_batches:
                    print(f"Fetching batch {index + 1}/{total_batches}...")
                
                # Create FC for this batch
                batch_feats = [ee.Feature(ee.Geometry.Point(p), {'id': f"{idx}"}) for idx, p in enumerate(points_batch)]
                batch_fc = ee.FeatureCollection(batch_feats)
                
                try:
                    samples = neighborhood_img.sampleRegions(
                        collection=batch_fc,
                        scale=self.SCALE,
                        projection='EPSG:3857',
                        geometries=True,
                        tileScale=4 # Reduced from 16 for speed (Batch sizes are small now) 
                    )
                    
                    data = samples.getInfo()
                    
                    batch_patches = []
                    batch_coords = []
                    
                    if data and 'features' in data:
                        for feature in data['features']:
                            props = feature['properties']
                            geo = feature['geometry']['coordinates']
                            
                            # OCEAN FILTER: Check if this point is over water
                            # Multi-signal approach for robust ocean detection
                            elevation = props.get('Elevation', [0])
                            if isinstance(elevation, list):
                                avg_elevation = np.mean(elevation) if len(elevation) > 0 else 0
                            else:
                                avg_elevation = elevation
                            
                            ndvi = props.get('NDVI', [0])
                            if isinstance(ndvi, list):
                                avg_ndvi = np.mean(ndvi) if len(ndvi) > 0 else 0
                            else:
                                avg_ndvi = ndvi
                            
                            # Ocean detection criteria:
                            # 1. Elevation at or below sea level (with small tolerance for precision)
                            # 2. Very low or negative NDVI (water has negative NDVI)
                            # 3. Check for all-zero patches (missing data over ocean)
                            is_ocean = (
                                (avg_elevation <= 0.01) and  # Normalized elevation ≈0 means sea level or below
                                (avg_ndvi < 0.15)             # Low NDVI indicates water/no vegetation
                            )
                            
                            if is_ocean:
                                # Skip this point entirely - it's over ocean
                                continue
                            
                            img_stack = np.zeros((257, 257, 16)) # Increased to 16 for debug band
                            
                            valid_patch = True
                            all_bands = self.BANDS + ['Temp_Raw']
                            
                            for idx, band_name in enumerate(all_bands):
                                if band_name not in props:
                                    if band_name == 'Temp_Raw': continue
                                    valid_patch = False
                                    break
                                arr = np.array(props[band_name])
                                h, w = arr.shape
                                img_stack[:h, :w, idx] = arr
                            
                            if valid_patch:
                                # Additional ocean check: If elevation band is all zeros, likely ocean
                                elevation_band = img_stack[:, :, 12]
                                ndvi_band = img_stack[:, :, 6]
                                
                                # If most of the patch has zero elevation and low NDVI, it's water
                                zero_elev_ratio = np.sum(elevation_band <= 0.01) / (256 * 256)
                                low_ndvi_ratio = np.sum(ndvi_band < 0.15) / (256 * 256)
                                
                                if zero_elev_ratio > 0.7 and low_ndvi_ratio > 0.5:
                                    # This patch is mostly ocean, skip it
                                    continue

                                # Crop to 256x256
                                img_stack = img_stack[:256, :256, :]
                                batch_patches.append(img_stack)
                                batch_coords.append((geo[1], geo[0]))
                                
                    return batch_patches, batch_coords

                except Exception as e:
                    print(f"Batch {index+1} failed: {e}")
                    return [], []

            # Reduce batch size for parallel execution
            BATCH_SIZE = 8
            
            # Prepare batches
            batches = []
            for i in range(0, len(points), BATCH_SIZE):
                batches.append((i // BATCH_SIZE, points[i : i + BATCH_SIZE]))

            # Execute in parallel
            # Use 10 threads to match urllib3 connection pool size and avoid warnings
            # This is the optimal balance between speed and resource usage
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(fetch_batch, batches))

            # Combine results
            for patches, coords in results:
                all_patches.extend(patches)
                all_coords.extend(coords)
                
            # Store full result in cache before returning
            print(f"Storing {len(all_patches)} patches in cache for date: {actual_date_str}")
            self.cache[actual_date_str] = (all_patches, all_coords, actual_date_str)
            
            return all_patches, all_coords, actual_date_str        
        except Exception as e:
            print(f"GEE Error: {e}")
            raise e

    def precache_florida_data(self):
        from datetime import datetime
        today_str = datetime.now().strftime('%Y-%m-%d')
        print("--- Starting Florida Pre-caching ---")
        
        # Bounding box for Florida
        min_lat = 24.3963
        max_lat = 31.0010
        min_lon = -87.6349
        max_lon = -79.9743
        
        # A high grid density to cover the state
        grid_density = 150 
        
        try:
            self.get_data_from_bounds(
                min_lat, min_lon, max_lat, max_lon, today_str, grid_density
            )
            print("--- Florida Pre-caching Complete ---")
        except Exception as e:
            print(f"--- Florida Pre-caching FAILED: {e} ---")