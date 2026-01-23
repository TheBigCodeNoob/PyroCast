"""
PyroCast Fire Spread Model - Step 3: Fetch Environmental Data
==============================================================
Fetches all environmental context data from Google Earth Engine:
1. Terrain: Elevation, Slope, Aspect
2. Fuel: LANDFIRE fuel models, NDVI, vegetation type
3. Weather: Temperature, Humidity, Wind (speed & direction), Precipitation
4. Fire Behavior Indices: ERC, BI, SC from GRIDMET

This data forms the environmental context that drives fire spread behavior.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize Earth Engine
import ee
try:
    ee.Initialize(project=GEE_PROJECT)
    logger.info("Google Earth Engine initialized successfully")
except Exception as e:
    logger.error(f"GEE initialization failed: {e}")
    logger.error("Run 'earthengine authenticate' to set up credentials")
    raise

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# Terrain from SRTM
DEM_SOURCE = "USGS/SRTMGL1_003"

# LANDFIRE Fuel Data (2020 version - most comprehensive)
LANDFIRE_FUEL = "LANDFIRE/Fire/FVT/v1_4_0"  # Fuel Vegetation Type
LANDFIRE_FBFM = "LANDFIRE/Fire/FBFM40/v1_4_0"  # 40 Fire Behavior Fuel Models

# Vegetation Indices
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
MODIS_NDVI = "MODIS/061/MOD13A2"  # 1km NDVI (16-day composite)

# Weather from GRIDMET
GRIDMET_COLLECTION = "IDAHO_EPSCOR/GRIDMET"

# Live Fuel Moisture from GRIDMET
GRIDMET_DROUGHT = "GRIDMET/DROUGHT"

# Output paths
ENVIRONMENTAL_DIR = os.path.join(RAW_DATA_DIR, "environmental")
PROGRESSIONS_DIR = os.path.join(RAW_DATA_DIR, "progressions")
os.makedirs(ENVIRONMENTAL_DIR, exist_ok=True)


class EnvironmentalDataFetcher:
    """
    Fetches and processes environmental data for fire spread modeling.
    All data is aligned to a common grid and normalized for model input.
    """
    
    def __init__(self):
        self.scale = SPATIAL_RESOLUTION
        self.img_size = IMG_SIZE
        self.terrain_cache = {}  # Cache terrain data (doesn't change with date)
        
    def _get_terrain(self, region, cache_key=None):
        """
        Extract terrain features: elevation, slope, and aspect.
        Simplified for lightweight model - aspect in radians (0-2π).
        Uses caching since terrain is static.
        """
        # Check cache first
        if cache_key and cache_key in self.terrain_cache:
            return self.terrain_cache[cache_key]
        
        dem = ee.Image(DEM_SOURCE).clip(region)
        
        # Elevation (normalized to 0-1 for 0-4000m range)
        elevation = dem.select('elevation').divide(4000).clamp(0, 1).rename('elevation')
        
        # Slope (normalized to 0-1 for 0-60 degree range)
        # Extended range for steep mountainous terrain
        slope = ee.Terrain.slope(dem).divide(60).clamp(0, 1).rename('slope')
        
        # Aspect in radians (0-2π normalized to 0-1)
        # Represents sun exposure and wind interaction
        aspect_deg = ee.Terrain.aspect(dem)
        aspect = aspect_deg.multiply(np.pi / 180).divide(2 * np.pi).rename('aspect')
        
        terrain_img = ee.Image.cat([elevation, slope, aspect])
        
        # Cache if key provided
        if cache_key:
            self.terrain_cache[cache_key] = terrain_img
        
        return terrain_img
    
    def _get_fuel_data(self, region, date):
        """
        Extract fuel-related data (simplified):
        - fuel_type: NDVI-based vegetation proxy (LANDFIRE not globally available)
        - fuel_moisture: From GRIDMET humidity (validated to exist by _validate_data_availability)
        - fuel_density: NDVI as proxy for fuel load
        """
        date_ee = ee.Date(date.strftime('%Y-%m-%d'))
        
        # --- Fuel Density & Type from NDVI ---
        # Use 30-day lookback for Sentinel-2
        s2_start = date_ee.advance(-30, 'day')
        
        s2_col = ee.ImageCollection(SENTINEL2_COLLECTION) \
            .filterBounds(region) \
            .filterDate(s2_start, date_ee) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)) \
            .limit(5)  # Only need a few images for median
        
        # Fallback to MODIS if no Sentinel-2 available
        ndvi = ee.Algorithms.If(
            s2_col.size().gt(0),
            s2_col.median().normalizedDifference(['B8', 'B4']).rename('ndvi'),
            self._get_modis_ndvi(region, date)
        )
        
        ndvi_img = ee.Image(ndvi)
        
        # Fuel density: NDVI scaled to 0-1
        fuel_density = ndvi_img.clamp(-1, 1).add(1).divide(2).rename('fuel_density')
        
        # Fuel type: NDVI-based proxy for vegetation category
        # Low NDVI (~0-0.3) = sparse/grass, Mid (0.3-0.6) = shrub/brush, High (0.6+) = forest
        fuel_type = ndvi_img.clamp(-1, 1).add(1).divide(2).rename('fuel_type')
        
        # --- Fuel Moisture from GRIDMET ---
        # Use humidity as proxy (already validated to exist)
        gridmet = ee.ImageCollection(GRIDMET_COLLECTION) \
            .filterBounds(region) \
            .filterDate(date_ee.advance(-3, 'day'), date_ee.advance(1, 'day')) \
            .sort('system:time_start', False)
        
        weather = gridmet.first()
        
        # Use relative humidity (rmin, rmax) to estimate fuel moisture
        # Higher humidity = higher fuel moisture
        rmin = weather.select('rmin')
        rmax = weather.select('rmax')
        fuel_moisture = rmin.add(rmax).divide(2).divide(100).clamp(0, 1).rename('fuel_moisture')
        
        return ee.Image.cat([fuel_type, fuel_moisture, fuel_density])
    
    def _get_modis_ndvi(self, region, date):
        """Fallback NDVI from MODIS with proper empty collection handling."""
        date_ee = ee.Date(date.strftime('%Y-%m-%d'))
        
        modis = ee.ImageCollection(MODIS_NDVI) \
            .filterBounds(region) \
            .filterDate(date_ee.advance(-30, 'day'), date_ee)
        
        # Handle empty collection properly
        return ee.Algorithms.If(
            modis.size().gt(0),
            modis.first().select('NDVI').divide(10000),
            ee.Image.constant(0.3).rename('ndvi')
        )
    
    def _validate_data_availability(self, region, date):
        """
        Validate that required data sources have coverage for this region and date.
        Returns True only if ALL required data sources are available.
        This prevents processing fires in regions without proper data coverage.
        """
        date_ee = ee.Date(date.strftime('%Y-%m-%d'))
        
        try:
            # Check GRIDMET coverage (required for weather and fire indices)
            gridmet = ee.ImageCollection(GRIDMET_COLLECTION) \
                .filterBounds(region) \
                .filterDate(date_ee.advance(-3, 'day'), date_ee.advance(1, 'day'))
            
            # Must have at least one GRIDMET image in 3-day window
            if gridmet.size().getInfo() == 0:
                return False
            
            # Verify the image has required bands
            first_img = gridmet.first()
            band_names = first_img.bandNames().getInfo()
            required_bands = ['vs', 'th', 'tmmx', 'rmin', 'erc', 'bi']
            
            if not all(band in band_names for band in required_bands):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Data availability check failed: {e}")
            return False
    
    def _get_weather(self, region, date):
        """
        Extract weather data (simplified):
        - wind_u, wind_v: Wind components (east-west, north-south)
        - temperature: Max daily temperature
        - humidity: Min daily humidity
        """
        date_ee = ee.Date(date.strftime('%Y-%m-%d'))
        
        # Get GRIDMET data with 3-day lookback window
        gridmet = ee.ImageCollection(GRIDMET_COLLECTION) \
            .filterBounds(region) \
            .filterDate(date_ee.advance(-3, 'day'), date_ee.advance(1, 'day')) \
            .sort('system:time_start', False)
        
        weather = gridmet.first()
        
        # Wind speed (m/s) and direction (degrees) -> u,v components
        vs = weather.select('vs')  # Wind speed
        th = weather.select('th').multiply(np.pi / 180)  # Direction in radians
        
        # Convert to u,v components (normalized to 0-1 for -20 to +20 m/s range)
        wind_u = vs.multiply(th.sin()).add(20).divide(40).clamp(0, 1).rename('wind_u')
        wind_v = vs.multiply(th.cos()).add(20).divide(40).clamp(0, 1).rename('wind_v')
        
        # Temperature (Kelvin -> normalized)
        # Range: ~250K to ~320K -> 0-1
        temperature = weather.select('tmmx').subtract(250).divide(70).clamp(0, 1).rename('temperature')
        
        # Humidity (% -> normalized)
        humidity = weather.select('rmin').divide(100).clamp(0, 1).rename('humidity')
        
        return ee.Image.cat([wind_u, wind_v, temperature, humidity])
    
    def _get_fire_indices(self, region, date):
        """
        Extract combined fire behavior indices (simplified):
        - fwi: Fire Weather Index (combined metric from ERC, BI)
        - spread_potential: Expected spread rate from environmental conditions
        """
        date_ee = ee.Date(date.strftime('%Y-%m-%d'))
        
        # Get GRIDMET data with 3-day lookback
        gridmet = ee.ImageCollection(GRIDMET_COLLECTION) \
            .filterBounds(region) \
            .filterDate(date_ee.advance(-3, 'day'), date_ee.advance(1, 'day')) \
            .sort('system:time_start', False)
        
        weather = gridmet.first()
        
        # --- Fire Weather Index (FWI) ---
        # Combine ERC (energy available) and BI (burning intensity)
        # ERC: 0-100+ range, BI: 0-200+ range
        erc = weather.select('erc').divide(100).clamp(0, 1)
        bi = weather.select('bi').divide(200).clamp(0, 1)
        
        # FWI = weighted combination emphasizing energy release
        fwi = erc.multiply(0.6).add(bi.multiply(0.4)).rename('fwi')
        
        # --- Spread Potential ---
        # Proxy for expected fire spread rate based on wind and fire danger
        # Combines wind speed with fire weather conditions
        wind_speed = weather.select('vs').divide(20).clamp(0, 1)  # Normalized wind
        
        # Spread potential = wind effect + fire danger effect
        spread_potential = wind_speed.multiply(0.5).add(fwi.multiply(0.5)).rename('spread_potential')
        
        return ee.Image.cat([fwi, spread_potential])
    
    def get_full_environmental_stack(self, region, date, cache_key=None):
        """
        Build the complete environmental feature stack for a given region and date.
        
        Returns 12 channels (excluding fire mask which is added during dataset building):
        - Terrain: elevation, slope, aspect (3)
        - Fuel: fuel_type, fuel_moisture, fuel_density (3)
        - Weather: wind_u, wind_v, temperature, humidity (4)
        - Fire indices: fwi, spread_potential (2)
        
        Total: 12 environmental channels + 1 fire mask = 13 input channels
        """
        terrain = self._get_terrain(region, cache_key=cache_key)
        fuel = self._get_fuel_data(region, date)
        weather = self._get_weather(region, date)
        fire_indices = self._get_fire_indices(region, date)
        
        # Stack all bands
        full_stack = ee.Image.cat([terrain, fuel, weather, fire_indices])
        
        # Unmask all bands with 0
        full_stack = full_stack.unmask(0)
        
        return full_stack
    
    def extract_patch(self, region, date, patch_id, cache_key=None):
        """
        Extract a fixed-size patch of environmental data.
        
        Args:
            region: ee.Geometry (bounding box)
            date: datetime object
            patch_id: unique identifier for this patch
            cache_key: optional key for terrain caching
        
        Returns:
            numpy array of shape (IMG_SIZE, IMG_SIZE, 17) or None if failed
        """
        try:
            # Validate data availability first
            if not self._validate_data_availability(region, date):
                logger.debug(f"Skipping {patch_id}: Insufficient data coverage")
                return None
            
            # Get the environmental stack (uses cached terrain if available)
            env_stack = self.get_full_environmental_stack(region, date, cache_key=cache_key)
            
            # Sample as array
            kernel = ee.Kernel.rectangle(self.img_size // 2, self.img_size // 2, 'pixels')
            
            # Get center point
            center = region.centroid()
            
            # Sample neighborhood
            patch = env_stack.neighborhoodToArray(kernel).sampleRegions(
                collection=ee.FeatureCollection([ee.Feature(center)]),
                scale=self.scale,
                projection='EPSG:4326',
                tileScale=4
            )
            
            # Get the data
            patch_info = patch.first().getInfo()
            
            if patch_info is None:
                return None
            
            props = patch_info['properties']
            
            # Reconstruct the array
            band_names = [
                'elevation', 'slope', 'aspect',
                'fuel_type', 'fuel_moisture', 'fuel_density',
                'wind_u', 'wind_v', 'temperature', 'humidity',
                'fwi', 'spread_potential'
            ]
            
            img_array = np.zeros((self.img_size, self.img_size, len(band_names)), dtype=np.float32)
            
            for i, band in enumerate(band_names):
                if band in props:
                    arr = np.array(props[band], dtype=np.float32)
                    # Reshape to 2D
                    side = int(np.sqrt(len(arr)))
                    if side * side == len(arr):
                        arr = arr.reshape((side, side))
                        # Crop/pad to target size
                        h, w = arr.shape
                        img_array[:min(h, self.img_size), :min(w, self.img_size), i] = \
                            arr[:min(h, self.img_size), :min(w, self.img_size)]
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error extracting patch {patch_id}: {e}")
            return None
    
    def process_fire_event(self, fire_id, idx, total):
        """
        Process all environmental data for a single fire event.
        Extracts data for each day in the fire's progression.
        """
        # Load progression data
        progression_path = os.path.join(PROGRESSIONS_DIR, f"progression_{fire_id}.json")
        
        if not os.path.exists(progression_path):
            logger.warning(f"[{idx+1}/{total}] No progression data for fire {fire_id}")
            return None
        
        with open(progression_path, 'r') as f:
            progression = json.load(f)
        
        fire_name = progression.get('fire_name', fire_id)
        logger.info(f"[{idx+1}/{total}] Processing environmental data for: {fire_name}")
        
        # Get bounding box
        bbox = progression['bbox']
        region = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        # Get daily environmental data
        daily_masks = progression.get('daily_masks', {})
        
        if len(daily_masks) < MIN_PROGRESSION_DAYS:
            logger.warning(f"  Insufficient progression days: {len(daily_masks)}")
            return None
        
        # CRITICAL: Pre-validate data coverage for this fire
        # Check first day to see if region has GRIDMET coverage
        first_date_str = sorted(daily_masks.keys())[0]
        first_date = datetime.strptime(first_date_str, '%Y-%m-%d')
        
        if not self._validate_data_availability(region, first_date):
            logger.warning(f"  Skipping fire - no GRIDMET coverage for region (lat: {bbox[1]:.2f}, lon: {bbox[0]:.2f})")
            return None
        
        # Extract environmental data for each day
        env_data = {}
        cache_key = f"terrain_{fire_id}"  # Cache key for this fire's terrain
        
        for date_str in sorted(daily_masks.keys())[:MAX_PROGRESSION_DAYS]:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            logger.debug(f"  Extracting environmental data for {date_str}")
            
            try:
                patch = self.extract_patch(region, date, f"{fire_id}_{date_str}", cache_key=cache_key)
                
                if patch is not None:
                    env_data[date_str] = {
                        'data': patch.tolist(),  # Convert to list for JSON
                        'shape': patch.shape
                    }
            except Exception as e:
                logger.warning(f"  Failed to extract data for {date_str}: {e}")
        
        if len(env_data) < MIN_PROGRESSION_DAYS:
            logger.warning(f"  Insufficient environmental data days: {len(env_data)}")
            return None
        
        # Save environmental data
        output = {
            'fire_id': fire_id,
            'fire_name': fire_name,
            'bbox': bbox,
            'days_extracted': len(env_data),
            'dates': list(env_data.keys()),
            'env_data': env_data
        }
        
        output_path = os.path.join(ENVIRONMENTAL_DIR, f"env_{fire_id}.json")
        
        # Note: This creates large files. In production, use NumPy's .npz format instead
        with open(output_path, 'w') as f:
            json.dump(output, f)
        
        logger.info(f"  Saved {len(env_data)} days of environmental data")
        
        return output
    
    def run(self):
        """Execute the environmental data fetching pipeline with parallel processing."""
        logger.info("=" * 60)
        logger.info("PyroCast Fire Spread Model - Environmental Data Fetcher")
        logger.info("=" * 60)
        
        # Get list of fires with progression data
        progression_files = [f for f in os.listdir(PROGRESSIONS_DIR) 
                           if f.startswith('progression_') and f.endswith('.json')]
        
        if not progression_files:
            logger.error("No progression files found. Run 02_fetch_daily_progressions.py first.")
            return
        
        # Extract fire IDs
        fire_ids = [f.replace('progression_', '').replace('.json', '') for f in progression_files]
        
        logger.info(f"Found {len(fire_ids)} fires with progression data")
        logger.info(f"Using parallel processing with 10 workers")
        
        # Process fires in parallel
        successful = []
        failed = []
        completed = 0
        start_time = time.time()
        
        def process_wrapper(args):
            nonlocal completed
            idx, fire_id = args
            try:
                result = self.process_fire_event(fire_id, idx, len(fire_ids))
                completed += 1
                
                # Progress update every 50 fires
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed * 60  # fires per minute
                    eta = (len(fire_ids) - completed) / (completed / elapsed) / 60  # hours
                    logger.info(f"Progress: {completed}/{len(fire_ids)} fires ({completed/len(fire_ids)*100:.1f}%) | "
                              f"Rate: {rate:.1f} fires/min | ETA: {eta:.1f}h")
                
                return (fire_id, result is not None)
            except Exception as e:
                logger.error(f"Error processing fire {fire_id}: {e}")
                completed += 1
                return (fire_id, False)
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_wrapper, enumerate(fire_ids)))
        
        # Collect results
        for fire_id, success in results:
            if success:
                successful.append(fire_id)
            else:
                failed.append(fire_id)
        
        # Save summary
        summary = {
            'total_processed': len(fire_ids),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(fire_ids) * 100 if fire_ids else 0,
            'successful_ids': successful,
            'failed_ids': failed
        }
        
        summary_path = os.path.join(ENVIRONMENTAL_DIR, "environmental_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=" * 60)
        logger.info(f"Pipeline complete!")
        logger.info(f"Successful: {len(successful)}/{len(fire_ids)}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Output directory: {ENVIRONMENTAL_DIR}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    fetcher = EnvironmentalDataFetcher()
    fetcher.run()


if __name__ == "__main__":
    main()
