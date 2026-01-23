"""
PyroCast Fire Spread Model - Step 2: Fetch Daily Fire Progressions
===================================================================
Uses VIIRS (Visible Infrared Imaging Radiometer Suite) and MODIS active fire 
detections to reconstruct daily fire progression for each historical fire event.

Key Data Sources:
1. VIIRS 375m Active Fire Product (most detailed)
2. MODIS 1km Active Fire Product (longer history, backup)
3. NASA FIRMS (Fire Information for Resource Management System)

Output: Daily fire masks for each fire event showing progression
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import unary_union
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# NASA FIRMS API CONFIGURATION
# =============================================================================

# NASA FIRMS API Key - Get yours at https://firms.modaps.eosdis.nasa.gov/api/
# This is a public key for demo purposes - replace with your own for production
FIRMS_API_KEY = "DEMO_KEY"  # Replace with your actual API key

# FIRMS API endpoints
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Data sources available through FIRMS
FIRMS_SOURCES = {
    'VIIRS_SNPP': 'VIIRS_SNPP_NRT',  # Suomi NPP VIIRS (2012-present)
    'VIIRS_NOAA20': 'VIIRS_NOAA20_NRT',  # NOAA-20 VIIRS (2018-present)
    'MODIS': 'MODIS_NRT',  # MODIS (2000-present)
}

# Paths
PROGRESSIONS_DIR = os.path.join(RAW_DATA_DIR, "progressions")
FIRE_CATALOG_PATH = os.path.join(RAW_DATA_DIR, "fire_catalog.csv")
PROGRESS_CHECKPOINT = os.path.join(PROGRESSIONS_DIR, "_progress_checkpoint.json")
os.makedirs(PROGRESSIONS_DIR, exist_ok=True)


def is_fire_processed(fire_id):
    """Check if progression file already exists for this fire."""
    output_path = os.path.join(PROGRESSIONS_DIR, f"progression_{fire_id}.json")
    return os.path.exists(output_path)


class DailyProgressionFetcher:
    """
    Fetches and processes daily fire progression data from VIIRS/MODIS.
    Creates daily fire masks that show how fires spread over time.
    """
    
    def __init__(self, api_key=FIRMS_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PyroCast-FireSpreadModel/1.0'
        })
        
    def load_fire_catalog(self, skip_processed=True):
        """Load the fire catalog created by step 1."""
        if not os.path.exists(FIRE_CATALOG_PATH):
            raise FileNotFoundError(
                f"Fire catalog not found at {FIRE_CATALOG_PATH}. "
                "Run 01_fetch_fire_perimeters.py first."
            )
        
        df = pd.read_csv(FIRE_CATALOG_PATH)
        logger.info(f"Loaded {len(df)} fire perimeters from catalog")
        
        # NOTE: NOT deduplicating by fire_id!
        # Multiple perimeters of same fire = different temporal snapshots = more training data
        # Each perimeter represents a different growth stage
        logger.info(f"Keeping all {len(df)} perimeter snapshots (including temporal duplicates)")
        
        if skip_processed:
            # Skip fires that already have progression files
            processed_mask = df['fire_id'].apply(is_fire_processed)
            already_done = processed_mask.sum()
            df = df[~processed_mask]
            logger.info(f"Skipping {already_done} already-processed fires")
            logger.info(f"Remaining to process: {len(df)} fires")
        
        return df
    
    def fetch_firms_data(self, bbox, start_date, end_date, source='VIIRS_SNPP'):
        """
        Fetch active fire detections from NASA FIRMS API.
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_date: datetime object
            end_date: datetime object
            source: 'VIIRS_SNPP', 'VIIRS_NOAA20', or 'MODIS'
        
        Returns:
            DataFrame with fire detections
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # FIRMS API uses area string format: minLon,minLat,maxLon,maxLat
        area_str = f"{min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}"
        
        # Calculate day range
        day_range = (end_date - start_date).days + 1
        
        # FIRMS API limit is 10 days per request for free tier
        MAX_DAYS_PER_REQUEST = 10
        
        all_detections = []
        current_start = start_date
        
        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=MAX_DAYS_PER_REQUEST-1), end_date)
            
            # Format dates for API
            date_range = f"{(current_end - current_start).days + 1}"
            
            url = f"{FIRMS_BASE_URL}/{self.api_key}/{FIRMS_SOURCES.get(source, source)}/{area_str}/{date_range}/{current_end.strftime('%Y-%m-%d')}"
            
            try:
                response = self.session.get(url, timeout=60)
                
                if response.status_code == 200 and response.text.strip():
                    # Parse CSV response
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    
                    if len(df) > 0:
                        all_detections.append(df)
                        logger.debug(f"  Got {len(df)} detections for {current_start.date()} to {current_end.date()}")
                        
                elif response.status_code == 401:
                    logger.warning("FIRMS API key invalid. Using GEE fallback.")
                    return None
                    
            except Exception as e:
                logger.warning(f"FIRMS request failed: {e}")
            
            current_start = current_end + timedelta(days=1)
            time.sleep(0.5)  # Rate limiting
        
        if all_detections:
            return pd.concat(all_detections, ignore_index=True)
        return None
    
    def fetch_viirs_from_gee(self, bbox, start_date, end_date):
        """
        Alternative: Fetch VIIRS active fire data from Google Earth Engine.
        More reliable than FIRMS API for historical data.
        """
        try:
            import ee
            ee.Initialize(project=GEE_PROJECT)
        except Exception as e:
            logger.error(f"GEE initialization failed: {e}")
            return None
        
        min_lon, min_lat, max_lon, max_lat = bbox
        region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        
        # VIIRS Active Fire Product in GEE (updated to non-deprecated version)
        viirs = ee.ImageCollection('NASA/VIIRS/002/VNP14A1') \
            .filterBounds(region) \
            .filterDate(start_date.strftime('%Y-%m-%d'), 
                       (end_date + timedelta(days=1)).strftime('%Y-%m-%d'))
        
        # Also get MODIS as backup
        modis = ee.ImageCollection('MODIS/061/MOD14A1') \
            .filterBounds(region) \
            .filterDate(start_date.strftime('%Y-%m-%d'),
                       (end_date + timedelta(days=1)).strftime('%Y-%m-%d'))
        
        detections = []
        
        # Process VIIRS (limit to first 21 days max for speed)
        viirs_list = viirs.toList(viirs.size())
        try:
            viirs_count = min(viirs.size().getInfo(), 21)  # Cap at 21 iterations
        except:
            viirs_count = 0
        
        for i in range(viirs_count):
            try:
                img = ee.Image(viirs_list.get(i))
                date = datetime.fromtimestamp(img.date().millis().getInfo() / 1000)
                
                # Get fire mask (MaxFRP > 0 indicates fire)
                fire_mask = img.select('MaxFRP').gt(0)
                
                # Sample fire pixels (use 1.5km scale for maximum speed)
                fire_points = fire_mask.selfMask().reduceToVectors(
                    geometry=region,
                    scale=1500,  # Use 1.5km for maximum speed
                    maxPixels=5e7,  # Reduced from 1e8
                    geometryType='centroid',
                    eightConnected=False,
                    bestEffort=True  # Allow incomplete results if timeout
                )
                
                # Add timeout to prevent hanging
                try:
                    points_info = fire_points.getInfo()
                except Exception as timeout_err:
                    # Early exit if multiple consecutive timeouts
                    if i > 5 and len(detections) == 0:
                        break
                    continue
                
                if points_info and 'features' in points_info:
                    for feat in points_info['features']:
                        coords = feat['geometry']['coordinates']
                        frp = feat['properties'].get('MaxFRP', 0)
                        
                        detections.append({
                            'latitude': coords[1],
                            'longitude': coords[0],
                            'acq_date': date.strftime('%Y-%m-%d'),
                            'acq_time': '1200',  # Approximate
                            'confidence': 'high' if frp > 10 else 'nominal',
                            'frp': frp,
                            'source': 'VIIRS_GEE'
                        })
                        
            except Exception as e:
                logger.debug(f"Error processing VIIRS image: {e}")
                continue
        
        # If VIIRS didn't find enough detections, try MODIS as fallback
        if len(detections) < 5:
            logger.debug(f"Only {len(detections)} VIIRS detections, trying MODIS fallback")
            
            modis_list = modis.toList(modis.size())
            try:
                modis_count = min(modis.size().getInfo(), 21)  # Cap iterations
            except:
                modis_count = 0
            
            for i in range(modis_count):
                try:
                    img = ee.Image(modis_list.get(i))
                    date = datetime.fromtimestamp(img.date().millis().getInfo() / 1000)
                    
                    # Get fire mask (FireMask > 7 indicates fire)
                    fire_mask = img.select('FireMask').gte(7)
                    
                    # Sample fire pixels
                    fire_points = fire_mask.selfMask().reduceToVectors(
                        geometry=region,
                        scale=1500,  # Use faster scale
                        maxPixels=5e7,
                        geometryType='centroid',
                        eightConnected=False,
                        bestEffort=True
                    )
                    
                    try:
                        points_info = fire_points.getInfo()
                    except Exception:
                        continue
                    
                    if points_info and 'features' in points_info:
                        for feat in points_info['features']:
                            coords = feat['geometry']['coordinates']
                            
                            detections.append({
                                'latitude': coords[1],
                                'longitude': coords[0],
                                'acq_date': date.strftime('%Y-%m-%d'),
                                'acq_time': '1300',
                                'confidence': 'nominal',
                                'frp': 10,  # Default FRP for MODIS
                                'source': 'MODIS_GEE'
                            })
                            
                except Exception as e:
                    logger.debug(f"Error processing MODIS image: {e}")
                    continue
        
        if detections:
            return pd.DataFrame(detections)
        return None
    
    def create_daily_masks(self, fire_info, detections_df):
        """
        Create daily fire masks from point detections.
        
        Uses kernel density estimation and buffering to create
        continuous fire extent estimates from sparse point data.
        """
        if detections_df is None or len(detections_df) == 0:
            return None
        
        # Parse dates
        if 'acq_date' in detections_df.columns:
            detections_df['date'] = pd.to_datetime(detections_df['acq_date'])
        elif 'ACQ_DATE' in detections_df.columns:
            detections_df['date'] = pd.to_datetime(detections_df['ACQ_DATE'])
        else:
            logger.warning("No date column found in detections")
            return None
        
        # Group by date
        daily_groups = detections_df.groupby(detections_df['date'].dt.date)
        
        daily_masks = {}
        cumulative_points = []
        
        for date, group in sorted(daily_groups):
            # Extract coordinates
            if 'latitude' in group.columns:
                lats = group['latitude'].values
                lons = group['longitude'].values
            else:
                lats = group['LATITUDE'].values
                lons = group['LONGITUDE'].values
            
            # Filter by confidence if available
            if 'confidence' in group.columns:
                conf_mask = group['confidence'].isin(['high', 'nominal', 'h', 'n'])
                lats = lats[conf_mask]
                lons = lons[conf_mask]
            
            # Create points for this day
            day_points = [Point(lon, lat) for lat, lon in zip(lats, lons)]
            
            if not day_points:
                continue
            
            # Add to cumulative (fire doesn't un-burn)
            cumulative_points.extend(day_points)
            
            # Create buffered union of all points (375m for VIIRS resolution)
            # Buffer in degrees (approximate: 375m ≈ 0.0034 degrees at mid-latitudes)
            buffer_deg = 375 / 111000
            
            buffered_points = [p.buffer(buffer_deg) for p in cumulative_points]
            daily_extent = unary_union(buffered_points)
            
            daily_masks[str(date)] = {
                'geometry': mapping(daily_extent),
                'new_points': len(day_points),
                'total_points': len(cumulative_points),
                'area_km2': daily_extent.area * 111 * 111  # Approximate
            }
        
        return daily_masks
    
    def process_fire(self, fire_row, idx, total):
        """Process a single fire event to extract daily progression."""
        fire_id = fire_row['fire_id']
        fire_name = fire_row['fire_name']
        acres = fire_row.get('acres', 0)
        
        logger.info(f"[{idx+1}/{total}] Processing: {fire_name} ({fire_id})")
        
        # Get bounding box with buffer
        bbox = (
            fire_row['bbox_min_lon'] - 0.1,  # ~10km buffer
            fire_row['bbox_min_lat'] - 0.1,
            fire_row['bbox_max_lon'] + 0.1,
            fire_row['bbox_max_lat'] + 0.1
        )
        
        # Determine date range
        if pd.notna(fire_row.get('start_date')):
            start_date = pd.to_datetime(fire_row['start_date'])
        else:
            # Estimate from year - use fire season (May-October)
            start_date = datetime(int(fire_row['year']), 5, 1)
        
        # Fire duration estimate based on size (larger fires burn longer)
        estimated_duration = min(max(int(np.sqrt(acres) / 12), 7), 21)  # 7-21 days
        end_date = start_date + timedelta(days=estimated_duration)
        
        # Skip debug logging for speed
        # logger.debug(f"  Date range: {start_date.date()} to {end_date.date()}")
        
        # Skip FIRMS API (DEMO_KEY has rate limits), go straight to GEE (faster and more reliable)
        detections = self.fetch_viirs_from_gee(bbox, start_date, end_date)
        
        if detections is None or len(detections) < 5:
            return None
        
        # Create daily masks
        daily_masks = self.create_daily_masks(fire_row, detections)
        
        if daily_masks is None or len(daily_masks) < MIN_PROGRESSION_DAYS:
            return None
        
        # Save progression data
        progression_data = {
            'fire_id': str(fire_id),
            'fire_name': fire_name,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'bbox': bbox,
            'total_detections': len(detections),
            'progression_days': len(daily_masks),
            'daily_masks': daily_masks
        }
        
        output_path = os.path.join(PROGRESSIONS_DIR, f"progression_{fire_id}.json")
        with open(output_path, 'w') as f:
            json.dump(progression_data, f, indent=2)
        
        return progression_data
    
    def run(self, max_workers=1, use_parallel=False):
        """Execute the daily progression fetching pipeline."""
        logger.info("=" * 60)
        logger.info("PyroCast Fire Spread Model - Daily Progression Fetcher")
        logger.info("=" * 60)
        
        # Load fire catalog (skip already-processed)
        catalog = self.load_fire_catalog(skip_processed=True)
        
        # Filter to fires with valid bounding boxes and good satellite coverage
        valid_fires = catalog[
            (catalog['bbox_min_lon'].notna()) & 
            (catalog['bbox_max_lon'].notna()) &
            (catalog['year'] >= 2012)  # VIIRS started in 2012, MODIS fallback for earlier
        ]
        
        if len(valid_fires) == 0:
            logger.info("All fires already processed!")
            return []
        
        logger.info(f"Processing {len(valid_fires)} fires with valid bounding boxes")
        
        # Process fires
        successful = []
        failed = []
        skipped = []
        
        if use_parallel and max_workers > 1:
            # Parallel processing (careful with GEE rate limits)
            logger.info(f"Using parallel processing with {max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for idx, (_, row) in enumerate(valid_fires.iterrows()):
                    future = executor.submit(self.process_fire, row, idx, len(valid_fires))
                    futures[future] = row['fire_id']
                
                for future in concurrent.futures.as_completed(futures):
                    fire_id = futures[future]
                    try:
                        result = future.result()
                        if result:
                            if result.get('status') == 'skipped':
                                skipped.append(fire_id)
                            else:
                                successful.append(result)
                        else:
                            failed.append(fire_id)
                    except Exception as e:
                        logger.error(f"Error processing fire {fire_id}: {e}")
                        failed.append(fire_id)
        else:
            # Sequential processing (more reliable for GEE)
            logger.info("Using sequential processing (more reliable)")
            for idx, (_, row) in enumerate(valid_fires.iterrows()):
                try:
                    result = self.process_fire(row, idx, len(valid_fires))
                    if result:
                        if result.get('status') == 'skipped':
                            skipped.append(row['fire_id'])
                        else:
                            successful.append(result)
                    else:
                        failed.append(row['fire_id'])
                except Exception as e:
                    logger.error(f"Error processing fire {row['fire_id']}: {e}")
                    failed.append(row['fire_id'])
                
                # Save checkpoint every 100 fires (less I/O overhead)
                if (idx + 1) % 100 == 0:
                    self.save_checkpoint(successful, failed, skipped, idx + 1, len(valid_fires))
                
                # Rate limiting (reduced for faster processing)
                time.sleep(0.3)
        
        # Save final summary
        self.save_checkpoint(successful, failed, skipped, len(valid_fires), len(valid_fires), is_final=True)
        
        logger.info("=" * 60)
        logger.info(f"Pipeline complete!")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Skipped: {len(skipped)}")
        if len(valid_fires) > 0:
            logger.info(f"Success rate: {len(successful) / len(valid_fires) * 100:.1f}%")
        logger.info(f"Output directory: {PROGRESSIONS_DIR}")
        logger.info("=" * 60)
        
        return successful
    
    def save_checkpoint(self, successful, failed, skipped, current, total, is_final=False):
        """Save progress checkpoint."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'progress': f"{current}/{total}",
            'successful_count': len(successful),
            'failed_count': len(failed),
            'skipped_count': len(skipped),
            'failed_ids': failed,
            'is_final': is_final
        }
        
        with open(PROGRESS_CHECKPOINT, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        if not is_final:
            logger.info(f"Checkpoint saved: {current}/{total} processed")


class GEEProgressionFetcher:
    """
    Alternative fetcher that uses Google Earth Engine directly.
    More reliable for historical data but slower due to API limits.
    """
    
    def __init__(self):
        import ee
        try:
            ee.Initialize(project=GEE_PROJECT)
            logger.info("GEE initialized successfully")
        except Exception as e:
            logger.error(f"GEE initialization failed: {e}")
            raise
    
    def fetch_modis_viirs_composite(self, region, date):
        """
        Fetch combined MODIS/VIIRS fire data for a single day.
        Returns a binary fire mask image.
        """
        import ee
        
        date_str = date.strftime('%Y-%m-%d')
        next_date_str = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # MODIS Thermal Anomalies
        modis = ee.ImageCollection('MODIS/061/MOD14A1') \
            .filterBounds(region) \
            .filterDate(date_str, next_date_str) \
            .select('MaxFRP')
        
        # VIIRS Active Fire
        viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP14A1') \
            .filterBounds(region) \
            .filterDate(date_str, next_date_str) \
            .select('MaxFRP')
        
        # Combine: Take max FRP from both sources
        combined = modis.merge(viirs)
        
        if combined.size().getInfo() == 0:
            return None
        
        # Create fire mask (FRP > 0 means active fire)
        fire_mask = combined.max().gt(0).rename('fire_mask')
        
        return fire_mask
    
    def create_progression_sequence(self, fire_id, bbox, start_date, num_days=14):
        """
        Create a sequence of daily fire masks for a fire event.
        """
        import ee
        
        region = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        progression = []
        cumulative_mask = None
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            
            daily_mask = self.fetch_modis_viirs_composite(region, current_date)
            
            if daily_mask is not None:
                # Update cumulative mask (fire extent only grows)
                if cumulative_mask is None:
                    cumulative_mask = daily_mask
                else:
                    cumulative_mask = cumulative_mask.Or(daily_mask)
                
                progression.append({
                    'day': day,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'mask': cumulative_mask
                })
            
            time.sleep(0.1)  # GEE rate limiting (reduced for speed)
        
        return progression


def main():
    """Main entry point."""
    
    # Initialize Google Earth Engine
    import ee
    ee.Initialize(project=GEE_PROJECT)
    
    # Check for API key
    if FIRMS_API_KEY == "DEMO_KEY":
        logger.warning("=" * 60)
        logger.warning("USING DEMO API KEY - Limited functionality!")
        logger.warning("Get your free API key at:")
        logger.warning("https://firms.modaps.eosdis.nasa.gov/api/")
        logger.warning("Then update FIRMS_API_KEY in this script or config.py")
        logger.warning("=" * 60)
        logger.warning("Falling back to GEE-based fetching...")
        time.sleep(3)
    
    fetcher = DailyProgressionFetcher()
    
    # Use 10 parallel workers
    results = fetcher.run(use_parallel=True, max_workers=10)
    
    # Print summary statistics
    if results:
        print("\n" + "=" * 60)
        print("PROGRESSION DATA SUMMARY")
        print("=" * 60)
        
        total_days = sum(r['progression_days'] for r in results)
        avg_days = total_days / len(results)
        
        print(f"\nFires with valid progression: {len(results)}")
        print(f"Total progression days: {total_days}")
        print(f"Average days per fire: {avg_days:.1f}")
        print(f"\nOutput directory: {PROGRESSIONS_DIR}")


if __name__ == "__main__":
    main()
