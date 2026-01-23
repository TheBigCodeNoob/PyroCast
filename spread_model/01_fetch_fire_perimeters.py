"""
PyroCast Fire Spread Model - Step 1: Fetch Fire Perimeters
===========================================================
Downloads historical wildfire perimeter data from multiple sources:
1. NIFC (National Interagency Fire Center) - Perimeter Archive
2. MTBS (Monitoring Trends in Burn Severity) - Burned area boundaries
3. GeoMAC (legacy, now integrated into NIFC)

Output: CSV catalog of fires with metadata + GeoJSON perimeter files
"""

import os
import sys
import json
import time
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sanitize_fire_id(fire_id):
    """
    Sanitize fire ID to remove invalid filename characters.
    Removes newlines, tabs, and other problematic characters.
    """
    if not fire_id:
        return fire_id
    
    # Convert to string and remove problematic characters
    fire_id_str = str(fire_id)
    # Remove newlines, carriage returns, tabs
    fire_id_str = fire_id_str.replace('\n', '').replace('\r', '').replace('\t', '')
    # Remove other invalid filename characters
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        fire_id_str = fire_id_str.replace(char, '_')
    # Remove leading/trailing whitespace
    fire_id_str = fire_id_str.strip()
    
    return fire_id_str

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# NIFC Historic GeoMAC Perimeters (yearly archives 2000-2019) - HAS FIRE PROGRESSIONS!
# These have multiple perimeters per fire showing spread over time
GEOMAC_HISTORIC_URL_TEMPLATE = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Historic_Geomac_Perimeters_{year}/FeatureServer/0/query"
GEOMAC_2019_URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Historic_GeoMAC_Perimeters_2019/FeatureServer/0/query"

# NIFC/WFIGS Current Year Perimeters (works for 2020+)
NIFC_CURRENT_URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query"

# InterAgency Fire Perimeter History (all years consolidated) - backup
NIFC_ALL_YEARS_URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query"

# MTBS Fire Perimeters (final burned area boundaries)
MTBS_PERIMETERS_URL = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer/1/query"

# NASA FIRMS archive for verification
FIRMS_ARCHIVE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Output paths
FIRE_CATALOG_PATH = os.path.join(RAW_DATA_DIR, "fire_catalog.csv")
PERIMETERS_DIR = os.path.join(RAW_DATA_DIR, "perimeters")
os.makedirs(PERIMETERS_DIR, exist_ok=True)


class FirePerimeterFetcher:
    """Fetches and processes historical fire perimeter data."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PyroCast-FireSpreadModel/1.0'
        })
        self.fires = []
        
    def fetch_nifc_perimeters(self, year):
        """
        Fetch fire perimeters from NIFC for a specific year.
        Uses different endpoints based on year:
        - 2000-2019: Historic GeoMAC archives (has fire progressions!)
        - 2020+: WFIGS current perimeters
        """
        logger.info(f"Fetching NIFC perimeters for {year}...")
        
        # Choose the right endpoint based on year
        if year <= 2018:
            base_url = GEOMAC_HISTORIC_URL_TEMPLATE.format(year=year)
            use_geomac = True
        elif year == 2019:
            base_url = GEOMAC_2019_URL
            use_geomac = True
        else:
            base_url = NIFC_CURRENT_URL
            use_geomac = False
        
        fires_found = []
        offset = 0
        batch_size = 500  # Smaller batches to avoid timeouts
        
        while True:
            # Build query based on endpoint type
            if use_geomac:
                # GeoMAC uses lowercase field names - keep query simple for speed
                params = {
                    'where': f"gisacres >= {MIN_FIRE_SIZE_ACRES}",
                    'outFields': 'incidentname,gisacres,perimeterdatetime,uniquefireidentifier,state,OBJECTID',
                    'returnGeometry': 'true',
                    'f': 'geojson',
                    'resultOffset': offset,
                    'resultRecordCount': batch_size
                }
            else:
                # WFIGS uses attr_ prefix fields
                params = {
                    'where': f"attr_FireDiscoveryDateTime >= DATE '{year}-01-01' AND attr_FireDiscoveryDateTime < DATE '{year+1}-01-01' AND attr_FinalAcres >= {MIN_FIRE_SIZE_ACRES}",
                    'outFields': 'attr_IncidentName,attr_FinalAcres,attr_IncidentSize,attr_FireDiscoveryDateTime,attr_IrwinID,attr_POOState,poly_IncidentName,poly_GISAcres,OBJECTID',
                    'returnGeometry': 'true',
                    'f': 'geojson',
                    'resultOffset': offset,
                    'resultRecordCount': batch_size
                }
            
            try:
                response = self.session.get(base_url, params=params, timeout=120)
                response.raise_for_status()
                data = response.json()
                
                if 'features' not in data or len(data['features']) == 0:
                    break
                    
                for feature in data['features']:
                    props = feature.get('properties', {})
                    geom = feature.get('geometry')
                    
                    if geom is None:
                        continue
                    
                    # Parse fields based on source
                    if use_geomac:
                        # GeoMAC lowercase field names
                        fire_name = props.get('incidentname', 'Unknown')
                        acres = props.get('gisacres', 0) or 0
                        
                        # Parse perimeter datetime
                        discovery_date = None
                        date_val = props.get('perimeterdatetime')
                        if date_val:
                            try:
                                if isinstance(date_val, (int, float)):
                                    discovery_date = datetime.fromtimestamp(date_val / 1000)
                            except:
                                pass
                        
                        fire_id = sanitize_fire_id(props.get('uniquefireidentifier', props.get('OBJECTID', f'geomac_{year}_{len(fires_found)}')))
                        state = props.get('state', '')
                        source = 'GeoMAC_Historic'
                    else:
                        # WFIGS attr_ prefix fields
                        fire_name = 'Unknown'
                        for name_field in ['attr_IncidentName', 'poly_IncidentName', 'IncidentName']:
                            if name_field in props and props[name_field]:
                                fire_name = props[name_field]
                                break
                        
                        acres = 0
                        for acres_field in ['attr_FinalAcres', 'attr_IncidentSize', 'poly_GISAcres']:
                            if acres_field in props and props[acres_field]:
                                try:
                                    acres = float(props[acres_field])
                                    break
                                except:
                                    continue
                        
                        discovery_date = None
                        date_val = props.get('attr_FireDiscoveryDateTime')
                        if date_val:
                            try:
                                if isinstance(date_val, (int, float)):
                                    discovery_date = datetime.fromtimestamp(date_val / 1000)
                            except:
                                pass
                        
                        fire_id = sanitize_fire_id(props.get('attr_IrwinID', props.get('OBJECTID', f'wfigs_{year}_{len(fires_found)}')))
                        state = props.get('attr_POOState', '')
                        source = 'WFIGS_Current'
                    
                    fire_info = {
                        'fire_id': fire_id,
                        'fire_name': fire_name,
                        'year': year,
                        'acres': acres,
                        'start_date': discovery_date,
                        'state': state,
                        'source': source,
                        'geometry': geom
                    }
                    
                    # Filter by minimum size
                    if fire_info['acres'] >= MIN_FIRE_SIZE_ACRES:
                        fires_found.append(fire_info)
                
                logger.info(f"  Fetched {len(data['features'])} features (offset {offset})")
                
                if len(data['features']) < batch_size:
                    break
                    
                offset += batch_size
                time.sleep(0.3)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching NIFC data for {year}: {e}")
                break
        
        logger.info(f"Found {len(fires_found)} fire perimeters >= {MIN_FIRE_SIZE_ACRES} acres for {year}")
        return fires_found
    
    def fetch_mtbs_fires(self, year):
        """
        Fetch fire data from MTBS (Monitoring Trends in Burn Severity).
        MTBS provides high-quality burned area data with better temporal info.
        """
        logger.info(f"Fetching MTBS fires for {year}...")
        
        fires_found = []
        
        # MTBS uses YEAR field (integer) not date timestamps
        params = {
            'where': f"YEAR = {year} AND ACRES >= {MIN_FIRE_SIZE_ACRES}",
            'outFields': '*',
            'returnGeometry': 'true',
            'f': 'geojson',
            'resultRecordCount': 2000
        }
        
        try:
            response = self.session.get(MTBS_PERIMETERS_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if 'features' in data:
                for feature in data['features']:
                    props = feature.get('properties', {})
                    geom = feature.get('geometry')
                    
                    if geom is None:
                        continue
                    
                    # Parse ignition date from IG_DATE (integer YYYYMMDD format)
                    ig_date = None
                    ig_date_val = props.get('IG_DATE')
                    if ig_date_val:
                        try:
                            # IG_DATE is in YYYYMMDD format (e.g., 20200815)
                            date_str = str(ig_date_val)
                            if len(date_str) == 8:
                                ig_date = datetime.strptime(date_str, '%Y%m%d')
                        except:
                            # Try building from YEAR, STARTMONTH, STARTDAY
                            try:
                                ig_date = datetime(year, props.get('STARTMONTH', 1), props.get('STARTDAY', 1))
                            except:
                                ig_date = None
                    
                    fire_info = {
                        'fire_id': sanitize_fire_id(props.get('FIRE_ID', props.get('OBJECTID', ''))),
                        'fire_name': props.get('FIRE_NAME', 'Unknown'),
                        'year': year,
                        'acres': props.get('ACRES', 0) or 0,
                        'start_date': ig_date,
                        'state': '',  # MTBS doesn't have state in simple format
                        'source': 'MTBS',
                        'geometry': geom
                    }
                    
                    if fire_info['acres'] >= MIN_FIRE_SIZE_ACRES:
                        fires_found.append(fire_info)
            
            logger.info(f"Found {len(fires_found)} MTBS fires >= {MIN_FIRE_SIZE_ACRES} acres for {year}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching MTBS data for {year}: {e}")
        
        return fires_found
    
    def fetch_all_fires(self):
        """
        Fetch fires from all sources for the configured year range.
        Deduplicates based on spatial overlap and name similarity.
        """
        all_fires = []
        
        for year in range(START_YEAR, END_YEAR + 1):
            # Fetch from multiple sources
            nifc_fires = self.fetch_nifc_perimeters(year)
            mtbs_fires = self.fetch_mtbs_fires(year)
            
            # Combine and tag
            year_fires = nifc_fires + mtbs_fires
            all_fires.extend(year_fires)
            
            logger.info(f"Year {year}: {len(year_fires)} total fires collected")
            time.sleep(1)  # Rate limiting between years
        
        logger.info(f"Total fires collected: {len(all_fires)}")
        return all_fires
    
    def deduplicate_fires(self, fires):
        """
        Remove duplicate fires based on spatial overlap.
        Keeps the record with better metadata (MTBS > NIFC).
        """
        logger.info("Deduplicating fires based on spatial overlap...")
        
        # Convert to GeoDataFrame for spatial operations
        gdf_list = []
        for fire in fires:
            try:
                geom = shape(fire['geometry'])
                if geom.is_valid and not geom.is_empty:
                    gdf_list.append({
                        **{k: v for k, v in fire.items() if k != 'geometry'},
                        'geometry': geom
                    })
            except Exception as e:
                logger.warning(f"Invalid geometry for fire {fire.get('fire_name', 'Unknown')}: {e}")
        
        if not gdf_list:
            return []
        
        gdf = gpd.GeoDataFrame(gdf_list, crs="EPSG:4326")
        
        # Sort by source priority (MTBS first, then GeoMAC, then WFIGS) and size (larger first)
        source_priority = {'MTBS': 0, 'GeoMAC_Historic': 1, 'WFIGS_Current': 2, 'NIFC_Historic': 3, 'NIFC_Current': 4}
        gdf['source_priority'] = gdf['source'].map(source_priority).fillna(5)
        gdf = gdf.sort_values(['source_priority', 'acres'], ascending=[True, False])
        
        # Deduplicate using spatial index
        unique_fires = []
        used_indices = set()
        
        # Create spatial index
        sindex = gdf.sindex
        
        for idx, row in gdf.iterrows():
            if idx in used_indices:
                continue
            
            # Find potential duplicates
            bounds = row.geometry.bounds
            candidates = list(sindex.intersection(bounds))
            
            for candidate_idx in candidates:
                if candidate_idx != idx and candidate_idx not in used_indices:
                    candidate_row = gdf.loc[candidate_idx]
                    
                    # Check for significant overlap (IoU > 0.5)
                    try:
                        intersection = row.geometry.intersection(candidate_row.geometry)
                        union = row.geometry.union(candidate_row.geometry)
                        iou = intersection.area / union.area if union.area > 0 else 0
                        
                        if iou > 0.5:
                            used_indices.add(candidate_idx)
                    except:
                        pass
            
            # Keep this fire
            unique_fires.append({
                **{k: v for k, v in row.items() if k not in ['source_priority']},
                'geometry': mapping(row.geometry)
            })
            used_indices.add(idx)
        
        logger.info(f"Deduplicated to {len(unique_fires)} unique fires")
        return unique_fires
    
    def compute_fire_bounds(self, fires):
        """
        Compute bounding box and centroid for each fire.
        Used for patch generation.
        """
        for fire in fires:
            try:
                geom = shape(fire['geometry'])
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                centroid = geom.centroid
                
                fire['bbox'] = {
                    'min_lon': bounds[0],
                    'min_lat': bounds[1],
                    'max_lon': bounds[2],
                    'max_lat': bounds[3]
                }
                fire['centroid'] = {
                    'lon': centroid.x,
                    'lat': centroid.y
                }
                fire['area_km2'] = geom.area * 111 * 111  # Rough conversion at mid-latitudes
                
            except Exception as e:
                logger.warning(f"Error computing bounds for {fire.get('fire_name', 'Unknown')}: {e}")
                fire['bbox'] = None
                fire['centroid'] = None
                fire['area_km2'] = 0
        
        return fires
    
    def filter_and_rank_fires(self, fires):
        """
        Filter and rank fires based on:
        1. Data completeness (has start date)
        2. Size (larger fires = better spread data)
        3. Geographic diversity
        """
        # Filter out fires without valid bounding boxes
        valid_fires = [f for f in fires if f.get('bbox') is not None]
        
        # Prefer fires with known start dates
        dated_fires = [f for f in valid_fires if f.get('start_date') is not None]
        undated_fires = [f for f in valid_fires if f.get('start_date') is None]
        
        # Sort by size
        dated_fires.sort(key=lambda x: x.get('acres', 0), reverse=True)
        undated_fires.sort(key=lambda x: x.get('acres', 0), reverse=True)
        
        # Balance by year
        fires_by_year = {}
        for fire in dated_fires:
            year = fire['year']
            if year not in fires_by_year:
                fires_by_year[year] = []
            fires_by_year[year].append(fire)
        
        balanced_fires = []
        for year in range(START_YEAR, END_YEAR + 1):
            year_fires = fires_by_year.get(year, [])
            if TARGET_FIRES_PER_YEAR is not None:
                year_fires = year_fires[:TARGET_FIRES_PER_YEAR]
            balanced_fires.extend(year_fires)
        
        # Add undated fires up to limit
        remaining_slots = MAX_FIRES - len(balanced_fires) if MAX_FIRES else len(undated_fires)
        balanced_fires.extend(undated_fires[:remaining_slots])
        
        logger.info(f"Selected {len(balanced_fires)} fires for processing")
        return balanced_fires
    
    def save_catalog(self, fires):
        """Save fire catalog to CSV and individual perimeter GeoJSON files."""
        
        # Save catalog CSV
        catalog_data = []
        for fire in fires:
            catalog_data.append({
                'fire_id': fire.get('fire_id'),
                'fire_name': fire.get('fire_name'),
                'year': fire.get('year'),
                'acres': fire.get('acres'),
                'area_km2': fire.get('area_km2'),
                'start_date': fire.get('start_date'),
                'state': fire.get('state'),
                'source': fire.get('source'),
                'centroid_lon': fire.get('centroid', {}).get('lon'),
                'centroid_lat': fire.get('centroid', {}).get('lat'),
                'bbox_min_lon': fire.get('bbox', {}).get('min_lon'),
                'bbox_min_lat': fire.get('bbox', {}).get('min_lat'),
                'bbox_max_lon': fire.get('bbox', {}).get('max_lon'),
                'bbox_max_lat': fire.get('bbox', {}).get('max_lat'),
                'perimeter_file': f"perimeter_{fire.get('fire_id')}.geojson"
            })
        
        df = pd.DataFrame(catalog_data)
        df.to_csv(FIRE_CATALOG_PATH, index=False)
        logger.info(f"Saved fire catalog to {FIRE_CATALOG_PATH}")
        
        # Save individual perimeter GeoJSON files
        for fire in fires:
            perimeter_path = os.path.join(PERIMETERS_DIR, f"perimeter_{fire.get('fire_id')}.geojson")
            
            # Convert datetime objects to ISO strings for JSON serialization
            properties = {}
            for k, v in fire.items():
                if k == 'geometry':
                    continue
                if isinstance(v, datetime):
                    properties[k] = v.isoformat()
                elif hasattr(v, 'isoformat'):  # pandas Timestamp
                    properties[k] = v.isoformat()
                else:
                    properties[k] = v
            
            geojson_data = {
                'type': 'Feature',
                'properties': properties,
                'geometry': fire['geometry']
            }
            
            with open(perimeter_path, 'w') as f:
                json.dump(geojson_data, f)
        
        logger.info(f"Saved {len(fires)} perimeter files to {PERIMETERS_DIR}")
    
    def run(self):
        """Execute the complete fire perimeter fetching pipeline."""
        logger.info("=" * 60)
        logger.info("PyroCast Fire Spread Model - Perimeter Fetcher")
        logger.info("=" * 60)
        
        # Step 1: Fetch from all sources
        all_fires = self.fetch_all_fires()
        
        if not all_fires:
            logger.error("No fires found! Check API connectivity.")
            return
        
        # Step 2: Deduplicate
        unique_fires = self.deduplicate_fires(all_fires)
        
        # Step 3: Compute spatial info
        fires_with_bounds = self.compute_fire_bounds(unique_fires)
        
        # Step 4: Filter and rank
        selected_fires = self.filter_and_rank_fires(fires_with_bounds)
        
        # Step 5: Save
        self.save_catalog(selected_fires)
        
        logger.info("=" * 60)
        logger.info(f"Pipeline complete! Processed {len(selected_fires)} fires.")
        logger.info(f"Catalog: {FIRE_CATALOG_PATH}")
        logger.info(f"Perimeters: {PERIMETERS_DIR}")
        logger.info("=" * 60)
        
        return selected_fires


def main():
    """Main entry point."""
    fetcher = FirePerimeterFetcher()
    fires = fetcher.run()
    
    # Print summary statistics
    if fires:
        print("\n" + "=" * 60)
        print("FIRE CATALOG SUMMARY")
        print("=" * 60)
        
        df = pd.DataFrame([{
            'year': f['year'],
            'acres': f['acres'],
            'state': f.get('state', 'Unknown'),
            'has_date': f.get('start_date') is not None
        } for f in fires])
        
        print(f"\nTotal Fires: {len(fires)}")
        print(f"\nBy Year:")
        print(df.groupby('year').size())
        print(f"\nBy State (Top 10):")
        print(df['state'].value_counts().head(10))
        print(f"\nFires with Start Date: {df['has_date'].sum()} ({df['has_date'].mean()*100:.1f}%)")
        print(f"\nTotal Acres: {df['acres'].sum():,.0f}")
        print(f"Average Fire Size: {df['acres'].mean():,.0f} acres")
        print(f"Largest Fire: {df['acres'].max():,.0f} acres")
        

if __name__ == "__main__":
    main()
