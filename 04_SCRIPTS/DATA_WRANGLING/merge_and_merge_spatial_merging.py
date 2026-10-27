#!/usr/bin/env python3
"""Combined merge workflow: merge_and_merge, followed by spatial_merging.
"""
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

#
# ================= BEGIN merge_and_merge.ipynb =================

# ===== merge_and_merge.ipynb: cell 0 (code) =====
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString

# ===== merge_and_merge.ipynb: cell 1 (code) =====

# 1. Configuration
ROOT_DIR = '../02_INTERMEDIATE/'  # Point this to your master folder path
TARGET_FILE_NAME = 'individual_bamboo_params_breast_height_v1.gpkg'
OUTPUT_FILE_PATH = '../02_INTERMEDIATE/merged_drone_bamboo_params_breast_height.gpkg'

# ===== merge_and_merge.ipynb: cell 2 (code) =====
# 2. Extract PSP subfolders and file paths
gpkg_files = []
for dirpath, _, filenames in os.walk(ROOT_DIR):
    if TARGET_FILE_NAME in filenames:
        gpkg_files.append(os.path.join(dirpath, TARGET_FILE_NAME))

# ===== merge_and_merge.ipynb: cell 3 (code) =====
# 3. Process, append calculations, and archive
gdf_list = []
for file_path in gpkg_files:
    # Extract the name of the folder containing the file as the PSP ID
    psp_id = os.path.basename(os.path.dirname(file_path))
    
    gdf = gpd.read_file(file_path)
    
   
    gdf['psp_id'] = psp_id
    
    # Save updates directly into the individual subfolder files
    gdf.to_file(file_path, driver='GPKG')
    
    gdf_list.append(gdf)

# ===== merge_and_merge.ipynb: cell 4 (code) =====

# 4. Standardize Coordinate Systems and Merge
if gdf_list:
    target_crs = gdf_list[0].crs
    
    # Reproject layer datasets dynamically if spatial reference systems mismatch across plots
    standardized_gdfs = [g.to_crs(target_crs) if g.crs != target_crs else g for g in gdf_list]
    
    merged_gdf = gpd.GeoDataFrame(pd.concat(standardized_gdfs, ignore_index=True), crs=target_crs)
    merged_gdf.to_file(OUTPUT_FILE_PATH, driver='GPKG')
    print(f"Success! Master compilation file saved to {OUTPUT_FILE_PATH}")
    print(f"Total PSP plots processed: {len(gdf_list)}")
else:
    print(f"No files matching '{TARGET_FILE_NAME}' were found under the specified root directory.")

# ===== merge_and_merge.ipynb: cell 5 (code) =====

# ================== END merge_and_merge.ipynb ==================

#
# ================= BEGIN spatial_merging.ipynb =================

# ===== spatial_merging.ipynb: cell 0 (markdown) =====
# ##### This is a notebook to execute merges for 3 datasets while preserving:
# * Data Lineage
# * Correctness of what is being merged to what
# * Knowing which dataset joins to which(what is the base dataset)

# ===== spatial_merging.ipynb: cell 1 (code) =====
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from scipy.spatial import cKDTree

# ===== spatial_merging.ipynb: cell 2 (code) =====
ecoplanet_dataset_path = '../../02_INTERMEDIATE/final_bamboo_parameters_v03.gpkg'
acre_drone_dataset_path = '../../02_INTERMEDIATE/merged_drone_bamboo_params_breast_height_centroids.gpkg'
# acre_height_dataset_path = '../02_INTERMEDIATE/acre_height_validation.gpkg'

# ===== spatial_merging.ipynb: cell 3 (code) =====
ecoplanet_df = gpd.read_file(ecoplanet_dataset_path)
drone_df = gpd.read_file(acre_drone_dataset_path)

# ===== spatial_merging.ipynb: cell 4 (code) =====
drone_df['geometry'].head()

# ===== spatial_merging.ipynb: cell 5 (code) =====
drone_df = gpd.GeoDataFrame(drone_df, geometry="geometry", crs="EPSG:4326")
ecoplanet_df = gpd.GeoDataFrame(
    ecoplanet_df,
    geometry=gpd.GeoSeries.from_wkt(ecoplanet_df['geometry']),
    crs="EPSG:4326",
)

# ===== spatial_merging.ipynb: cell 6 (code) =====
ecoplanet_df.geometry.head()

# ===== spatial_merging.ipynb: cell 7 (code) =====
drone_df.columns

# ===== spatial_merging.ipynb: cell 8 (code) =====
drone_df['fid'] = drone_df.index + 1

# ===== spatial_merging.ipynb: cell 9 (code) =====
ecoplanet_df.columns

# ===== spatial_merging.ipynb: cell 10 (markdown) =====
# ##### Add prefixes to maintain data lineage

# ===== spatial_merging.ipynb: cell 11 (code) =====
ecoplanet_df.columns = [f"ecoplanet_{col}" if col != 'geometry' else col for col in ecoplanet_df.columns]
drone_df.columns = [f"acre_drone_{col}" if col != 'geometry' else col for col in drone_df.columns]

# ===== spatial_merging.ipynb: cell 12 (code) =====
ecoplanet_df.columns

# ===== spatial_merging.ipynb: cell 13 (code) =====
drone_df.columns

# ===== spatial_merging.ipynb: cell 14 (markdown) =====
# ##### Local coordinate system to ensure accurate distance matching

# ===== spatial_merging.ipynb: cell 15 (code) =====
crs = 'EPSG:32735'
ecoplanet_df = ecoplanet_df.to_crs(crs)
drone_df = drone_df.to_crs(crs)

# ===== spatial_merging.ipynb: cell 16 (code) =====
drone_df.head()

# ===== spatial_merging.ipynb: cell 17 (code) =====
drone_df.isna().sum()

# ===== spatial_merging.ipynb: cell 18 (markdown) =====
# ##### Nearest neighbour analysis

# ===== spatial_merging.ipynb: cell 19 (code) =====
# 4. Extract coordinates for fast Nearest-Neighbor matching using a KDTree
owner_coords = list(zip(ecoplanet_df.geometry.x, ecoplanet_df.geometry.y))
drone_coords = list(zip(drone_df.geometry.x, drone_df.geometry.y))
drone_tree = cKDTree(drone_coords)

# ===== spatial_merging.ipynb: cell 20 (code) =====
# Find the single closest drone point for every ecoplanet point
MAX_DISTANCE = 2.0  # 2 meters max cap
distances, indices = drone_tree.query(owner_coords, k=1, distance_upper_bound=MAX_DISTANCE)

# ===== spatial_merging.ipynb: cell 21 (code) =====
 #Build the visual inspection dataset
visual_lines = []

# ===== spatial_merging.ipynb: cell 22 (code) =====
for i, (owner_idx, drone_idx) in enumerate(enumerate(indices)):
    owner_point = ecoplanet_df.geometry.iloc[owner_idx]
    match_distance = distances[owner_idx]
    
    # Check if a valid match was found within 2 meters
    if drone_idx < drone_tree.n and match_distance <= MAX_DISTANCE:
        drone_point = drone_df.geometry.iloc[drone_idx]
        # Create a line connecting the matched clumps
        line = LineString([owner_point, drone_point])
        status = "Matched"
        
        # Pull drone data since it's a valid match
        drone_id = drone_df["acre_drone_fid"].iloc[drone_idx] # Update col name
        drone_area = drone_df["acre_drone_fid"].iloc[drone_idx] # Update col name
    else:
        # If it failed the 2m cap, create a tiny 10cm dummy line pointing north
        # This keeps the unmapped point visible in the line layer for your audit
        line = LineString([owner_point, (owner_point.x, owner_point.y + 0.1)])
        status = "Missed by Drone (No Match)"
        match_distance = np.nan
        drone_id = np.nan
        drone_area = np.nan

    # Combine attributes into the diagnostic record
    line_data = {
        "geometry": line,
        "match_status": status,
        "match_dist_m": match_distance,
        "owner_id": ecoplanet_df["ecoplanet_clump_id"].iloc[owner_idx], # Update col name
        "owner_agb": ecoplanet_df["ecoplanet_total_agb"].iloc[owner_idx],
        "drone_id": drone_id,
        "drone_canopy_area": drone_area
    }
    visual_lines.append(line_data)

# Convert list to a GeoDataFrame
inspection_gdf = gpd.GeoDataFrame(visual_lines, crs=crs)

# 6. Export back to WGS84 for perfect overlay on satellite imagery in QGIS
inspection_gdf.to_crs("EPSG:4326").to_file(f"../../02_INTERMEDIATE/capped_match_inspection.gpkg", driver="GPKG")

print("Inspection file 'capped_match_inspection.gpkg' created successfully with a 2m cap!")

# ===== spatial_merging.ipynb: cell 23 (markdown) =====
# #### Onto the actual merging of the dataframes

# ===== spatial_merging.ipynb: cell 24 (code) =====
visual_lines

# ===== spatial_merging.ipynb: cell 25 (code) =====
drone_df.columns

# ===== spatial_merging.ipynb: cell 26 (code) =====
ecoplanet_df.columns

# ===== spatial_merging.ipynb: cell 27 (code) =====
ecoplanet_df = ecoplanet_df.rename(columns={'ecoplanet_psp_id':'ecoplanet_psp'})

# ===== spatial_merging.ipynb: cell 28 (code) =====
# ------------------------------------------------------------------------------
# 1. Base Setup and Drone Column Reset
# ------------------------------------------------------------------------------
final_production_gdf = ecoplanet_df.copy()

# Explicitly set the dataset year column to 2026
# final_production_gdf["year"] = 2026
print(f"Starting merge for {len(final_production_gdf)} EcoPlanet clumps for Year 2026...")
# Clean up prefixes from the base dataset (e.g., 'ecoplanet_height' -> 'height')
final_production_gdf.columns = [
    col.replace("ecoplanet_", "") for col in final_production_gdf.columns
]

# Define the precise drone metrics columns you want to replace
drone_cols_to_pull = [
    'breast_height_points', 'breast_height_valid_points',
       'breast_height_filtered_points', 'breast_height_cleaning_distance',
       'breast_height_area', 'breast_height_perimeter_m',
       'breast_height_diameter', 'bh_outside_area_m2'
]
# Clean up the column headers of your incoming new drone dataframe to match
drone_df.columns = [
    col.replace("acre_drone_", "") for col in drone_df.columns
]
# CRITICAL TWEAK: Drop any existing versions of these drone columns to avoid conflicts,
# while keeping all other anchor data columns completely untouched.
existing_drone_cols = [col for col in drone_cols_to_pull if col in final_production_gdf.columns]
if existing_drone_cols:
    final_production_gdf = final_production_gdf.drop(columns=existing_drone_cols)
    print(f"Dropped {len(existing_drone_cols)} old drone metric columns for clean replacement.")

# Initialize the fresh drone columns as object type to prevent TypeErrors during assignment
for col in drone_cols_to_pull:
    final_production_gdf[col] = None

# ------------------------------------------------------------------------------
# 2. Extract and assign NEW drone metrics using your verified inspection indices
# ------------------------------------------------------------------------------
# Generate your coordinates from the anchor dataframe for KDTree querying
owner_coords = list(zip(final_production_gdf.geometry.x, final_production_gdf.geometry.y))

# Assuming drone_df contains your newly re-done drone metrics/centroids
drone_coords = list(zip(drone_df.geometry.x, drone_df.geometry.y))
drone_tree = cKDTree(drone_coords)
distances, indices = drone_tree.query(owner_coords, k=1, distance_upper_bound=MAX_DISTANCE)

# Mask out pairings that exceed your distance thresholds or index bounds
drone_match_mask = (indices < drone_tree.n) & (distances <= MAX_DISTANCE)

# Vectorized transfer of matched new drone records
if drone_match_mask.any():
    matched_drone_indices = indices[drone_match_mask]
    for col in drone_cols_to_pull:
        final_production_gdf.loc[drone_match_mask, col] = drone_df[col].iloc[matched_drone_indices].values

# Add or overwrite tracking column to separate matched clumps from missed drone ones
final_production_gdf["drone_match_status"] = np.where(drone_match_mask, "Matched", "Missed by Drone")

# ------------------------------------------------------------------------------
# 4. Verification Check & Export
# ------------------------------------------------------------------------------
# Print out columns to visually verify the spatial append worked perfectly
print("\n--- Verified Output Columns ---")
validation_cols_created = [col for col in final_production_gdf.columns if col.startswith('val_')]
print(f"Total columns in dataset: {len(final_production_gdf.columns)}")
print(f"Created/Retained {len(validation_cols_created)} validation columns: {validation_cols_created}")

output_production_path = "../../02_INTERMEDIATE/final_bamboo_parameters_v09.gpkg"
final_production_gdf.to_file(output_production_path, driver="GPKG")

print("\n--- Final Merge Summary ---")
print(f"Master file saved successfully to: {output_production_path}")
print(f"Total anchor EcoPlanet clumps: {len(final_production_gdf)}")
print(f"Total clumps paired with drone metrics: {drone_match_mask.sum()}")

# ===== spatial_merging.ipynb: cell 29 (code) =====
final_production_gdf.columns

# ===== spatial_merging.ipynb: cell 30 (code) =====
len(final_production_gdf)

# ===== spatial_merging.ipynb: cell 31 (code) =====
final_production_gdf = final_production_gdf[['psp', 'year', 'date_of_collection', 'clump_id', 'total_culms', 'total_agb',
       'max_culm_diameter_cm', 'avg_culm_diameter_cm', 'min_culm_diameter_cm',
       'median_culm_diameter_cm', 'max_culm_agb', 'avg_culm_agb',
       'min_culm_agb', 'median_culm_agb', 'hmin', 'hmax', 'hmean', 'hstd', 'hmedian', 'total_points',
       'canopy_volume_m3', 'gap_fraction', 'canopy_area', 'val_height', 'val_canopy_diameter', 'breast_height_points', 'breast_height_valid_points',
       'breast_height_filtered_points', 'breast_height_cleaning_distance',
       'breast_height_area', 'breast_height_perimeter_m',
       'breast_height_diameter', 'bh_outside_area_m2', 'geometry']]

# ===== spatial_merging.ipynb: cell 32 (code) =====
final_production_gdf.to_file(output_production_path, driver="GPKG")

# ===== spatial_merging.ipynb: cell 33 (code) =====

# ================== END spatial_merging.ipynb ==================
