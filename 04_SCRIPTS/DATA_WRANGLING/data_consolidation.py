#!/usr/bin/env python3
"""Data Consolidation notebook exported as a Python script.

Run with Python from any working directory. Relative data paths resolve
from this script directory, matching the original notebook location.
"""
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# ===== data_consolidation.ipynb: cell 0 (markdown) =====
# ### DATA CLEANING AND CONSOLIDATION FOR ECOPLANET BAMBOO
# ### OUTCOME: ONE TABLE CONTAINING ALL DATA FROM DIFFERENT SOURCES PER CLUMP FOR MODELLING]
# ### AUTHOR: ELKANA KIPRUTO

# ===== data_consolidation.ipynb: cell 1 (markdown) =====
# #### Consolidation process for different sources of data:
# * Ecoplanet Ground Truth Data (June 2025, March 2026, June 2026)
# * Acre Insights Drone Derived Data
# * Acre Insights Ground Truth Data

# ===== data_consolidation.ipynb: cell 2 (markdown) =====
# #### Step 1: Importing the libraries

# ===== data_consolidation.ipynb: cell 3 (code) =====
import pandas as pd
import geopandas as gpd
# import xml.etree.ElementTree as ET
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import contextily as ctx
import os
import re
# import fiona
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# ===== data_consolidation.ipynb: cell 4 (markdown) =====
# #### Define the standard final columns we are targeting

# ===== data_consolidation.ipynb: cell 5 (code) =====
target_cols = [
    'psp_id',
    'date_of_measurement',
    'geometry',
    'centroid',
    'clump_id',
    'canopy_area',
    'canopy_volume',
    'hmax',
    'hmin',
    'hmean',
    'hmedian',
    'hstd',
    'gap_fraction',
    'gt_hmax',
    'gt_canopy_diam',
    'gt_dbh_max',
    'gt_dbh_mean',
    'gt_dbh_min',
    'gt_no_of_culms',
    'biomass'
]

# ===== data_consolidation.ipynb: cell 6 (code) =====
df_final_template = pd.DataFrame(columns=target_cols)

# ===== data_consolidation.ipynb: cell 7 (code) =====
print(f'Target template schema initialized successfully. Total columns: {len(df_final_template.columns)}')

# ===== data_consolidation.ipynb: cell 8 (markdown) =====
# #### Reading in the paths and ground truth datasets

# ===== data_consolidation.ipynb: cell 9 (code) =====
june_25_path = '../01_RAW/7PSPs_data_June25.xlsx'
march_26_path = '../01_RAW/7PSPs_data_March26.xlsx'
acre_psp_01_path = '../02_INTERMEDIATE/PSP-1A-APM-01/individual_bamboo_params.gpkg'
acre_psp_07_path = '../02_INTERMEDIATE/PSP-1A-APM-07/individual_bamboo_params.gpkg'
acre_psp_08_path = '../02_INTERMEDIATE/PSP-1A-APM-08/individual_bamboo_params.gpkg'

#ground truth validation kml
acre_ground_validation_path = '../01_RAW/height_validation_acre.kml'

# ===== data_consolidation.ipynb: cell 10 (code) =====
#initial done 3 psps
df_apm1_june25 = pd.read_excel(june_25_path, sheet_name='PSP 1a-APM-1')
df_apm1_march26 = pd.read_excel(march_26_path, sheet_name='PSP 1a-APM-1')
df_apm7_june25 = pd.read_excel(june_25_path, sheet_name='PSP 1a-APM-7')
df_apm7_march26 = pd.read_excel(march_26_path, sheet_name='PSP 1a-APM-7')
df_apm8_june25 = pd.read_excel(june_25_path, sheet_name='PSP 1a-APM-8')
df_apm8_march26 = pd.read_excel(march_26_path, sheet_name='PSP 1a-APM-8')

#the other 4 psps
df_apm2_june25 = pd.read_excel(june_25_path, sheet_name='PSP 1a-APM-2')
df_apm2_march26 = pd.read_excel(march_26_path, sheet_name='PSP 1a-APM-2')
df_apm13_june25 = pd.read_excel(june_25_path, sheet_name='PSP 1a-APM-13')
df_apm13_march26 = pd.read_excel(march_26_path, sheet_name='PSP 1a-APM-13')
df_gb3_june25 = pd.read_excel(june_25_path, sheet_name='PSP 1a-GB-3')
df_gb3_march26 = pd.read_excel(march_26_path, sheet_name='PSP 1a-GB-3')
df_gb5_june25 = pd.read_excel(june_25_path, sheet_name='PSP 1a-GB-5')
df_gb5_march26 = pd.read_excel(march_26_path, sheet_name='PSP 1a-GB-5')

# ===== data_consolidation.ipynb: cell 11 (markdown) =====
# #### Cleaning the ground truth data

# ===== data_consolidation.ipynb: cell 12 (code) =====
df_apm1_june25.head()

# ===== data_consolidation.ipynb: cell 13 (code) =====
df_apm1_march26.head()

# ===== data_consolidation.ipynb: cell 14 (code) =====
to_be_cleaned = [df_apm1_june25, df_apm1_march26, df_apm7_june25, df_apm7_march26, df_apm8_june25, df_apm8_march26, df_apm2_june25, df_apm2_march26,
                 df_apm13_june25, df_apm13_march26, df_gb3_june25, df_gb3_march26, df_gb5_june25, df_gb5_march26
                 ]

# ===== data_consolidation.ipynb: cell 15 (markdown) =====
# #### Function to parse all the data from the excel workbook to pandas dataframe

# ===== data_consolidation.ipynb: cell 16 (code) =====
def clean_biomass_df(df):
    """
    Cleans and structures biomass data from the provided excel sheet layout.
    Preserves clumps with no records by creating a single row placeholder 
    noting "No records found" instead of skipping them.
    """
    plot_id = None
    date_val = None
    
    # 1. Target exact cell coordinates for global Plot Metadata
    for r in range(min(3, df.shape[0])):
        row_values = [str(x).strip() for x in df.iloc[r].tolist()]
        for idx, val in enumerate(row_values):
            if '1a-' in val.lower() or 'apm' in val.lower():
                plot_id = val
            if 'DATE' in val.upper() and idx + 1 < len(row_values):
                date_val = row_values[idx + 1]

    # 2. Dynamically find all Clump start columns from Row index 2
    clump_indices = []
    if df.shape[0] > 2:
        row_2 = df.iloc[2].tolist()
        for idx, val in enumerate(row_2):
            if pd.notna(val) and 'CLUMP' in str(val).upper():
                clump_indices.append(idx)
                
    all_clumps = []
    num_cols = df.shape[1]
    
    # 3. Process each clump based on dynamic boundaries
    for i, start_col in enumerate(clump_indices):
        end_col = clump_indices[i + 1] if i + 1 < len(clump_indices) else num_cols
        
        # Slice the chunk for this specific clump
        clump_chunk = df.iloc[:, start_col:end_col]
        
        # Extract Clump ID
        clump_id = str(df.iloc[2, start_col]).strip()
        
        if df.shape[0] <= 5:
            continue
            
        # Extract data rows from row index 5 downward
        data_rows = clump_chunk.iloc[5:].copy()
        
        # Dynamic Column Naming using robust case-insensitive substring checks
        headers = [str(h).strip().upper() for h in clump_chunk.iloc[4].tolist()]
        col_names = []
        for idx, h in enumerate(headers):
            if 'NUMBER' in h:
                col_names.append('Culm_Number')
            elif 'DIAMETER (CM)' in h or 'DIAMETER_CM' in h:
                col_names.append('Culm_Diameter_cm')
            elif 'DIAMETER (MM)' in h or 'DIAMETER_MM' in h:
                col_names.append('Culm_Diameter_mm')
            elif 'AGB_CULM' in h or '(KG)' in h:
                col_names.append('AGB_Culm')
            else:
                col_names.append(f'Spacer_Col_{idx}')
                
        data_rows.columns = col_names
        
        # 4. Clean data rows and drop footer/totals
        # First, drop completely empty records in Culm_Number
        data_rows = data_rows.dropna(subset=['Culm_Number'])
        
        # Safe string evaluation to eliminate rows matching summary labels like 'AGB CLUMP'
        def is_summary_row(row):
            for val in row.values:
                if pd.notna(val) and 'AGB CLUMP' in str(val).upper():
                    return True
            return False
            
        summary_mask = data_rows.apply(is_summary_row, axis=1)
        data_rows = data_rows[~summary_mask]
        
        # Filter down to valid numeric indices to separate spacer values from numeric logs
        data_rows['Culm_Number_Numeric'] = pd.to_numeric(data_rows['Culm_Number'], errors='coerce')
        data_rows = data_rows.dropna(subset=['Culm_Number_Numeric']).drop(columns=['Culm_Number_Numeric'])
        
        # --- NEW LOGIC: Handle empty clumps gracefully ---
        if data_rows.empty:
            # Create a fallback single row dictionary filled with your preservation note
            fallback_row = {
                'Culm_Number': 'No records found',
                'Culm_Diameter_cm': 'No records found',
                'AGB_Culm': 'No records found'
            }
            # Fallback for mm tracking columns if present in sheet structures
            if 'Culm_Diameter_mm' in col_names:
                fallback_row['Culm_Diameter_mm'] = 'No records found'
                
            data_rows = pd.DataFrame([fallback_row])
        else:
            # Remove unwanted spacer columns dynamically if the data rows exist
            data_rows = data_rows.loc[:, ~data_rows.columns.str.startswith('Spacer_Col_')]
        
        # Inject metadata
        data_rows['Plot_ID'] = plot_id
        data_rows['Date'] = date_val
        data_rows['Clump_ID'] = clump_id
        
        all_clumps.append(data_rows)
        
    # 5. Concatenate all data chunks safely
    cols_order = ['Plot_ID', 'Date', 'Clump_ID', 'Culm_Number', 'Culm_Diameter_cm', 'AGB_Culm']
    
    if not all_clumps:
        return pd.DataFrame(columns=cols_order)
        
    cleaned_df = pd.concat(all_clumps, ignore_index=True)
    
    # Ensure all columns exist in final output schema even if empty
    for c in cols_order:
        if c not in cleaned_df.columns:
            cleaned_df[c] = None
            
    return cleaned_df[cols_order]

# ===== data_consolidation.ipynb: cell 17 (code) =====
master_list = []

for raw_df in to_be_cleaned:
    # Clean each already-loaded df using the function
    cleaned_data = clean_biomass_df(raw_df)
    
    # FIX 1: Only append if the dataframeA actually contains extracted rows
    if not cleaned_data.empty:
        master_list.append(cleaned_data)

# FIX 2: Prevent pd.concat crash if no valid data was found across all sheets
if master_list:
    final_dataset = pd.concat(master_list, ignore_index=True)
    
    # FIX 3: Safe datetime conversion with errors='coerce' to bypass messy strings
    final_dataset['Date'] = pd.to_datetime(final_dataset["Date"], errors='coerce', format="mixed")
    
    # Extract only the date component safely, ignoring nulls
    final_dataset['Year'] = final_dataset['Date'].dt.year
    final_dataset['Date'] = final_dataset["Date"].dt.date
else:
    # Fallback to an empty dataframe with correct schema if no data matched
    cols_order = ['Plot_ID', 'Date', 'Clump_ID', 'Culm_Number', 'Culm_Diameter_cm', 'AGB_Culm']
    final_dataset = pd.DataFrame(columns=cols_order)

# ===== data_consolidation.ipynb: cell 18 (code) =====
len(final_dataset)

# ===== data_consolidation.ipynb: cell 19 (code) =====
final_dataset.info()

# ===== data_consolidation.ipynb: cell 20 (code) =====
final_dataset[final_dataset['Culm_Number'] == 'No records found']

# ===== data_consolidation.ipynb: cell 21 (code) =====
# Columns to update
cols = ["Culm_Number", "Culm_Diameter_cm", "AGB_Culm"]

# Replace the text with 0 and convert columns to numeric types
for col in cols:
    final_dataset[col] = final_dataset[col].replace("No records found", 0)

# Cast Culm_Number to integer, others to float
final_dataset["Culm_Number"] = final_dataset["Culm_Number"].astype(int)
final_dataset["Culm_Diameter_cm"] = final_dataset["Culm_Diameter_cm"].astype(float)
final_dataset["AGB_Culm"] = final_dataset["AGB_Culm"].astype(float)

# ===== data_consolidation.ipynb: cell 22 (code) =====
final_dataset.info()

# ===== data_consolidation.ipynb: cell 23 (code) =====
df_2026 = final_dataset[final_dataset['Year'] == 2026].copy()
df_2025 = final_dataset[final_dataset['Year'] == 2025].copy()

# ===== data_consolidation.ipynb: cell 24 (code) =====
sns.histplot(df_2025['Culm_Number'])

# ===== data_consolidation.ipynb: cell 25 (code) =====
sns.histplot(df_2026['Culm_Number'])

# ===== data_consolidation.ipynb: cell 26 (code) =====
df_2026.describe()

# ===== data_consolidation.ipynb: cell 27 (code) =====
df_2025.describe()

# ===== data_consolidation.ipynb: cell 28 (markdown) =====
# ##### Clean the 2025 dataframe

# ===== data_consolidation.ipynb: cell 29 (code) =====
def clean_bamboo_2025_zeros(df):
    # 1. Subset the records for the year 2025
    df_2025 = df[df['Year'] == 2025].copy()
    
    # 2. Separate rows with real culms (> 0 diameter) from zero entries
    valid_culms = df_2025[df_2025['Culm_Diameter_cm'] > 0].copy()
    zero_culms = df_2025[df_2025['Culm_Diameter_cm'] == 0].copy()
    
    # 3. Handle clumps where every single record is 0
    # Find unique Plot+Clump combinations that have at least one valid culm
    valid_clump_keys = valid_culms[['Plot_ID', 'Clump_ID']].drop_duplicates()
    
    # Left join zero records with valid keys to flag clumps that have valid culms
    merged = zero_culms.merge(valid_clump_keys, on=['Plot_ID', 'Clump_ID'], how='left', indicator=True)
    
    # Filter for rows where the Plot+Clump combination has NO valid culms at all
    all_zero_clumps = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # Drop duplicates to keep exactly one record per all-zero clump
    preserved_clumps = all_zero_clumps.drop_duplicates(subset=['Plot_ID', 'Clump_ID']).copy()
    
    # Force the Culm_Number to 0 for these completely empty clumps
    preserved_clumps['Culm_Number'] = 0
    
    # 4. Recombine valid culms and preserved empty clumps
    cleaned_df = pd.concat([valid_culms, preserved_clumps]).sort_index()
    
    # 5. Recalculate and update the Culm_Number to reflect the true non-zero count
    # Count how many valid culms exist per unique Plot + Clump combination
    true_counts = valid_culms.groupby(['Plot_ID', 'Clump_ID']).size().reset_index(name='True_Count')
    
    # Map the true counts back to the cleaned dataframe
    cleaned_df = cleaned_df.merge(true_counts, on=['Plot_ID', 'Clump_ID'], how='left')
    cleaned_df['Culm_Number'] = cleaned_df['True_Count'].fillna(0).astype(int)
    cleaned_df = cleaned_df.drop(columns=['True_Count'])
    
    return cleaned_df

# ===== data_consolidation.ipynb: cell 30 (code) =====
df_2025_no_zeros = clean_bamboo_2025_zeros(final_dataset)

# ===== data_consolidation.ipynb: cell 31 (code) =====
df_2025_no_zeros.describe()

# ===== data_consolidation.ipynb: cell 32 (code) =====
df_2026.describe()

# ===== data_consolidation.ipynb: cell 33 (code) =====
final_dataset = pd.concat([df_2026, df_2025_no_zeros], ignore_index=True)

# ===== data_consolidation.ipynb: cell 34 (code) =====
len(final_dataset)

# ===== data_consolidation.ipynb: cell 35 (code) =====
final_dataset.head()

# ===== data_consolidation.ipynb: cell 36 (code) =====
final_dataset[final_dataset['Culm_Diameter_cm'].isna()]

# ===== data_consolidation.ipynb: cell 37 (code) =====
final_dataset.sample(10)

# ===== data_consolidation.ipynb: cell 38 (code) =====
final_dataset['Date'].unique()

# ===== data_consolidation.ipynb: cell 39 (code) =====
final_dataset.columns = final_dataset.columns.str.lower()

# ===== data_consolidation.ipynb: cell 40 (code) =====
final_dataset.head()

# ===== data_consolidation.ipynb: cell 41 (code) =====
final_dataset.info()

# ===== data_consolidation.ipynb: cell 42 (code) =====
msno.matrix(final_dataset)

# ===== data_consolidation.ipynb: cell 43 (code) =====
final_dataset['plot_id'].unique()

# ===== data_consolidation.ipynb: cell 44 (code) =====
# Strip whitespace and pad the final number with a leading zero if it is a single digit
final_dataset['plot_id'] = final_dataset['plot_id'].astype(str).str.replace(
    r'-(\d+)$', 
    lambda x: f"-{int(x.group(1)):02d}", 
    regex=True
)

# ===== data_consolidation.ipynb: cell 45 (code) =====
final_dataset['plot_id'].unique()

# ===== data_consolidation.ipynb: cell 46 (code) =====
len(final_dataset)

# ===== data_consolidation.ipynb: cell 47 (code) =====
final_dataset.info()

# ===== data_consolidation.ipynb: cell 48 (code) =====
final_dataset.to_csv("../02_INTERMEDIATE/7psps_with_biomass_ecoplanet.csv", index=False)

# ===== data_consolidation.ipynb: cell 49 (code) =====
df_gt_ecoplanet = pd.read_csv("../02_INTERMEDIATE/7psps_with_biomass_ecoplanet.csv")

# ===== data_consolidation.ipynb: cell 50 (code) =====
df_gt_ecoplanet.head()

# ===== data_consolidation.ipynb: cell 51 (markdown) =====
# #### Review completeness

# ===== data_consolidation.ipynb: cell 52 (code) =====
# 1. Get all unique plots and years actually present in your data
unique_plots = df_gt_ecoplanet['plot_id'].unique()
unique_years = df_gt_ecoplanet['year'].unique()

# 2. Generate the 15 expected clump IDs
expected_clumps = [f"CLUMP #{i}" for i in range(1, 16)]

# 3. Build a perfect reference grid of all required combinations
expected_grid = pd.MultiIndex.from_product(
    [unique_plots, unique_years, expected_clumps],
    names=['plot_id', 'year', 'clump_id']
)

# ===== data_consolidation.ipynb: cell 53 (code) =====
# 4. Convert your current data's columns into a MultiIndex
actual_grid = pd.MultiIndex.from_frame(df_gt_ecoplanet[['plot_id', 'year', 'clump_id']])

# 5. Find the combinations that exist in the expected grid but not in your data
missing_combinations = expected_grid.difference(actual_grid)

# 6. Convert the result into a readable DataFrame
missing_df = missing_combinations.to_frame(index=False)

# ===== data_consolidation.ipynb: cell 54 (code) =====
missing_df

# ===== data_consolidation.ipynb: cell 55 (markdown) =====
# ##### All PSPs are complete for all years: have 15 clumps recorded for each psp, each year

# ===== data_consolidation.ipynb: cell 56 (code) =====
df_gt_ecoplanet.info()

# ===== data_consolidation.ipynb: cell 57 (code) =====
df_gt_ecoplanet['culm_diameter_cm'] = df_gt_ecoplanet['culm_diameter_cm'].round(2)
df_gt_ecoplanet['agb_culm'] = df_gt_ecoplanet['agb_culm'].round(2)

# ===== data_consolidation.ipynb: cell 58 (markdown) =====
# #### Aggregate biomass values

# ===== data_consolidation.ipynb: cell 59 (code) =====
df_clean = df_gt_ecoplanet.copy()

# --- STEP 1.5: Temporarily turn 0s to NaN for biomass calculations ---
# This ensures 0s don't skew min, mean, or median downward.
df_clean["agb_culm_nan"] = df_clean["agb_culm"].replace(0, np.nan)

# --- STEP 2: Aggregate the Data ---
aggregated_df_7psps = (
    df_clean.groupby(["plot_id", "year", "clump_id"])
    .agg(
        # Calculate stats using the column where 0s are NaN (they will be ignored)
        max_culm_agb=("agb_culm_nan", "max"),
        avg_culm_agb=("agb_culm_nan", "mean"),
        min_culm_agb=("agb_culm_nan", "min"),
        median_culm_agb=("agb_culm_nan", "median"),
        total_agb=("agb_culm_nan", "sum"),
        # Culm count still looks at the original column to capture everything
        total_culms=("culm_number", "max"),
    )
    .reset_index()
)

# --- STEP 3: Handle completely empty clumps ---
# If a clump only had 0s, its stats are NaN. Turn them back to 0.0.
cols_to_zero = [
    "max_culm_agb",
    "avg_culm_agb",
    "min_culm_agb",
    "median_culm_agb",
    "total_agb"
]
aggregated_df_7psps[cols_to_zero] = aggregated_df_7psps[cols_to_zero].fillna(
    0.0
)

# ===== data_consolidation.ipynb: cell 60 (code) =====
aggregated_df_7psps.head()

# ===== data_consolidation.ipynb: cell 61 (code) =====
len(aggregated_df_7psps)

# ===== data_consolidation.ipynb: cell 62 (markdown) =====
# #### Merge with initially corrected geometries dataset

# ===== data_consolidation.ipynb: cell 63 (code) =====
corrected_geom_ecoplanet_path = '../02_INTERMEDIATE/ecoplanet_final_dataset_V01.gpkg'
gdf_corrected = gpd.read_file(corrected_geom_ecoplanet_path)

# ===== data_consolidation.ipynb: cell 64 (code) =====
gdf_corrected.head()

# ===== data_consolidation.ipynb: cell 65 (code) =====
len(gdf_corrected)

# ===== data_consolidation.ipynb: cell 66 (code) =====
# 1. Select only the necessary columns from the second (spatial) dataframe
df_spatial_selected = gdf_corrected[
    [
        "plot_id",
        "year",
        "clump_id",
        "max_culm_diameter_cm",
        "avg_culm_diameter_cm",
        "min_culm_diameter_cm",
        "median_culm_diameter_cm",
        "lat",
        "lon",
        "geometry",
    ]
]

# 2. Merge them together using an inner join on the 3 unique keys
merged_df = pd.merge(
    df_spatial_selected,
    aggregated_df_7psps,  # Your first df with AGB stats and total_culms
    on=["plot_id", "year", "clump_id"],
    how="inner",
)

# 3. Ensure Python recognizes it as a GeoDataFrame so you don't lose map functionality
merged_gpd = gpd.GeoDataFrame(merged_df, geometry="geometry", crs='EPSG:32735')

# ===== data_consolidation.ipynb: cell 67 (code) =====
len(merged_gpd)

# ===== data_consolidation.ipynb: cell 68 (code) =====
merged_gpd.head()

# ===== data_consolidation.ipynb: cell 69 (code) =====
# 1. Extract the number from 'clump_id' as an integer for proper sorting
merged_gpd["clump_sort_num"] = (
    merged_gpd["clump_id"].str.extract(r"(\d+)").astype(int)
)

# 2. Sort by plot_id, year, and then the numeric clump order
merged_gpd = merged_gpd.sort_values(
    by=["plot_id", "year", "clump_sort_num"]
).reset_index(drop=True)

# 3. Drop the temporary sorting column to keep the dataset clean
merged_gpd = merged_gpd.drop(columns=["clump_sort_num"])

# ===== data_consolidation.ipynb: cell 70 (code) =====
merged_gpd.columns

# ===== data_consolidation.ipynb: cell 71 (code) =====
merged_gpd = merged_gpd[['plot_id', 'year', 'clump_id','total_culms','total_agb', 'max_culm_diameter_cm',
       'avg_culm_diameter_cm', 'min_culm_diameter_cm',
       'median_culm_diameter_cm', 'max_culm_agb',
       'avg_culm_agb', 'min_culm_agb', 'median_culm_agb','lat', 'lon', 'geometry']]

# ===== data_consolidation.ipynb: cell 72 (code) =====
##round all numerics to 2 decimal places
merged_gpd.info()

# ===== data_consolidation.ipynb: cell 73 (code) =====
merged_gpd = merged_gpd.astype({'max_culm_diameter_cm' : 'float64', 'avg_culm_diameter_cm' : 'float64', 'min_culm_diameter_cm' : 'float64',
                                'median_culm_diameter_cm' : 'float64'})

# ===== data_consolidation.ipynb: cell 74 (code) =====
merged_gpd.info()

# ===== data_consolidation.ipynb: cell 75 (code) =====
merged_gpd.head()

# ===== data_consolidation.ipynb: cell 76 (code) =====
#find float columns and round them to 2 decimal places
float_cols = merged_gpd.select_dtypes(include=['float']).columns

# 2. Round those columns and assign them back
merged_gpd[float_cols] = merged_gpd[float_cols].round(2)

# ===== data_consolidation.ipynb: cell 77 (code) =====
merged_gpd.head()

# ===== data_consolidation.ipynb: cell 78 (code) =====
merged_gpd.to_file(f'../02_INTERMEDIATE/ecoplanet_final_dataset_V01.gpkg', index=False)
merged_gpd.to_csv(f'../02_INTERMEDIATE/ecoplanet_final_dataset_V01.csv', index=False)

# ===== data_consolidation.ipynb: cell 79 (code) =====
# # 1. Split the coordinate string into two separate columns
# df_gt_ecoplanet[["lat", "lon"]] = df_gt_ecoplanet["clump_gps"].str.split(",", expand=True)

# # 2. Convert coordinates to float numbers
# df_gt_ecoplanet["lat"] = df_gt_ecoplanet["lat"].astype(float)
# df_gt_ecoplanet["lon"] = df_gt_ecoplanet["lon"].astype(float)

# # 3. Create the GeoDataFrame (using EPSG:4326 for standard GPS WGS84)
# gdf_gt_ecoplanet = gpd.GeoDataFrame(
#     df_gt_ecoplanet,
#     geometry=gpd.points_from_xy(df_gt_ecoplanet["lon"], df_gt_ecoplanet["lat"]),
#     crs="EPSG:4326",
# )

# gdf_gt_ecoplanet_26 = gdf_gt_ecoplanet[gdf_gt_ecoplanet['year'] == 2026]
# gdf_gt_ecoplanet_25 = gdf_gt_ecoplanet[gdf_gt_ecoplanet['year'] == 2025]
# gdf_gt_ecoplanet.to_file(f'../02_INTERMEDIATE/psp_biomass_ecoplanet.gpkg', index=False)

# ===== data_consolidation.ipynb: cell 80 (code) =====
gdf_gt_ecoplanet.head(15)

# ===== data_consolidation.ipynb: cell 81 (code) =====
gdf_gt_ecoplanet.info()

# ===== data_consolidation.ipynb: cell 82 (code) =====
gdf_gt_ecoplanet['clump_id'].unique()

# ===== data_consolidation.ipynb: cell 83 (code) =====
gdf_clean = gdf_gt_ecoplanet.copy()
gdf_clean['culm_diameter_cm_cleaned'] = gdf_clean['culm_diameter_cm'].replace(0, pd.NA)

# --- STEP 2: Aggregate the Data ---
aggregated_df = gdf_clean.groupby(['plot_id', 'year', 'clump_id']).agg(
    # Diameter stats (ignoring the 0s we turned into NaNs)
    max_culm_diameter_cm=('culm_diameter_cm_cleaned', 'max'),
    avg_culm_diameter_cm=('culm_diameter_cm_cleaned', 'mean'),
    min_culm_diameter_cm=('culm_diameter_cm_cleaned', 'min'),
    median_culm_diameter_cm=('culm_diameter_cm_cleaned', 'median'),
    
    # Culm count (takes max of all rows, including those with 0 diameter)
    total_culms=('culm_number', 'max'),
    
    # Retain spatial information (takes the first instance since it's identical)
    clump_gps=('clump_gps', 'first'),
    lat=('lat', 'first'),
    lon=('lon', 'first'),
    geometry=('geometry', 'first')
).reset_index()

# --- STEP 3: Handle completely empty clumps ---
# If a clump only had 0s (no measurable plants), its min/avg/median/max will be NaN. 
# We should cleanly turn those NaNs back to 0.0.
cols_to_zero = ['max_culm_diameter_cm', 'avg_culm_diameter_cm', 'min_culm_diameter_cm', 'median_culm_diameter_cm']
aggregated_df[cols_to_zero] = aggregated_df[cols_to_zero].fillna(0.0)

# ===== data_consolidation.ipynb: cell 84 (code) =====
aggregated_df.head()

# ===== data_consolidation.ipynb: cell 85 (markdown) =====
# #### Is the df complete

# ===== data_consolidation.ipynb: cell 86 (code) =====
len(aggregated_df)

# ===== data_consolidation.ipynb: cell 87 (code) =====
# 1. Get all unique plots and years actually present in your data
unique_plots = df_gt_ecoplanet['plot_id'].unique()
unique_years = df_gt_ecoplanet['year'].unique()

# 2. Generate the 15 expected clump IDs
expected_clumps = [f"CLUMP #{i}" for i in range(1, 16)]

# 3. Build a perfect reference grid of all required combinations
expected_grid = pd.MultiIndex.from_product(
    [unique_plots, unique_years, expected_clumps],
    names=['plot_id', 'year', 'clump_id']
)

# ===== data_consolidation.ipynb: cell 88 (code) =====
# 4. Convert your current data's columns into a MultiIndex
actual_grid = pd.MultiIndex.from_frame(aggregated_df[['plot_id', 'year', 'clump_id']])

# 5. Find the combinations that exist in the expected grid but not in your data
missing_combinations = expected_grid.difference(actual_grid)

# 6. Convert the result into a readable DataFrame
missing_df = missing_combinations.to_frame(index=False)

# ===== data_consolidation.ipynb: cell 89 (code) =====
missing_df

# ===== data_consolidation.ipynb: cell 90 (markdown) =====
# #### The dataframe is complete

# ===== data_consolidation.ipynb: cell 91 (code) =====
aggregated_gdf = gpd.GeoDataFrame(aggregated_df, 
    geometry='geometry',
    crs="EPSG:4326")

# ===== data_consolidation.ipynb: cell 92 (code) =====
aggregated_gdf = aggregated_gdf.to_crs('EPSG:32735')

# ===== data_consolidation.ipynb: cell 93 (code) =====
aggregated_gdf.to_file('../02_INTERMEDIATE/ecoplanet_final_dataset.gpkg', index=False)
aggregated_gdf.to_csv(f'../02_INTERMEDIATE/ecoplanet_final_dataset.csv', index=False)

# ===== data_consolidation.ipynb: cell 94 (markdown) =====
# #### Acre Insights Dataset

# ===== data_consolidation.ipynb: cell 95 (code) =====
acre_psp_01 = gpd.read_file(acre_psp_01_path)
acre_psp_07 = gpd.read_file(acre_psp_07_path)
acre_psp_08 = gpd.read_file(acre_psp_08_path)

# ===== data_consolidation.ipynb: cell 96 (code) =====
acre_psp_01.head()

# ===== data_consolidation.ipynb: cell 97 (code) =====
len(acre_psp_08.columns)

# ===== data_consolidation.ipynb: cell 98 (code) =====
acre_psp_01.columns

# ===== data_consolidation.ipynb: cell 99 (code) =====
acre_psp_07.columns

# ===== data_consolidation.ipynb: cell 100 (code) =====
gdf_acre_drone = pd.concat([acre_psp_01, acre_psp_07, acre_psp_08], join='inner', ignore_index=True)

# ===== data_consolidation.ipynb: cell 101 (code) =====
gdf_acre_drone['centroid'] = gdf_acre_drone.geometry.centroid

# ===== data_consolidation.ipynb: cell 102 (code) =====
gdf_acre_drone

# ===== data_consolidation.ipynb: cell 103 (code) =====
# gdf_acre_drone.to_file('../02_INTERMEDIATE/acre_drone_gdf.gpkg', index=False)

# ===== data_consolidation.ipynb: cell 104 (markdown) =====
# #### Acre Ground Truth with Height

# ===== data_consolidation.ipynb: cell 105 (code) =====
# Enable KML driver in fiona
fiona.drvsupport.supported_drivers['KML'] = 'r'

# ===== data_consolidation.ipynb: cell 106 (code) =====
def parse_kml_to_gpkg(kml_path, output_gpkg_path, layer_name="kml_validation"):
    """
    Parses a KML file, extracts names, heights, and diameters (even if messy),
    and saves the cleaned data into a GeoPackage layer.
    """
    # Parse the XML structure
    tree = ET.parse(kml_path)
    root = tree.getroot()
    
    # KML namespaces can vary, this dynamic lookup finds the active namespace
    ns_match = re.match(r'{.*}', root.tag)
    ns = ns_match.group(0) if ns_match else ''
    
    records = []
    
    # Find all Placemark elements in the KML file
    for placemark in root.findall(f'.//{ns}Placemark'):
        # 1. Get the Point Name
        name_node = placemark.find(f'{ns}name')
        point_name = name_node.text if name_node is not None else 'Unnamed_Point'
        
        # 2. Get the Description text (where height/diameter might hide)
        desc_node = placemark.find(f'{ns}description')
        description = desc_node.text if desc_node is not None else ''
        
        # 3. Parse coordinates
        coord_node = placemark.find(f'.//{ns}coordinates')
        if coord_node is None or not coord_node.text:
            continue  # Skip features without physical spatial coordinates
            
        coords_str = coord_node.text.strip()
        # KML coordinates are structured as: longitude,latitude,elevation
        parts = coords_str.split(',')
        try:
            lon = float(parts[0])
            lat = float(parts[1])
            geom = Point(lon, lat)
        except (ValueError, IndexError):
            continue  # Skip corrupt geometries
            
        # 4. Extract metrics from description text using regex
        kml_height = None
        kml_diameter = None
        
        if description:
            # Clean HTML tags and convert to lowercase
            clean_desc = re.sub('<[^<]+?>', ' ', description).lower()
            
            # Extract height (looks for 'height 8' or 'height 8m' or 'height:8m')
            # Extracts only the numeric digits and decimals
            h_match = re.search(r'(?:height|h)\s*[:=]?\s*([0-9.]+)', clean_desc)
            if h_match:
                kml_height = float(h_match.group(1))
                
            # Extract canopy diameter (looks for 'canopy diam 6.4' or 'canopy diam: 6.4m')
            d_match = re.search(r'(?:canopy diam|diameter|dia|d|cabopy diam)\s*[:=]?\s*([0-9.]+)', clean_desc)
            if d_match:
                kml_diameter = float(d_match.group(1))
                
        # Append clean data dictionary
        records.append({
            'geometry': geom,
            'point_name': point_name,
            'height': kml_height,
            'canopy_diameter': kml_diameter
        })
        
    if not records:
        print("Warning: No spatial points were found or extracted from the KML.")
        return None
        
    # 5. Convert to GeoDataFrame (KML is natively WGS84)
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    
    # 6. Save directly to GeoPackage (GPKG driver is universally supported)
    gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")
    print(f"Successfully processed {len(gdf)} points into {output_gpkg_path}')")
    return gdf

# ===== data_consolidation.ipynb: cell 107 (code) =====
# --- Execution ---
gpkg_file = "../02_INTERMEDIATE/acre_height_validation.gpkg"  # Your output file

# Run the function
gdf_cleaned = parse_kml_to_gpkg(acre_ground_validation_path, gpkg_file)

# ===== data_consolidation.ipynb: cell 108 (code) =====
# Preview the data structure
print(gdf_cleaned.head())

# ===== data_consolidation.ipynb: cell 109 (markdown) =====
# #### Merging the two dfs to one source of truth

# ===== data_consolidation.ipynb: cell 110 (markdown) =====
# ##### Spatial join

# ===== data_consolidation.ipynb: cell 111 (markdown) =====
# #### 

# ===== data_consolidation.ipynb: cell 112 (code) =====
gdf_gt_ecoplanet = gdf_gt_ecoplanet.to_crs(gdf_acre_drone.crs)

# ===== data_consolidation.ipynb: cell 113 (code) =====
fig, ax = plt.subplots(figsize = (10, 8))
gdf_acre_drone.set_geometry('geometry').plot(ax=ax, facecolor='none', edgecolor='green', linewidth=2, label='Drone Polygon')

#plot drone centroids
gdf_acre_drone.set_geometry('centroid').plot(ax=ax, color='green', marker='x', markersize=100, label='Drone Centroid')

#plot ground truth points
gdf_gt_ecoplanet.plot(ax=ax, color='blue', marker='o', markersize=50, label='Ground Truth Point')

# ===== data_consolidation.ipynb: cell 114 (markdown) =====
# ##### Adding back a date of collection column to the dataset

# ===== data_consolidation.ipynb: cell 115 (code) =====
final = '../02_INTERMEDIATE/final_merged_bamboo_production.gpkg'
final_gdf = gpd.read_file(final)

# ===== data_consolidation.ipynb: cell 116 (code) =====
final_gdf.columns

# ===== data_consolidation.ipynb: cell 117 (code) =====
final_gdf.drop(columns=['year', 'acre_drone_fid', 'acre_drone_layer', 'acre_drone_path',
       'drone_match_status', 'val_point_name', 'validation_match_status'], inplace=True)

# ===== data_consolidation.ipynb: cell 118 (code) =====
final_gdf.head()

# ===== data_consolidation.ipynb: cell 119 (code) =====
final_gdf['ecoplanet_plot_id'].unique()

# ===== data_consolidation.ipynb: cell 120 (code) =====
date_mapping = {
    ('1a-APM-01', 2025): "2025-04-24",
    ('1a-APM-01', 2026): "2026-04-02",
    ('1a-APM-02', 2025): "2025-04-25",
    ('1a-APM-02', 2026): "2026-05-03",
    ('1a-APM-07', 2025): "2025-05-15",
    ('1a-APM-07', 2026): "2026-02-20",
    ('1a-APM-08', 2025): "2025-05-15",
    ('1a-APM-08', 2026): "2026-02-24",
    ('1a-APM-13', 2025): "2025-06-18",
    ('1a-APM-13', 2026): "2026-03-17",
    ('1a-GB-03', 2025): "2025-05-20",
    ('1a-GB-03', 2026): "2026-06-03",
    ('1a-GB-05', 2025): "2025-06-18",
    ('1a-GB-05', 2026): "2026-03-13"
}

# 2. Create a temporary lookup key from the existing GDF columns
lookup_key = zip(final_gdf["ecoplanet_plot_id"], final_gdf["ecoplanet_year"])

# 3. Map the dates to a new column and convert to datetime format
final_gdf["date_of_collection"] = [date_mapping.get(key, None) for key in lookup_key]
final_gdf["date_of_collection"] = gpd.pd.to_datetime(final_gdf["date_of_collection"])

# ===== data_consolidation.ipynb: cell 121 (code) =====
final_gdf.head()

# ===== data_consolidation.ipynb: cell 122 (markdown) =====
# #### Adding the June 2026 dataset for Ecoplanet to the final geodataframe

# ===== data_consolidation.ipynb: cell 123 (code) =====
psp_05_june_26 = '../01_RAW/1a-GB-5__June_24_2026.xlsx'
psp_05_june_26_df = pd.read_excel(psp_05_june_26)

# ===== data_consolidation.ipynb: cell 124 (code) =====
to_be_cleaned = [psp_05_june_26_df]

# ===== data_consolidation.ipynb: cell 125 (code) =====
def clean_biomass_df(df):
    """
    Cleans and structures biomass data from the provided excel sheet layout.
    Preserves clumps with no records by creating a single row placeholder 
    noting "No records found" instead of skipping them.
    """
    plot_id = None
    date_val = None
    
    # 1. Target exact cell coordinates for global Plot Metadata
    for r in range(min(3, df.shape[0])):
        row_values = [str(x).strip() for x in df.iloc[r].tolist()]
        for idx, val in enumerate(row_values):
            if '1a-' in val.lower() or 'apm' in val.lower():
                plot_id = val
            if 'DATE' in val.upper() and idx + 1 < len(row_values):
                date_val = row_values[idx + 1]

    # 2. Dynamically find all Clump start columns from Row index 2
    clump_indices = []
    if df.shape[0] > 2:
        row_2 = df.iloc[2].tolist()
        for idx, val in enumerate(row_2):
            if pd.notna(val) and 'CLUMP' in str(val).upper():
                clump_indices.append(idx)
                
    all_clumps = []
    num_cols = df.shape[1]
    
    # 3. Process each clump based on dynamic boundaries
    for i, start_col in enumerate(clump_indices):
        end_col = clump_indices[i + 1] if i + 1 < len(clump_indices) else num_cols
        
        # Slice the chunk for this specific clump
        clump_chunk = df.iloc[:, start_col:end_col]
        
        # Extract Clump ID
        clump_id = str(df.iloc[2, start_col]).strip()
        
        if df.shape[0] <= 5:
            continue
            
        # Extract data rows from row index 5 downward
        data_rows = clump_chunk.iloc[5:].copy()
        
        # Dynamic Column Naming using robust case-insensitive substring checks
        headers = [str(h).strip().upper() for h in clump_chunk.iloc[4].tolist()]
        col_names = []
        for idx, h in enumerate(headers):
            if 'NUMBER' in h:
                col_names.append('Culm_Number')
            elif 'DIAMETER (CM)' in h or 'DIAMETER_CM' in h:
                col_names.append('Culm_Diameter_cm')
            elif 'DIAMETER (MM)' in h or 'DIAMETER_MM' in h:
                col_names.append('Culm_Diameter_mm')
            elif 'AGB_CULM' in h or '(KG)' in h:
                col_names.append('AGB_Culm')
            else:
                col_names.append(f'Spacer_Col_{idx}')
                
        data_rows.columns = col_names
        
        # 4. Clean data rows and drop footer/totals
        # First, drop completely empty records in Culm_Number
        data_rows = data_rows.dropna(subset=['Culm_Number'])
        
        # Safe string evaluation to eliminate rows matching summary labels like 'AGB CLUMP'
        def is_summary_row(row):
            for val in row.values:
                if pd.notna(val) and 'AGB CLUMP' in str(val).upper():
                    return True
            return False
            
        summary_mask = data_rows.apply(is_summary_row, axis=1)
        data_rows = data_rows[~summary_mask]
        
        # Filter down to valid numeric indices to separate spacer values from numeric logs
        data_rows['Culm_Number_Numeric'] = pd.to_numeric(data_rows['Culm_Number'], errors='coerce')
        data_rows = data_rows.dropna(subset=['Culm_Number_Numeric']).drop(columns=['Culm_Number_Numeric'])
        
        # --- NEW LOGIC: Handle empty clumps gracefully ---
        if data_rows.empty:
            # Create a fallback single row dictionary filled with your preservation note
            fallback_row = {
                'Culm_Number': 'No records found',
                'Culm_Diameter_cm': 'No records found',
                'AGB_Culm': 'No records found'
            }
            # Fallback for mm tracking columns if present in sheet structures
            if 'Culm_Diameter_mm' in col_names:
                fallback_row['Culm_Diameter_mm'] = 'No records found'
                
            data_rows = pd.DataFrame([fallback_row])
        else:
            # Remove unwanted spacer columns dynamically if the data rows exist
            data_rows = data_rows.loc[:, ~data_rows.columns.str.startswith('Spacer_Col_')]
        
        # Inject metadata
        data_rows['Plot_ID'] = plot_id
        data_rows['Date'] = date_val
        data_rows['Clump_ID'] = clump_id
        
        all_clumps.append(data_rows)
        
    # 5. Concatenate all data chunks safely
    cols_order = ['Plot_ID', 'Date', 'Clump_ID', 'Culm_Number', 'Culm_Diameter_cm', 'AGB_Culm']
    
    if not all_clumps:
        return pd.DataFrame(columns=cols_order)
        
    cleaned_df = pd.concat(all_clumps, ignore_index=True)
    
    # Ensure all columns exist in final output schema even if empty
    for c in cols_order:
        if c not in cleaned_df.columns:
            cleaned_df[c] = None
            
    return cleaned_df[cols_order]

# ===== data_consolidation.ipynb: cell 126 (code) =====
master_list = []

for raw_df in to_be_cleaned:
    # Clean each already-loaded df using the function
    cleaned_data = clean_biomass_df(raw_df)
    
    # FIX 1: Only append if the dataframeA actually contains extracted rows
    if not cleaned_data.empty:
        master_list.append(cleaned_data)

# FIX 2: Prevent pd.concat crash if no valid data was found across all sheets
if master_list:
    final_dataset = pd.concat(master_list, ignore_index=True)
    
    # FIX 3: Safe datetime conversion with errors='coerce' to bypass messy strings
    final_dataset['Date'] = pd.to_datetime(final_dataset["Date"], errors='coerce', format="mixed")
    
    # Extract only the date component safely, ignoring nulls
    final_dataset['Year'] = final_dataset['Date'].dt.year
    final_dataset['Date'] = final_dataset["Date"].dt.date
else:
    # Fallback to an empty dataframe with correct schema if no data matched
    cols_order = ['Plot_ID', 'Date', 'Clump_ID', 'Culm_Number', 'Culm_Diameter_cm', 'AGB_Culm']
    final_dataset = pd.DataFrame(columns=cols_order)

# ===== data_consolidation.ipynb: cell 127 (code) =====
final_dataset.head()

# ===== data_consolidation.ipynb: cell 128 (code) =====
final_dataset.info()

# ===== data_consolidation.ipynb: cell 129 (code) =====
def clean_bamboo_2025_zeros(df):
    # 1. Subset the records for the year 2025
    df_2025 = df[df['Year'] == 2026].copy()
    
    # 2. Separate rows with real culms (> 0 diameter) from zero entries
    valid_culms = df_2025[df_2025['Culm_Diameter_cm'] > 0].copy()
    zero_culms = df_2025[df_2025['Culm_Diameter_cm'] == 0].copy()
    
    # 3. Handle clumps where every single record is 0
    # Find unique Plot+Clump combinations that have at least one valid culm
    valid_clump_keys = valid_culms[['Plot_ID', 'Clump_ID']].drop_duplicates()
    
    # Left join zero records with valid keys to flag clumps that have valid culms
    merged = zero_culms.merge(valid_clump_keys, on=['Plot_ID', 'Clump_ID'], how='left', indicator=True)
    
    # Filter for rows where the Plot+Clump combination has NO valid culms at all
    all_zero_clumps = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # Drop duplicates to keep exactly one record per all-zero clump
    preserved_clumps = all_zero_clumps.drop_duplicates(subset=['Plot_ID', 'Clump_ID']).copy()
    
    # Force the Culm_Number to 0 for these completely empty clumps
    preserved_clumps['Culm_Number'] = 0
    
    # 4. Recombine valid culms and preserved empty clumps
    cleaned_df = pd.concat([valid_culms, preserved_clumps]).sort_index()
    
    # 5. Recalculate and update the Culm_Number to reflect the true non-zero count
    # Count how many valid culms exist per unique Plot + Clump combination
    true_counts = valid_culms.groupby(['Plot_ID', 'Clump_ID']).size().reset_index(name='True_Count')
    
    # Map the true counts back to the cleaned dataframe
    cleaned_df = cleaned_df.merge(true_counts, on=['Plot_ID', 'Clump_ID'], how='left')
    cleaned_df['Culm_Number'] = cleaned_df['True_Count'].fillna(0).astype(int)
    cleaned_df = cleaned_df.drop(columns=['True_Count'])
    
    return cleaned_df

# ===== data_consolidation.ipynb: cell 130 (code) =====
final_no_zeros = clean_bamboo_2025_zeros(final_dataset)

# ===== data_consolidation.ipynb: cell 131 (code) =====
final_no_zeros.head()

# ===== data_consolidation.ipynb: cell 132 (code) =====
final_no_zeros.info()

# ===== data_consolidation.ipynb: cell 133 (code) =====
final_no_zeros.sample(10)

# ===== data_consolidation.ipynb: cell 134 (code) =====
final_no_zeros = final_no_zeros.astype({'Culm_Diameter_cm': float, 'AGB_Culm':float})

# ===== data_consolidation.ipynb: cell 135 (code) =====
final_no_zeros.isna().sum()

# ===== data_consolidation.ipynb: cell 136 (code) =====
final_no_zeros.describe()

# ===== data_consolidation.ipynb: cell 137 (code) =====
final_no_zeros.columns = final_no_zeros.columns.str.lower()

# ===== data_consolidation.ipynb: cell 138 (code) =====
final_no_zeros.columns

# ===== data_consolidation.ipynb: cell 139 (code) =====
# Strip whitespace and pad the final number with a leading zero if it is a single digit
final_no_zeros['plot_id'] = final_no_zeros['plot_id'].astype(str).str.replace(
    r'-(\d+)$', 
    lambda x: f"-{int(x.group(1)):02d}", 
    regex=True
)

# ===== data_consolidation.ipynb: cell 140 (code) =====
final_no_zeros['plot_id'].unique()

# ===== data_consolidation.ipynb: cell 141 (markdown) =====
# #### Is the dataframe complete

# ===== data_consolidation.ipynb: cell 142 (code) =====
# 1. Get all unique plots and years actually present in your data
unique_plots = final_no_zeros['plot_id'].unique()
unique_years = final_no_zeros['year'].unique()

# 2. Generate the 15 expected clump IDs
expected_clumps = [f"CLUMP #{i}" for i in range(1, 16)]

# 3. Build a perfect reference grid of all required combinations
expected_grid = pd.MultiIndex.from_product(
    [unique_plots, unique_years, expected_clumps],
    names=['plot_id', 'year', 'clump_id']
)

# ===== data_consolidation.ipynb: cell 143 (code) =====
# 4. Convert your current data's columns into a MultiIndex
actual_grid = pd.MultiIndex.from_frame(final_no_zeros[['plot_id', 'year', 'clump_id']])

# 5. Find the combinations that exist in the expected grid but not in your data
missing_combinations = expected_grid.difference(actual_grid)

# 6. Convert the result into a readable DataFrame
missing_df = missing_combinations.to_frame(index=False)

# ===== data_consolidation.ipynb: cell 144 (code) =====
missing_df

# ===== data_consolidation.ipynb: cell 145 (markdown) =====
# #### Aggregating the biomass values for 2026

# ===== data_consolidation.ipynb: cell 146 (code) =====
final_no_zeros['agb_culm'] = final_no_zeros['agb_culm'].round(2)
final_no_zeros['culm_diameter_cm'] = final_no_zeros['culm_diameter_cm'].round(2)

# ===== data_consolidation.ipynb: cell 147 (code) =====
df_clean = final_no_zeros.copy()

# --- STEP 1.5: Temporarily turn 0s to NaN for biomass calculations ---
# This ensures 0s don't skew min, mean, or median downward.
df_clean["agb_culm_nan"] = df_clean["agb_culm"].replace(0, np.nan)

# --- STEP 2: Aggregate the Data ---
aggregated_df = (
    df_clean.groupby(["plot_id", "date", "year", "clump_id"])
    .agg(
        # Calculate stats using the column where 0s are NaN (they will be ignored)
        max_culm_agb=("agb_culm_nan", "max"),
        max_culm_diameter_cm=("culm_diameter_cm", "max"),
        avg_culm_diameter_cm=("culm_diameter_cm", "mean"),
        median_culm_diameter_cm=("culm_diameter_cm", "median"),
        min_culm_diameter_cm=("culm_diameter_cm", "min"),
        avg_culm_agb=("agb_culm_nan", "mean"),
        min_culm_agb=("agb_culm_nan", "min"),
        median_culm_agb=("agb_culm_nan", "median"),
        total_agb=("agb_culm_nan", "sum"),
        # Culm count still looks at the original column to capture everything
        total_culms=("culm_number", "max"),
    )
    .reset_index()
)

# --- STEP 3: Handle completely empty clumps ---
# If a clump only had 0s, its stats are NaN. Turn them back to 0.0.
cols_to_zero = [
    "max_culm_agb",
    "avg_culm_agb",
    "min_culm_agb",
    "median_culm_agb",
    "total_agb"
]
aggregated_df[cols_to_zero] = aggregated_df[cols_to_zero].fillna(
    0.0
)

# ===== data_consolidation.ipynb: cell 148 (code) =====
aggregated_df.head()

# ===== data_consolidation.ipynb: cell 149 (code) =====
aggregated_df.to_csv('../02_INTERMEDIATE/psp_05_june_26.csv', index=False)

# ===== data_consolidation.ipynb: cell 150 (markdown) =====
# #### Final geodataframe joining to the psp 05 

# ===== data_consolidation.ipynb: cell 151 (code) =====
final_gdf.head()

# ===== data_consolidation.ipynb: cell 152 (code) =====
final_gdf.columns

# ===== data_consolidation.ipynb: cell 153 (code) =====
prefixes_to_drop = ['ecoplanet_', 'acre_']

# ===== data_consolidation.ipynb: cell 154 (code) =====
drone_prefix = ['drone_']

# ===== data_consolidation.ipynb: cell 155 (code) =====
final_gdf.columns = [
    next((col.removeprefix(p) for p in prefixes_to_drop if col.startswith(p)), col)
    for col in final_gdf.columns
]

# ===== data_consolidation.ipynb: cell 156 (code) =====
final_gdf.columns = [
    next((col.removeprefix(p) for p in drone_prefix if col.startswith(p)), col)
    for col in final_gdf.columns
]

# ===== data_consolidation.ipynb: cell 157 (code) =====
final_gdf.columns

# ===== data_consolidation.ipynb: cell 158 (code) =====
aggregated_df.columns

# ===== data_consolidation.ipynb: cell 159 (code) =====
aggregated_df.rename(columns={'date':'date_of_collection'}, inplace=True)

# ===== data_consolidation.ipynb: cell 160 (code) =====
aggregated_df['date_of_collection'] = pd.to_datetime(aggregated_df['date_of_collection'])

# ===== data_consolidation.ipynb: cell 161 (code) =====
aggregated_df.dtypes

# ===== data_consolidation.ipynb: cell 162 (code) =====
final_gdf.dtypes

# ===== data_consolidation.ipynb: cell 163 (code) =====
combined_gdf = pd.concat([final_gdf, aggregated_df], ignore_index=True)

# ===== data_consolidation.ipynb: cell 164 (code) =====
combined_gdf.sample(6)

# ===== data_consolidation.ipynb: cell 165 (code) =====
len(combined_gdf)

# ===== data_consolidation.ipynb: cell 166 (code) =====
combined_gdf.dtypes

# ===== data_consolidation.ipynb: cell 167 (code) =====
combined_gdf = combined_gdf.astype({'hmin':float, 'hmax': float, 'hmean':float, 'hstd':float, 'hmedian':float, 'total_points':float, 'canopy_volume_m3':float, 'gap_fraction':float, 'canopy_area': float, 'val_height':float,
                                    'val_canopy_diameter':float})

# ===== data_consolidation.ipynb: cell 168 (code) =====
combined_gdf["clump_num"] = (
    combined_gdf["clump_id"].str.extract(r"(\d+)").astype(int)
)

# Sort using the temporary numeric columns, but keep dates chronological
combined_gdf = combined_gdf.sort_values(by=["plot_id", "clump_num", "date_of_collection"])

# Drop the temporary sorting columns when done
combined_gdf = combined_gdf.drop(columns=["clump_num"])

# ===== data_consolidation.ipynb: cell 169 (code) =====
combined_gdf.head()

# ===== data_consolidation.ipynb: cell 170 (code) =====
combined_gdf.isna().sum()

# ===== data_consolidation.ipynb: cell 171 (code) =====
combined_gdf.columns

# ===== data_consolidation.ipynb: cell 172 (code) =====
combined_gdf.drop(columns=['lat', 'lon'], inplace=True)

# ===== data_consolidation.ipynb: cell 173 (code) =====
combined_gdf = combined_gdf[['plot_id', 'year', 'date_of_collection', 'clump_id', 'total_culms', 'total_agb',
       'max_culm_diameter_cm', 'avg_culm_diameter_cm', 'min_culm_diameter_cm',
       'median_culm_diameter_cm', 'max_culm_agb', 'avg_culm_agb',
       'min_culm_agb', 'median_culm_agb', 'hmin', 'hmax',
       'hmean', 'hstd', 'hmedian', 'total_points', 'canopy_volume_m3',
       'gap_fraction', 'canopy_area', 'val_height', 'val_canopy_diameter',
       'geometry']]

# ===== data_consolidation.ipynb: cell 174 (code) =====
# combined_gdf.to_csv('../02_INTERMEDIATE/combined_csv.csv', index=False)

# ===== data_consolidation.ipynb: cell 175 (code) =====
combined_clean = pd.read_csv('../02_INTERMEDIATE/combined_csv.csv')

# ===== data_consolidation.ipynb: cell 176 (code) =====
combined_clean.head()

# ===== data_consolidation.ipynb: cell 177 (code) =====
combined_clean.columns

# ===== data_consolidation.ipynb: cell 178 (code) =====
combined_clean.dtypes

# ===== data_consolidation.ipynb: cell 179 (code) =====
# 1. Find all float column names
float_cols = combined_clean.select_dtypes(include=['float']).columns

# 2. Round only those columns to 2 decimal places
combined_clean[float_cols] = combined_clean[float_cols].round(2)

# ===== data_consolidation.ipynb: cell 180 (code) =====
combined_clean.describe()

# ===== data_consolidation.ipynb: cell 181 (code) =====
from shapely import wkt
combined_clean['geometry'] = combined_clean['geometry'].apply(wkt.loads)

# ===== data_consolidation.ipynb: cell 182 (code) =====
combined_clean_gdf = gpd.GeoDataFrame(combined_clean,
                                      geometry='geometry',
                                      crs="EPSG:32735")

# ===== data_consolidation.ipynb: cell 183 (code) =====
combined_clean_gdf.columns

# ===== data_consolidation.ipynb: cell 184 (code) =====
combined_clean_gdf.rename(columns={'plot_id':'psp_id'}, inplace=True)

# ===== data_consolidation.ipynb: cell 185 (code) =====
combined_clean_gdf.to_file('../02_INTERMEDIATE/final_bamboo_parameters.gpkg', index=False)

# ===== data_consolidation.ipynb: cell 186 (code) =====
combined_clean_gdf.to_csv('../02_INTERMEDIATE/final_bamboo_parameters.csv', index=False)

# ===== data_consolidation.ipynb: cell 187 (code) =====
combined_clean_gdf[combined_clean_gdf['year'] == 2026].isna().sum()

# ===== data_consolidation.ipynb: cell 188 (code) =====
