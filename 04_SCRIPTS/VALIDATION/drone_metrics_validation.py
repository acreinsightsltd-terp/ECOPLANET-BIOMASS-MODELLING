#!/usr/bin/env python3
"""Drone metrics validation notebook exported as a Python script.

Run with Python from any working directory. Relative data paths resolve
from the VALIDATION script directory, matching the notebook location.
"""
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# ===== drone_metrics_validation.ipynb: cell 0 (markdown) =====
# ### VALIDATION OF BAMBOO DRONE LIDAR OUTPUTS
# ### AUTHOR: ELKANA KIPRUTO
# #### STRATEGY: 
# * Using the ground truth data collected by Acre in the field, we validate the same same outputs we got from drone modelling. This is using various validation metrics that capture trend, exact errors, and other metrics
# #### HOW SUCCESS LOOKS LIKE:
# * The drone laser scanner captures the same patterns a human looking at the bamboo would
# * The drone laser scanner captures metrics similarly as well eg height

# ===== drone_metrics_validation.ipynb: cell 1 (code) =====
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ===== drone_metrics_validation.ipynb: cell 2 (markdown) =====
# ##### ------------------------------------------------
# ##### STEP 1: Read and understand the dataset
# ##### ------------------------------------------------

# ===== drone_metrics_validation.ipynb: cell 3 (code) =====
bamboo_params_path = '../../02_INTERMEDIATE/final_bamboo_parameters_v02.gpkg'
if os.path.exists(bamboo_params_path):
    gdf = gpd.read_file(bamboo_params_path)

# ===== drone_metrics_validation.ipynb: cell 4 (code) =====
gdf.head()

# ===== drone_metrics_validation.ipynb: cell 5 (code) =====
gdf.columns

# ===== drone_metrics_validation.ipynb: cell 6 (code) =====
# 2. Define the exact numeric columns from your head to convert
cols_to_float = [
    "total_agb",
    'total_points',
    'total_culms',
    "max_culm_diameter_cm",
    "avg_culm_diameter_cm",
    "min_culm_diameter_cm",
    "median_culm_diameter_cm",
    'hmin',
    'hmean',
    "hstd",
    'hmax',
    'max_culm_agb',          
    'avg_culm_agb',                   
    'min_culm_agb',               
    'median_culm_agb',
    "hmedian",
    "canopy_volume_m3",
    "gap_fraction",
    "canopy_area",
    "canopy_diameter",
    "val_height",
    "val_canopy_diameter",
]

# 3. Convert all specified columns to float, forcing errors to NaN if text exists
gdf[cols_to_float] = gdf[cols_to_float].apply(pd.to_numeric, errors="coerce")

# 4. Round the selected columns to 2 decimal places
gdf[cols_to_float] = gdf[cols_to_float].round(2)

# Verify the final types and values
gdf[cols_to_float].head()

# ===== drone_metrics_validation.ipynb: cell 7 (code) =====
gdf.dtypes

# ===== drone_metrics_validation.ipynb: cell 8 (code) =====
gdf.isna().sum()

# ===== drone_metrics_validation.ipynb: cell 9 (code) =====
gdf.columns

# ===== drone_metrics_validation.ipynb: cell 10 (code) =====
gdf.dtypes

# ===== drone_metrics_validation.ipynb: cell 11 (code) =====
gdf.info()

# ===== drone_metrics_validation.ipynb: cell 12 (code) =====
gdf.describe()

# ===== drone_metrics_validation.ipynb: cell 13 (markdown) =====
# #### -----------------------------------------
# #### STEP 2: Statistical Validation
# #### ------------------------------------------

# ===== drone_metrics_validation.ipynb: cell 14 (markdown) =====
# #####  1. Height Validation

# ===== drone_metrics_validation.ipynb: cell 15 (code) =====
gdf_26 = gdf[gdf['year'] == '2026'].copy()

# ===== drone_metrics_validation.ipynb: cell 16 (code) =====
len(gdf_26)

# ===== drone_metrics_validation.ipynb: cell 17 (code) =====
height_df = gdf_26.dropna(subset=['val_height', 'hmin', 'hmax', 'hmean', 'hstd', 'hmedian']).copy()

# ===== drone_metrics_validation.ipynb: cell 18 (code) =====
height_df.head()

# ===== drone_metrics_validation.ipynb: cell 19 (code) =====
len(height_df)

# ===== drone_metrics_validation.ipynb: cell 20 (code) =====
height_df[['psp', 'clump_id','val_height', 'hmax']]

# ===== drone_metrics_validation.ipynb: cell 21 (code) =====
#field values
height_true = height_df['val_height'].values
#modelled values
height_pred = height_df['hmax'].values

# ===== drone_metrics_validation.ipynb: cell 22 (code) =====
n = len(height_df)

# ===== drone_metrics_validation.ipynb: cell 23 (code) =====
r2 = r2_score(height_true, height_pred)
r = np.corrcoef(height_true, height_pred)[0, 1]
rmse = np.sqrt(mean_squared_error(height_true, height_pred))
mean_true = np.mean(height_true)
rrmse = (rmse / mean_true) * 100 if mean_true != 0 else np.nan
mae = mean_absolute_error(height_true, height_pred)
mbe = np.mean(height_pred - height_true)  # Bias

# Store metrics in a dictionary
metrics = {
    "Sample Size (n)": n,
    "R-squared (R²)": r2,
    "Pearson Correlation (r)": r,
    "RMSE": rmse,
    "Relative RMSE (rRMSE %)": rrmse,
    "MAE": mae,
    "Mean Bias Error (MBE)": mbe
}

# ===== drone_metrics_validation.ipynb: cell 24 (code) =====
# Print Summary Table
print(f"\n=== VALIDATION METRICS FOR: HEIGHT ===")
for metric, val in metrics.items():
    print(f"{metric}: {val:.4f}")

# ===== drone_metrics_validation.ipynb: cell 25 (markdown) =====
# ##### Note: 
# * Height validation metrics show evidence that the drone metrics capture the trends almost as well as someone on the field would
#
# * VALIDATION METRICS FOR: HEIGHT ===
# * Sample Size (n): 33.0000
# * R-squared (R²): 0.9239
# * Pearson Correlation (r): 0.9634
# * RMSE: 0.3945
# * Relative RMSE (rRMSE %): 6.7636
# * MAE: 0.2597
# * Mean Bias Error (MBE): 0.0409

# ===== drone_metrics_validation.ipynb: cell 26 (markdown) =====
# #### 2. Canopy Diameter

# ===== drone_metrics_validation.ipynb: cell 27 (code) =====
canopy_diam_df = gdf_26.dropna(subset=['canopy_diameter', 'val_canopy_diameter']).copy()

# ===== drone_metrics_validation.ipynb: cell 28 (code) =====
canopy_diam_df.head()

# ===== drone_metrics_validation.ipynb: cell 29 (code) =====
canopy_diam_df['date_of_collection'].unique()

# ===== drone_metrics_validation.ipynb: cell 30 (code) =====
canopy_diam_df = canopy_diam_df[canopy_diam_df['date_of_collection'] == '6/24/2026']

# ===== drone_metrics_validation.ipynb: cell 31 (code) =====
canopy_area_true = canopy_diam_df['val_canopy_diameter'].values
canopy_area_pred = canopy_diam_df['canopy_diameter'].values

# ===== drone_metrics_validation.ipynb: cell 32 (code) =====
n = len(canopy_diam_df)

# ===== drone_metrics_validation.ipynb: cell 33 (code) =====
len(canopy_diam_df)

# ===== drone_metrics_validation.ipynb: cell 34 (code) =====
r2 = r2_score(canopy_area_true, canopy_area_pred)
r = np.corrcoef(canopy_area_true, canopy_area_pred)[0, 1]
rmse = np.sqrt(mean_squared_error(canopy_area_true, canopy_area_pred))
mean_true = np.mean(canopy_area_true)
rrmse = (rmse / mean_true) * 100 if mean_true != 0 else np.nan
mae = mean_absolute_error(canopy_area_true, canopy_area_pred)
mbe = np.mean(canopy_area_pred - canopy_area_true)  # Bias

# Store metrics in a dictionary
metrics = {
    "Sample Size (n)": n,
    "R-squared (R²)": r2,
    "Pearson Correlation (r)": r,
    "RMSE": rmse,
    "Relative RMSE (rRMSE %)": rrmse,
    "MAE": mae,
    "Mean Bias Error (MBE)": mbe
}

# ===== drone_metrics_validation.ipynb: cell 35 (code) =====
# Print Summary Table
print(f"\n=== VALIDATION METRICS FOR: CANOPY DIAMETER ===")
for metric, val in metrics.items():
    print(f"{metric}: {val:.4f}")

# ===== drone_metrics_validation.ipynb: cell 36 (markdown) =====
# #### Note: 
# * Validation metrics show evidence that the drone outputs capture field trends and patterns almost as well as someone in teh field would
#
# * === VALIDATION METRICS FOR: CANOPY DIAMETER ===
# * Sample Size (n): 14.0000
# * R-squared (R²): 0.9621
# * Pearson Correlation (r): 0.9811
# * RMSE: 0.2393
# * Relative RMSE (rRMSE %): 4.1982
# * MAE: 0.1850
# * Mean Bias Error (MBE): -0.0236

# ===== drone_metrics_validation.ipynb: cell 37 (markdown) =====
# #### Visualization

# ===== drone_metrics_validation.ipynb: cell 38 (code) =====
# -----------------------------
# Residual Calculation
# -----------------------------

# Height residuals
gdf_26["height_residual"] = gdf_26["hmax"] - gdf_26["val_height"]

# Canopy area residuals
gdf_26["area_residual"] = gdf_26["canopy_area"] - gdf_26["canopy_area_gt"]

# -----------------------------
# HEIGHT VALIDATION
# -----------------------------

fig, ax = plt.subplots(figsize=(7,6))

ax.scatter(gdf_26["val_height"], gdf_26["hmax"], alpha=0.7)

# 1:1 Line
min_val = min(gdf_26["val_height"].min(), gdf_26["hmax"].min())
max_val = max(gdf_26["val_height"].max(), gdf_26["hmax"].max())

ax.plot([min_val, max_val],
        [min_val, max_val],
        'r--',
        linewidth=2,
        label="1:1 Line")

ax.set_xlabel("Field Height (m)")
ax.set_ylabel("Estimated Height (m)")
ax.set_title("Height Validation")
ax.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# HEIGHT RESIDUALS
# -----------------------------

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(gdf_26["val_height"], gdf_26["height_residual"], alpha=0.7)

ax.axhline(0,
           color='red',
           linestyle='--')

ax.set_xlabel("Field Height (m)")
ax.set_ylabel("Residual (Estimated - Field)")
ax.set_title("Height Residual Plot")

plt.tight_layout()
plt.show()

# -----------------------------
# HEIGHT RESIDUAL HISTOGRAM
# -----------------------------

fig, ax = plt.subplots(figsize=(7,5))

ax.hist(gdf_26["height_residual"],
        bins=12)

ax.axvline(0,
           color='red',
           linestyle='--')

ax.set_xlabel("Residual (m)")
ax.set_ylabel("Frequency")
ax.set_title("Height Residual Distribution")

plt.tight_layout()
plt.show()

# -----------------------------
# CANOPY DIAMETER VALIDATION
# -----------------------------

# Residuals
canopy_diam_df["diameter_residual"] = (
    canopy_diam_df["canopy_diameter"]
    - canopy_diam_df["val_canopy_diameter"]
)

# -----------------------------
# CANOPY DIAMETER VALIDATION
# -----------------------------

fig, ax = plt.subplots(figsize=(7,6))

ax.scatter(
    canopy_diam_df["val_canopy_diameter"],
    canopy_diam_df["canopy_diameter"],
    alpha=0.7
)

min_val = min(
    canopy_diam_df["val_canopy_diameter"].min(),
    canopy_diam_df["canopy_diameter"].min()
)

max_val = max(
    canopy_diam_df["val_canopy_diameter"].max(),
    canopy_diam_df["canopy_diameter"].max()
)

ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    'r--',
    linewidth=2,
    label="1:1 Line"
)

ax.set_xlabel("Field Canopy Diameter (m)")
ax.set_ylabel("Drone Canopy Diameter (m)")
ax.set_title("Canopy Diameter Validation")
ax.legend()

plt.tight_layout()
plt.show()


# -----------------------------
# DIAMETER RESIDUALS
# -----------------------------

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(
    canopy_diam_df["val_canopy_diameter"],
    canopy_diam_df["diameter_residual"],
    alpha=0.7
)

ax.axhline(
    0,
    color='red',
    linestyle='--'
)

ax.set_xlabel("Field Canopy Diameter (m)")
ax.set_ylabel("Residual (Drone - Field)")
ax.set_title("Canopy Diameter Residual Plot")

plt.tight_layout()
plt.show()


# -----------------------------
# DIAMETER RESIDUAL HISTOGRAM
# -----------------------------

fig, ax = plt.subplots(figsize=(7,5))

ax.hist(
    canopy_diam_df["diameter_residual"],
    bins=10
)

ax.axvline(
    0,
    color='red',
    linestyle='--'
)

ax.set_xlabel("Residual (m)")
ax.set_ylabel("Frequency")
ax.set_title("Canopy Diameter Residual Distribution")

plt.tight_layout()
plt.show()

# ===== drone_metrics_validation.ipynb: cell 39 (code) =====
fig, ax = plt.subplots(figsize=(8,5))

gdf_26.boxplot(column="height_residual", by="psp", ax=ax)

ax.set_xlabel("PSP")
ax.set_ylabel("Residual (m)")
ax.set_title("Height Residuals by PSP")

plt.suptitle("")
plt.tight_layout()
plt.show()
