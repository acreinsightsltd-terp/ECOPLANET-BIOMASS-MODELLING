#!/usr/bin/env python3
"""Cleaning Dataset. Clean and normalize the bamboo parameter GeoPackage."""
from pathlib import Path
import os
import geopandas as gpd
import pandas as pd


def main():
    """Run the data-wrangling workflow."""
    os.chdir(Path(__file__).resolve().parent)


    # 1. Load the GeoPackage
    # Replace 'your_data.gpkg' with your actual file path
    # If your GPKG has multiple layers, specify layer='layer_name'
    gpkg_path = "../02_INTERMEDIATE/final_bamboo_parameters_v01.gpkg"
    gdf = gpd.read_file(gpkg_path)



    gdf = gdf.drop(columns = ['canopy_diameter'])

    # 2. Define your columns explicitly based on your schema
    int_cols = ["year", "total_culms", "total_points"]

    # All your biometric, biomass (AGB), and canopy metrics
    float_cols = [
        "total_agb",
        "max_culm_diameter_cm",
        "avg_culm_diameter_cm",
        "min_culm_diameter_cm",
        "median_culm_diameter_cm",
        "max_culm_agb",
        "avg_culm_agb",
        "min_culm_agb",
        "median_culm_agb",
        "hmin",
        "hmax",
        "hmean",
        "hstd",
        "hmedian",
        "canopy_volume_m3",
        "gap_fraction",
        "canopy_area",
        "canopy_diameter",
        "val_height",
        "val_canopy_diameter",
    ]

    # 3. Clean and convert Integer columns
    for col in int_cols:
        if col in gdf.columns:
            # pd.to_numeric turns invalid strings/spaces into NaN, fillna(0) prevents crashing
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0).astype("int64")

    # 4. Clean, convert, and round Float columns to strictly 2 decimal places
    for col in float_cols:
        if col in gdf.columns:
            # Convert text to actual decimals
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
            # Round strictly to 2 decimal places
            gdf[col] = gdf[col].round(2)

    # 5. Save the properly typed GeoPackage
    output_path = "../02_INTERMEDIATE/final_bamboo_parameters_v01.gpkg"
    gdf.to_file(output_path, driver="GPKG")

    print("GeoPackage successfully cleaned and types corrected!")



if __name__ == "__main__":
    main()
