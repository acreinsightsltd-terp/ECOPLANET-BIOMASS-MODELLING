# Bamboo dMRV Biomass Modelling

Drone LiDAR-based biomass modelling workflow for bamboo clumps.

## Project structure

| Directory | Purpose |
|---|---|
| `01_RAW` | Original field, LiDAR and SLAM data |
| `02_INTERMEDIATE` | Data generated during processing and modelling |
| `03_PROCESSED` | Processed outputs |
| `04_SCRIPTS` |Parameter extraction, data preparation, and validation |
| `05_ANALYSIS` | Exploratory data analysis and modelling |

## Workflow

Raw field and LiDAR data

        ↓

Data wrangling

        ↓

Bamboo parameter extraction

        ↓

Drone metric validation

        ↓

Exploratory analysis

        ↓

Biomass modelling

## Modelling

The response variable is clump-level above-ground biomass.

Models investigated include:

- Simple linear models
- Multiple linear models
- Polynomial models
- Power models
- Polynomial models with linear passthrough variables
- Weighted power models

Model validation uses grouped validation by PSP to evaluate performance
on unseen PSPs to ensure there is no target leakage that might falsely improve our accuracies due to spatial autocorrelation.

## Dataset versions

Dataset development is documented separately from model development.

- DV-001: Initial dataset
- DV-002: Dataset with breast-height features
- DV-003: Corrected dataset
- DV-004: June 2026 standardized dataset

Large raw and intermediate datasets are intentionally excluded from Git.
Selected modelling datasets are tracked where they are useful for
reproducibility and development history.
