library(glue)
library(lidR)
library(terra)
library(sf)
library(future)
library(raster)
library(dplyr)
library(concaveman)

# --- 2. GLOBAL SETTINGS & PARALLELIZATION ---
setwd("E:/PROJECTS/20260525_FOR_ECOPLANET_BAMBOO_RWANDA")
raw_dir <- "01_RAW/OUTPUTS"
inter_dir <- "02_INTERMEDIATE/PSP-1A-APM-01"

# Enable parallel processing for lidR (uses 'future' package)
# This speeds up CHM generation and tree detection significantly
plan(multisession) 

# --- 3. EFFICIENT DATA LOADING ---
# Optimization: Use 'select' and 'filter' inside readLAS.
# This only loads necessary attributes (X, Y, Z, Intensity, etc.) into RAM.
las <- readLAS(glue("{inter_dir}/merged.las")) # Basic outlier cleaning at read-time

# Fix scan angle rank values by capping highs to 90
las@data$ScanAngleRank[las@data$ScanAngleRank > 90] <- 90L

# Fix scan angle rank values by capping lows to 90
las@data$ScanAngleRank[las@data$ScanAngleRank < -90] <- -90L
print(las)
las_check(las)
#removal of noise and erratic points
las <- classify_noise(las, sor(k = 15))
#plot(las)
# Keep only points that are NOT classified as noise
las <- filter_poi(las, Classification != 18)
writeLAS(las, glue("{inter_dir}/noise_filtered.las"))
# --- 4. GROUND & HEIGHT (Streamlined) ---
# Progress monitoring can be enabled with set_lidr_threads()
las <- classify_ground(las, algorithm = ptd(4, distance = 0.05))
writeLAS(las, glue("{inter_dir}/ground_classified.las"))
dtm <- rasterize_terrain(las, res = 0.2, algorithm = knnidw(k = 10, p = 2))
NAflag(dtm) <- -9999
writeRaster(dtm, file.path(inter_dir, "dtm.tif"), overwrite = TRUE)
plot(dtm)

# Normalization: Directly creates height-above-ground points
nlas <- normalize_height(las, dtm)
writeLAS(nlas, glue("{inter_dir}/normalized_las.las"))
#plot(nlas)
# --- 5. REFINED CHM & TREE DETECTION ---
# Pit-free is best for individual tree detection
chm <- rasterize_canopy(nlas, res = 0.2, algorithm = pitfree(subcircle = 0.1))
writeRaster(chm, file.path(inter_dir, "chm.tif"), overwrite = TRUE)
plot(chm)

# Smoothing CHM (Removes artifacts that cause false tree detections)
chm_s <- terra::focal(chm, w = matrix(1, 3, 3), fun = median, na.rm = TRUE)
writeRaster(chm_s, file.path(inter_dir, "chm_smoothed.tif"), overwrite = TRUE)
plot(chm_s)









#-----------------------Tree detection methodology using dalponte-------------------------------------
# Tree Detection
ttops <- locate_trees(nlas, lmf(ws = 6))
# --- 6. SEGMENTATION & METRICS ---
algo <- dalponte2016(chm_s, ttops, max_cr = 7, th_seed = 0.5)

las_seg <- segment_trees(nlas, algo)
plot(las_seg)
# Metrics: Standard forestry metrics + custom Intensity filter
# itot > 9 filters out low-intensity noise/ghost points
tree_table <- crown_metrics(las_seg, func = .stdtreemetrics)
tree_table <- tree_table[tree_table$Z > 1, ] 
coords <- st_coordinates(tree_table$geometry)
coords
tree_table
max(tree_table$convhull_area)
# Add them back to your dataframe as clean columns
tree_table <- tree_table %>%
  mutate(X = coords[,"X"],
         Y = coords[,"Y"],
         Z = coords[,"Z"])
tree_table = tree_table %>%
  select(X, Y, Z, treeID, convhull_area)
tree_table



# 1. Define the Custom Function
calculate_tree_stats <- function(x, y, z) {
  # Basic Metrics
  h_max <- max(z)
  
  # Calculate Spread (Horizontal distance)
  dx <- max(x) - min(x)
  dy <- max(y) - min(y)
  crown_diam <- (dx + dy) / 2
  
  # Return as a list
  return(list(
    Height = h_max,
    CrownDiam = crown_diam
  ))
}

# 2. Run crown_metrics
# Using 'convex' is safer for matching LiDAR360 areas
tree_table <- crown_metrics(las_seg, 
                            func = ~calculate_tree_stats(X, Y, Z), 
                            geom = "convex")

# 3. Extract Area, Volume, and Clean Coordinates
tree_table <- tree_table %>%
  mutate(
    CrownArea = as.numeric(st_area(geometry)),
    # Simple Volume: Area * Height (Standard approximation)
    CrownVol  = CrownArea * Height, 
    # Extract clean X and Y from the centroid
    X = st_coordinates(st_centroid(geometry))[,1],
    Y = st_coordinates(st_centroid(geometry))[,2]
  )

# 4. Filter for real trees
tree_table <- tree_table %>% filter(Height > 4)


# View results
tree_table <- st_drop_geometry(tree_table)
tree_table

# --- 7. EXPORT --
r_proj_path <- system.file("proj", package = "sf")[1]
# Set the environment variable for this session only
Sys.setenv(PROJ_LIB = r_proj_path)
writeRaster(chm, file.path(inter_dir, "chm.tif"), overwrite = TRUE)
writeLAS(nlas, "normalized_las.las")
write.csv(tree_table, "tree_inventory.csv", row.names = FALSE)
