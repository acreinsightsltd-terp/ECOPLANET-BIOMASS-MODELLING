library(glue)
library(lidR)
library(terra)

# =============================================================================
# STEP A: SETTINGS
# =============================================================================
# Run this after your existing pipeline (noise filtering -> ground classify ->
# DTM -> normalize -> CHM), using the same inter_dir per PSP.

setwd("E:/PROJECTS/20260525_FOR_ECOPLANET_BAMBOO_RWANDA")
inter_dir <- "02_INTERMEDIATE/PSP-1A-GB-03"

# Which heights to slice for the vertical-integration feature (stem "volume"
# proxy via Simpson's rule in Python). 1.3m (breast height) is deliberately
# included since it's also the primary horizontal-packing slice.
slice_heights   <- seq(0.3, 1.8, by = 0.3)   # 0.3, 0.6, 0.9, 1.2, 1.5, 1.8
half_thickness  <- 0.05                       # +/- 5cm band around each height
res_stack       <- 0.03                       # resolution for the multi-height stack
res_breast      <- 0.02                       # finer resolution for the dedicated 1.3m layer
breast_height   <- 1.3

# =============================================================================
# STEP B: LOAD NORMALIZED POINT CLOUD, DROP GROUND/NEAR-GROUND NOISE
# =============================================================================
# normalize_height() already put Z on a height-above-ground scale, so
# "ground points" here just means the residual near-zero noise floor, not a
# Classification filter (ground points were already stripped out for DTM
# purposes upstream; Z near 0 in the normalized cloud is what remains).

nlas <- readLAS(glue("{inter_dir}/normalized_las.las"))
nlas <- filter_poi(nlas, Z > 0.05)

if (npoints(nlas) == 0) {
  stop("No above-ground points remain after filtering -- check normalization.")
}

# =============================================================================
# STEP C: SLICE + RASTERIZE AT EACH HEIGHT -> MULTI-BAND DENSITY STACK
# =============================================================================
# rasterize_density() returns point density in points/m^2 per cell -- this is
# already the areal density Python needs; no extra Jacobian correction
# required on the Python side since it isn't operating on raw point counts.

density_layers <- list()

for (h in slice_heights) {
  z_lo <- h - half_thickness
  z_hi <- h + half_thickness
  
  slice_las <- filter_poi(nlas, Z >= z_lo & Z <= z_hi)
  
  if (npoints(slice_las) < 10) {
    message(glue("Skipping height {h}m: only {npoints(slice_las)} points in band."))
    next
  }
  
  dens <- rasterize_density(slice_las, res = res_stack)
  # rasterize_density can return a 2-layer object (point_density, pulse_density)
  # if pulseID is present; keep the point density layer specifically.
  dens_layer <- if ("point_density" %in% names(dens)) dens[["point_density"]] else dens[[1]]
  
  layer_name <- glue("dens_{sprintf('%.2f', h)}m")
  names(dens_layer) <- layer_name
  density_layers[[layer_name]] <- dens_layer
}

if (length(density_layers) == 0) {
  stop("No height slice produced enough points to rasterize -- check slice_heights / half_thickness.")
}

# Layers may not perfectly align cell-for-cell (different point subsets can
# yield slightly different extents) -- resample everything onto the first
# layer's grid before stacking so Python gets one consistent grid across bands.
ref_grid <- density_layers[[1]]
density_layers <- lapply(density_layers, function(r) {
  if (!compareGeom(r, ref_grid, stopOnError = FALSE)) {
    resample(r, ref_grid, method = "near")
  } else {
    r
  }
})

density_stack <- rast(density_layers)
# Explicit NA (not 0) for cells outside the actual scan extent -- 0 is a real,
# meaningful "beam reached here, found nothing" observation and must stay
# distinct from "no data was ever collected here".
writeRaster(density_stack, file.path(inter_dir, "height_slice_density_stack.tif"),
            overwrite = TRUE, NAflag = -9999)

# =============================================================================
# STEP D: DEDICATED BREAST-HEIGHT (1.3M) LAYER AT FINER RESOLUTION
# =============================================================================
# This is the primary horizontal-packing slice -- kept at a finer resolution
# than the multi-height stack since it carries the most weight in the model.

bh_las <- filter_poi(nlas, Z >= breast_height - half_thickness & Z <= breast_height + half_thickness)

if (npoints(bh_las) < 10) {
  warning(glue("Only {npoints(bh_las)} points at breast height -- check point density / thickness."))
}
#canopy height model at breast height
chm <- rasterize_canopy(bh_las, res = 0.2, algorithm = pitfree(subcircle = 0.1))
writeRaster(chm, file.path(inter_dir, "chm_breast_height.tif"), overwrite = TRUE)
plot(chm)

bh_dens <- rasterize_density(bh_las, res = res_breast)
bh_dens_layer <- if ("point_density" %in% names(bh_dens)) bh_dens[["point_density"]] else bh_dens[[1]]
names(bh_dens_layer) <- "dens_1.30m"

writeRaster(bh_dens_layer, file.path(inter_dir, "breast_height_density_1.3m.tif"),
            overwrite = TRUE, NAflag = -9999)

cat(glue("Saved {length(density_layers)}-band height-slice stack and dedicated breast-height layer to {inter_dir}\n"))
plot(density_stack)
plot(bh_dens_layer, main = "Breast-height (1.3m) point density")
