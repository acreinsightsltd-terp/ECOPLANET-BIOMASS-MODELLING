import numpy as np
import rasterio

from rasterio.mask import mask as rio_mask
from rasterio.transform import xy

from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, MultiPoint

from scipy.spatial import Delaunay, cKDTree
from scipy import ndimage

from sklearn.cluster import DBSCAN


# ============================================================================
# 2. ISOLATED CLUSTER REMOVAL
# ============================================================================

def remove_isolated_clusters(
    points,
    eps=None,
    min_samples=4,
    max_gap_multiplier=3.0,
):
    """
    Remove spatially detached point clusters.

    The largest DBSCAN cluster is treated as the main clump body.

    Secondary clusters are retained only when their closest point
    is sufficiently close to the main body.
    """

    n = len(points)

    if n < min_samples:

        return (
            np.zeros(
                n,
                dtype=bool,
            ),
            eps,
        )

    # --------------------------------------------------------
    # Automatically estimate DBSCAN epsilon
    # --------------------------------------------------------

    if eps is None:

        tree = cKDTree(points)

        k_for_eps = min(
            min_samples,
            n - 1,
        )

        k_dist, _ = tree.query(
            points,
            k=k_for_eps + 1,
        )

        kth_nn_dist = (
            k_dist[:, -1]
        )

        positive_distances = (
            kth_nn_dist[
                kth_nn_dist > 0
            ]
        )

        if len(
            positive_distances
        ):

            eps = float(
                np.percentile(
                    positive_distances,
                    95,
                )
            )

        else:

            eps = 1e-6

    # --------------------------------------------------------
    # DBSCAN
    # --------------------------------------------------------

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    ).fit_predict(points)

    real_clusters, counts = (
        np.unique(
            labels[
                labels >= 0
            ],
            return_counts=True,
        )
    )

    if len(real_clusters) == 0:

        return (
            np.zeros(
                n,
                dtype=bool,
            ),
            eps,
        )

    # Largest cluster = main body.
    main_label = (
        real_clusters[
            np.argmax(counts)
        ]
    )

    main_points = points[
        labels == main_label
    ]

    main_tree = cKDTree(
        main_points
    )

    keep_mask = (
        labels == main_label
    )

    max_gap = (
        eps * max_gap_multiplier
    )

    # --------------------------------------------------------
    # Test secondary clusters
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Test ALL non-main labels (including -1 noise points!)
    # --------------------------------------------------------
    # Get every unique label present in the data except the main one
    all_other_labels = np.unique(labels[labels != main_label])

    for label in all_other_labels:
        cluster_mask = (labels == label)
        cluster_points = points[cluster_mask]

        # Calculate distance from these points to the main body
        gap_distances, _ = main_tree.query(cluster_points, k=1)

        # If even a single point in this cluster/noise set is close enough, keep them
        if gap_distances.min() <= max_gap:
            keep_mask |= cluster_mask

    return keep_mask, eps

    return (
        keep_mask,
        eps,
    )


# ============================================================================
# 3. COMBINED POINT-CLOUD CLEANING
# ============================================================================

def clean_point_cloud(
    points,
    dbscan_min_samples=4,
    max_gap_multiplier=2.0,
    verbose=False,
):
    """
    Clean the positive raster-cell point cloud.

    Pipeline:

        RAW POINTS
            ↓
        Statistical Outlier Removal
            ↓
        DBSCAN isolated-cluster removal
            ↓
        CLEAN POINTS
    """

    n0 = len(points)

    # --------------------------------------------------------
    # Too few points
    # --------------------------------------------------------

    if n0 < 4:

        return (
            points,
            {
                "n_input": n0,
                "n_after_dbscan": n0,
                "n_removed": 0,
                "eps_used": None,
            },
        )

    # --------------------------------------------------------
    # DBSCAN
    # --------------------------------------------------------

    dbscan_mask, eps_used = (
        remove_isolated_clusters(
            points,
            eps=None,
            min_samples=dbscan_min_samples,
            max_gap_multiplier=max_gap_multiplier,
        )
    )

    points_clean = (
        points[dbscan_mask]
    )

    n2 = len(points_clean)

    stats = {
        "n_input": n0,
        "n_after_dbscan": n2,
        "n_removed": n0 - n2,
        "eps_used": eps_used,
    }

    if verbose:

        percentage_removed = (
            100.0
            * stats["n_removed"]
            / n0
            if n0 > 0
            else 0.0
        )

        print(
            f"Point cloud cleaning: "
            f"{n0} -> {n2}; "
            f"removed {stats['n_removed']} "
            f"({percentage_removed:.1f}%)"
        )

    return (
        points_clean,
        stats,
    )


# ============================================================================
# 4. ALPHA SHAPE
# ============================================================================

def alpha_shape(points, alpha=0.15):
    """
    Build a tight point-based concave hull.
    Criterion: circumradius <= 1 / alpha
    """
    if len(points) < 3:
        return None

    try:
        tri = Delaunay(points)
    except Exception:
        return None

    triangles = []
    max_radius = 1.0 / alpha

    for simplex in tri.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]

        # Side lengths
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)

        # Semi-perimeter for Heron's Formula
        s = (a + b + c) / 2.0
        
        # Numerical stability check to ensure we don't sqrt a negative near-zero float
        area_sq = s * (s - a) * (s - b) * (s - c)
        
        if area_sq <= 1e-9:
            continue
            
        triangle_area = np.sqrt(area_sq)

        # Calculate circumradius safely
        circumradius = (a * b * c) / (4.0 * triangle_area)

        if circumradius <= max_radius:
            triangles.append(Polygon([tuple(p1), tuple(p2), tuple(p3)]))

    if not triangles:
        return None

    geometry = unary_union(triangles)

    if geometry is None or geometry.is_empty:
        return None

    return geometry


# ============================================================================
# 5. RASTER -> OCCUPANCY GRID
# ============================================================================

def create_occupancy_grid(
    raster_path,
    polygon,
    band_index=4,
):
    """
    Read the breast-height density GeoTIFF and convert it into occupancy.

    Returns
    -------
    occupancy : np.ndarray
        -1 = nodata/outside
         0 = valid zero-density
         1 = positive-density

    transform : Affine
        Cropped raster transform.
    """

    # IMPORTANT:
    # raster_path MUST remain the path.
    # Do NOT read the raster here and then overwrite raster_path.

    with rasterio.open(
        raster_path
    ) as src:

        out_image, out_transform = (
            rio_mask(
                src,
                [polygon],
                crop=True,
                indexes=band_index,
                filled=False,
            )
        )

        # rasterio.mask.mask returns:
        #
        #   2D array when one band is requested in some versions
        #   3D array in others
        #
        if out_image.ndim == 3:

            arr = out_image[0]

        else:

            arr = out_image

        # ----------------------------------------------------
        # Mask handling
        # ----------------------------------------------------

        if np.ma.isMaskedArray(arr):

            values = np.asarray(
                arr.data,
                dtype=float,
            )

            masked = np.ma.getmaskarray(
                arr
            )

        else:

            values = np.asarray(
                arr,
                dtype=float,
            )

            masked = (
                ~np.isfinite(
                    values
                )
            )

        # ----------------------------------------------------
        # Occupancy
        # ----------------------------------------------------

        occupancy = np.full(
            values.shape,
            -1,
            dtype=np.int8,
        )

        valid = (
            ~masked
            &
            np.isfinite(values)
        )

        occupancy[
            valid
            & (values <= 0)
        ] = 0

        occupancy[
            valid
            & (values > 0)
        ] = 1

    return (
        occupancy,
        out_transform,
    )


# ============================================================================
# 6. POSITIVE RASTER CELLS -> XY POINTS
# ============================================================================

def get_positive_points(
    occupancy,
    transform,
):
    """
    Convert positive raster cells to their cell-centre XY coordinates.

    One point per positive raster cell.
    """

    rows, cols = np.where(
        occupancy == 1
    )

    if len(rows) < 3:

        return np.empty(
            (0, 2),
            dtype=float,
        )

    xs, ys = xy(
        transform,
        rows,
        cols,
    )

    points = np.column_stack(
        [
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
        ]
    )

    # Remove accidental duplicates.
    points = np.unique(
        points,
        axis=0,
    )

    return points


# ============================================================================
# 7. POINTS -> OUTER GEOMETRY
# ============================================================================

def _points_to_outer_geometry(
    points,
    transform,
    original_polygon,
    alpha,
):
    """
    Build the outer geometry from the supplied point array.

    This function receives CLEANED points directly.

    No raster re-reading occurs here.
    """

    if len(points) < 3:
        return None

    # --------------------------------------------------------
    # Check coordinates
    # --------------------------------------------------------

    points = np.asarray(
        points,
        dtype=float,
    )

    if points.ndim != 2:
        return None

    if points.shape[1] != 2:
        return None

    if not np.all(
        np.isfinite(points)
    ):
        return None

    # --------------------------------------------------------
    # Alpha shape
    # --------------------------------------------------------

    outer = alpha_shape(
        points,
        alpha=alpha,
    )

    if (
        outer is None
        or outer.is_empty
    ):
        return None

    # --------------------------------------------------------
    # Clip to original clump polygon
    # --------------------------------------------------------

    try:

        outer = (
            outer
            .intersection(
                original_polygon
            )
        )

    except Exception:

        return None

    if (
        outer is None
        or outer.is_empty
    ):
        return None

    # --------------------------------------------------------
    # MultiPolygon handling
    # --------------------------------------------------------

    if (
        outer.geom_type
        == "MultiPolygon"
    ):

        valid_polygons = []

        for poly in outer.geoms:

            if any(
                poly.covers(
                    Point(x, y)
                )
                for x, y in points
            ):

                valid_polygons.append(
                    poly
                )

        if not valid_polygons:
            return None

        outer = unary_union(
            valid_polygons
        )

    # --------------------------------------------------------
    # Repair topology
    # --------------------------------------------------------

    try:

        outer = outer.buffer(0)

    except Exception:

        pass

    if (
        outer is None
        or outer.is_empty
    ):
        return None

    return outer


# ============================================================================
# 8. PUBLIC OUTER GEOMETRY FUNCTION
# ============================================================================

def build_outer_geometry(
    occupancy,
    transform,
    original_polygon,
    alpha=0.15,
):
    """
    Public function for callers that already have occupancy.

    Extracts RAW positive cells and builds an outer geometry.
    """

    points = get_positive_points(
        occupancy,
        transform,
    )

    if len(points) < 3:
        return None

    return _points_to_outer_geometry(
        points=points,
        transform=transform,
        original_polygon=original_polygon,
        alpha=alpha,
    )


# ============================================================================
# 9. HOLLOW DETECTION
# ============================================================================

def find_hollow_geometry(
    occupancy,
    transform,
    outer_geometry,
    original_polygon,
    min_hollow_cells=4,
):
    """
    Detect internal zero-density regions.

    IMPORTANT:
    This deliberately uses the ORIGINAL occupancy grid.

    Cleaning is NOT allowed to modify occupancy because a removed
    noisy positive point must not become an artificial hollow.
    """

    if outer_geometry is None:
        return None

    zero = (
        occupancy == 0
    )

    if not zero.any():
        return None

    labelled, n_components = (
        ndimage.label(
            zero,
            structure=np.ones(
                (3, 3)
            ),
        )
    )

    hollow_polygons = []

    rows, cols = (
        zero.shape
    )

    for component_id in range(
        1,
        n_components + 1,
    ):

        component = (
            labelled
            == component_id
        )

        if (
            component.sum()
            < min_hollow_cells
        ):
            continue

        component_rows, component_cols = (
            np.where(component)
        )

        # ----------------------------------------------------
        # Ignore regions touching cropped raster boundary.
        #
        # Such regions are open to the outside and therefore
        # are not true internal hollows.
        # ----------------------------------------------------

        touches_boundary = (
            (component_rows == 0).any()
            or
            (
                component_rows
                == rows - 1
            ).any()
            or
            (component_cols == 0).any()
            or
            (
                component_cols
                == cols - 1
            ).any()
        )

        if touches_boundary:
            continue

        pts = []

        for r, c in zip(
            component_rows,
            component_cols,
        ):

            x, y = xy(
                transform,
                r,
                c,
            )

            pts.append(
                Point(
                    float(x),
                    float(y),
                )
            )

        if len(pts) < 3:
            continue

        try:

            hull = (
                MultiPoint(
                    pts
                ).convex_hull
            )

        except Exception:

            continue

        if hull.is_empty:
            continue

        try:

            hollow = (
                hull
                .intersection(
                    outer_geometry
                )
                .intersection(
                    original_polygon
                )
            )

        except Exception:

            continue

        if (
            not hollow.is_empty
            and hollow.area > 0
        ):

            hollow_polygons.append(
                hollow
            )

    if not hollow_polygons:
        return None

    hollow_geometry = (
        unary_union(
            hollow_polygons
        )
    )

    try:

        hollow_geometry = (
            hollow_geometry
            .intersection(
                outer_geometry
            )
            .intersection(
                original_polygon
            )
        )

    except Exception:

        return None

    if (
        hollow_geometry is None
        or hollow_geometry.is_empty
    ):
        return None

    return hollow_geometry


# ============================================================================
# 10. EQUIVALENT DIAMETER
# ============================================================================

def equivalent_diameter(
    area,
):
    if (
        area is None
        or area <= 0
    ):
        return np.nan

    return float(
        2.0
        * np.sqrt(
            area / np.pi
        )
    )


# ============================================================================
# 11. GEOMETRIC FEATURES
# ============================================================================

def geometry_features(
    outer_geometry,
    hollow_geometry,
    original_polygon,
):
    """
    Calculate geometry-derived features.
    """

    features = {

        "bh_outer_area_m2":
            np.nan,

        "bh_outer_perimeter_m":
            np.nan,

        "bh_outer_equivalent_diameter_m":
            np.nan,

        "bh_hollow_area_m2":
            0.0,

        "bh_hollow_perimeter_m":
            0.0,

        "bh_hollow_equivalent_diameter_m":
            0.0,

        "bh_outside_area_m2":
            np.nan,
    }

    if (
        outer_geometry is None
        or outer_geometry.is_empty
    ):

        return features

    # --------------------------------------------------------
    # Outer geometry
    # --------------------------------------------------------

    outer_area = float(
        outer_geometry.area
    )

    features[
        "bh_outer_area_m2"
    ] = outer_area

    features[
        "bh_outer_perimeter_m"
    ] = float(
        outer_geometry.length
    )

    features[
        "bh_outer_equivalent_diameter_m"
    ] = equivalent_diameter(
        outer_area
    )

    # --------------------------------------------------------
    # Hollow
    # --------------------------------------------------------

    if (
        hollow_geometry is not None
        and not hollow_geometry.is_empty
    ):

        hollow_area = float(
            hollow_geometry.area
        )

        features[
            "bh_hollow_area_m2"
        ] = hollow_area

        features[
            "bh_hollow_perimeter_m"
        ] = float(
            hollow_geometry.length
        )

        features[
            "bh_hollow_equivalent_diameter_m"
        ] = equivalent_diameter(
            hollow_area
        )

    else:

        features[
            "bh_solid_fraction"
        ] = 1.0

    # --------------------------------------------------------
    # Outside area
    # --------------------------------------------------------

    try:

        original_area = float(
            original_polygon.area
        )

        if original_area > 0:

            features[
                "bh_outside_area_m2"
            ] = float(
                max(
                    original_area
                    - outer_area,
                    0.0,
                )
            )

    except Exception:

        pass

    return features


# ============================================================================
# 12. FULL BREAST-HEIGHT PIPELINE
# ============================================================================

def process_breast_height_geometry(
    breast_path,
    original_polygon,
    band_index=6,
    min_hollow_cells=4,
    alpha=0.15,
    clean_kwargs=None,
):
    """
    Complete breast-height geometry pipeline.

    Parameters
    ----------
    breast_path : str or Path
        Path to the breast-height density GeoTIFF.

    original_polygon : shapely Polygon
        Clump boundary.

    band_index : int
        Raster band containing the density values.

    min_hollow_cells : int
        Minimum number of raster cells for a hollow.

    alpha : float
        Alpha-shape parameter.

    clean_kwargs : dict
        Arguments passed to clean_point_cloud().

    Returns
    -------
    features
    outer_geometry
    hollow_geometry
    smoothed_outer_geometry
    """

    # ========================================================================
    # DEFAULT OUTPUT
    # ========================================================================

    features = {

        "bh_n_positive_cells":
            0,

        "bh_n_valid_cells":
            0,

        "bh_n_points_used_for_boundary":
            0,
            
        "bh_cleaning_eps":
            np.nan,

        "bh_outer_area_m2":
            np.nan,

        "bh_outer_perimeter_m":
            np.nan,

        "bh_outer_equivalent_diameter_m":
            np.nan,

        "bh_hollow_area_m2":
            0.0,

        "bh_hollow_perimeter_m":
            0.0,

        "bh_hollow_equivalent_diameter_m":
            0.0,

        "bh_outside_area_m2":
            np.nan,
    }

    outer_geometry = None
    hollow_geometry = None
    smoothed_outer_geometry = None

    # ========================================================================
    # 1. CREATE OCCUPANCY FROM THE GEOTIFF
    # ==================================================================

    try:

        occupancy, transform = (
            create_occupancy_grid(
                raster_path=breast_path,
                polygon=original_polygon,
                band_index=band_index,
            )
        )

    except Exception as e:

        features[
            "bh_geometry_status"
        ] = (
            f"raster_read_failed: {e}"
        )

        return (
            features,
            None,
            None,
            None,
        )

    # ========================================================================
    # 2. BASIC RASTER QC
    # ========================================================================

    n_valid_cells = int(
        np.sum(
            occupancy >= 0
        )
    )

    n_positive_cells = int(
        np.sum(
            occupancy == 1
        )
    )

    features[
        "bh_n_valid_cells"
    ] = n_valid_cells

    features[
        "bh_n_positive_cells"
    ] = n_positive_cells

    print(
        f"BH raster: "
        f"valid={n_valid_cells}, "
        f"positive={n_positive_cells}"
    )

    # ========================================================================
    # 3. NO POSITIVE CELLS
    # ========================================================================

    if n_positive_cells == 0:

        features[
            "bh_geometry_status"
        ] = "no_positive_cells"

        return (
            features,
            None,
            None,
            None,
        )

    # ========================================================================
    # 4. POSITIVE CELLS -> XY POINTS
    # ========================================================================

    raw_points = (
        get_positive_points(
            occupancy,
            transform,
        )
    )

    if len(raw_points) < 3:

        features[
            "bh_geometry_status"
        ] = "too_few_positive_points"

        return (
            features,
            None,
            None,
            None,
        )

    print(
        f"BH points: "
        f"{len(raw_points)} raw"
    )

    # ========================================================================
    # 5. CLEAN POINT CLOUD
    # ========================================================================

    if clean_kwargs is None:

        clean_kwargs = {}

    clean_points, clean_stats = (
        clean_point_cloud(
            raw_points,
            **clean_kwargs,
        )
    )

    features[
        "bh_n_points_used_for_boundary"
    ] = len(clean_points)

    features[
        "bh_n_points_removed_as_noise"
    ] = clean_stats[
        "n_removed"
    ]

    # eps may be None
    if clean_stats[
        "eps_used"
    ] is not None:

        features[
            "bh_cleaning_eps"
        ] = float(
            clean_stats[
                "eps_used"
            ]
        )

    print(
        f"BH points after cleaning: "
        f"{len(clean_points)}"
    )

    # ========================================================================
    # 6. TOO FEW CLEAN POINTS
    # ========================================================================

    if len(clean_points) < 3:

        features[
            "bh_geometry_status"
        ] = (
            "too_few_clean_points"
        )

        return (
            features,
            None,
            None,
            None,
        )

    # ========================================================================
    # 7. BUILD OUTER GEOMETRY
    # ========================================================================

    outer_geometry = (
        _points_to_outer_geometry(
            points=clean_points,
            transform=transform,
            original_polygon=original_polygon,
            alpha=alpha,
        )
    )

    if (
        outer_geometry is None
        or outer_geometry.is_empty
    ):

        features[
            "bh_geometry_status"
        ] = (
            "outer_geometry_failed"
        )

        return (
            features,
            None,
            None,
            None,
        )

    print(
        f"BH outer geometry: "
        f"area={outer_geometry.area:.4f}, "
        f"perimeter={outer_geometry.length:.4f}"
    )

    # ========================================================================
    # 8. HOLLOW DETECTION
    # ========================================================================

    # VERY IMPORTANT:
    #
    # Use ORIGINAL occupancy.
    #
    # Do NOT replace occupancy with cleaned points.
    #
    # Cleaning determines which observed points are trusted for the
    # OUTER boundary. It does not mean that removed points were actually
    # empty.

    hollow_geometry = (
        find_hollow_geometry(
            occupancy=occupancy,
            transform=transform,
            outer_geometry=outer_geometry,
            original_polygon=original_polygon,
            min_hollow_cells=min_hollow_cells,
        )
    )

    # ========================================================================
    # 9. FEATURES
    # ========================================================================

    geometry_feature_values = (
        geometry_features(
            outer_geometry=outer_geometry,
            hollow_geometry=hollow_geometry,
            original_polygon=original_polygon,
        )
    )

    features.update(
        geometry_feature_values
    )

    # ========================================================================
    # 10. SMOOTHED OUTER
    # ========================================================================

    #
    # Keep this conservative.
    #
    # A small buffer in/out produces a visually smoother boundary while
    # retaining approximately the same area.
    #

    try:

        # Estimate pixel size from raster transform.
        pixel_size = float(
            max(
                abs(transform.a),
                abs(transform.e),
            )
        )

        smoothing_radius = (
            10.0 * pixel_size
        )

        smoothed_outer_geometry = (
            outer_geometry
            .buffer(
                smoothing_radius
            )
            .buffer(
                -smoothing_radius
            )
        )

        if (
            smoothed_outer_geometry.is_empty
        ):

            smoothed_outer_geometry = (
                outer_geometry
            )

    except Exception:

        smoothed_outer_geometry = (
            outer_geometry
        )

    # ========================================================================
    # 11. SUCCESS
    # ========================================================================
    features[
        "bh_geometry_status"
    ] = "success"

    return (
        features,
        outer_geometry,
        hollow_geometry,
        smoothed_outer_geometry,
    )


# ============================================================================
# 13. GEOMETRY RECORDS
# ============================================================================

def add_geometry_records(
    records,
    psp,
    clump_id,
    original_polygon,
    outer_geometry=None,
    hollow_geometry=None,
    smoothed_outer_geometry=None,
):
    """
    Add geometries to a list for visual inspection.
    """

    # ------------------------------------------------------------------------
    # Original clump
    # ------------------------------------------------------------------------

    if (
        original_polygon is not None
        and not original_polygon.is_empty
    ):

        records.append({
            "psp": psp,
            "clump_id": clump_id,
            "geometry_type": "original",
            "geometry": original_polygon,
        })

    # ------------------------------------------------------------------------
    # Outer slice
    # ------------------------------------------------------------------------

    if (
        outer_geometry is not None
        and not outer_geometry.is_empty
    ):

        records.append({
            "psp": psp,
            "clump_id": clump_id,
            "geometry_type": "outer_slice",
            "geometry": outer_geometry,
        })

        # ----------------------------------------------------
        # Outside
        # ----------------------------------------------------

        try:

            outside = (
                original_polygon
                .difference(
                    outer_geometry
                )
            )

            if (
                outside is not None
                and not outside.is_empty
            ):

                records.append({
                    "psp": psp,
                    "clump_id": clump_id,
                    "geometry_type": "outside",
                    "geometry": outside,
                })

        except Exception as e:

            print(
                f"Outside geometry failed "
                f"for clump {clump_id}: {e}"
            )

    # ------------------------------------------------------------------------
    # Smoothed outer
    # ------------------------------------------------------------------------

    if (
        smoothed_outer_geometry
        is not None
        and not smoothed_outer_geometry.is_empty
    ):

        records.append({
            "psp": psp,
            "clump_id": clump_id,
            "geometry_type":
                "outer_slice_smoothed",
            "geometry":
                smoothed_outer_geometry,
        })

    # ------------------------------------------------------------------------
    # Hollow
    # ------------------------------------------------------------------------

    if (
        hollow_geometry is not None
        and not hollow_geometry.is_empty
    ):

        records.append({
            "psp": psp,
            "clump_id": clump_id,
            "geometry_type": "hollow",
            "geometry": hollow_geometry,
        })


# ============================================================================
# 14. EXPORT GEOMETRY
# ============================================================================

def export_geometry_gpkg(
    records,
    output_path,
    layer="bh_geometry",
    crs=None,
):
    """
    Export geometry records to GeoPackage.
    """

    if not records:

        print(
            "No geometry records to export."
        )

        return None

    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=crs,
    )

    gdf = gdf[
        gdf.geometry.notna()
        &
        ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:

        print(
            "Geometry GeoDataFrame is empty."
        )

        return None

    gdf.to_file(
        output_path,
        layer=layer,
        driver="GPKG",
    )

    print(
        f"Exported {len(gdf)} geometry records."
    )

    print(
        gdf[
            "geometry_type"
        ].value_counts()
    )

    return gdf