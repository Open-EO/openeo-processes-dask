from openeo.local import LocalConnection
from xcube_eopf.utils import reproject_bbox

# Shared configuration
BBOX = [
    9.669372670305636,
    53.64026948239441,
    9.701345402674315,
    53.66341039786631,
]  # ~250x250 px Northern Germany, Moorrege.

TIME_RANGE = ["2025-05-01", "2025-05-03"]

STAC_URL = "https://stac.core.eopf.eodc.eu/collections/sentinel-2-l2a"
BANDS = ["red", "nir"]  # band names are specific to the STAC Catalogs

CRS_REPRJ = "EPSG:32632"


def _make_connection():
    """Create a LocalConnection using the current working directory."""
    return LocalConnection("./")


def _make_spatial_extent():
    """Reproject bbox and build spatial_extent dict."""
    bbox_reprj = reproject_bbox(BBOX, "EPSG:4326", CRS_REPRJ)
    spatial_extent_reprj = {
        "west": bbox_reprj[0],
        "south": bbox_reprj[1],
        "east": bbox_reprj[2],
        "north": bbox_reprj[3],
        "crs": CRS_REPRJ,
    }
    return spatial_extent_reprj


def test_eopf_direct_access():
    """Test accessing Sentinel-2 L2A data directly using xcube-eopf store with same config as load_stac"""
    from xcube.core.store import new_data_store

    # Create EOPF store instance
    store = new_data_store("eopf-zarr")

    # Use the EXACT same configuration as load_stac tests
    bbox_reprj = reproject_bbox(BBOX, "EPSG:4326", CRS_REPRJ)

    # Open dataset with identical parameters as load_stac tests
    ds = store.open_data(
        data_id="sentinel-2-l2a",
        bbox=bbox_reprj,  # Using the same reprojected bbox
        time_range=TIME_RANGE,  # Using the same time range
        spatial_res=10.0,  # 10 meters resolution in target CRS
        crs=CRS_REPRJ,  # Using the same CRS as load_stac
        variables=BANDS,  # Using the EXACT same band names as STAC
    )

    # Verify dataset properties
    assert ds is not None, "Dataset should not be None"

    # Verify spatial dimensions and coordinates
    assert "y" in ds.dims or "lat" in ds.dims, "Should have y/latitude dimension"
    assert "x" in ds.dims or "lon" in ds.dims, "Should have x/longitude dimension"
    assert "time" in ds.dims, "Should have time dimension"

    # Verify we have data for the requested time range
    time_coords = ds.time.values
    assert len(time_coords) > 0, "Should have at least one time step"

    print("=== Test 3: EOPF Direct Access ===")
    print(f"Successfully loaded dataset with shape: {ds.dims}")
    print(f"Time steps: {len(time_coords)}")
    print(f"Variables: {list(ds.data_vars.keys())}")

    return ds


def test_load_stac_executes():
    """
    Test 1:
    Load STAC cube and execute it.
    The test fails if any part of the pipeline raises an error.
    """
    connection = _make_connection()
    spatial_extent_reprj = _make_spatial_extent()

    cube = connection.load_stac(
        url=STAC_URL,
        spatial_extent=spatial_extent_reprj,
        temporal_extent=TIME_RANGE,
        bands=BANDS,
    )

    result = cube.execute()

    # Basic sanity check
    assert result is not None
    print("=== Test 1: Raw cube ===")
    print(result)


def test_ndvi_mean_executes():
    """
    Test 2:
    Compute NDVI and reduce over time (mean over 't').
    Fails if any processing step breaks.
    """
    connection = _make_connection()
    spatial_extent_reprj = _make_spatial_extent()

    cube = connection.load_stac(
        url=STAC_URL,
        spatial_extent=spatial_extent_reprj,
        temporal_extent=TIME_RANGE,
        bands=BANDS,
    )

    red = cube.band(BANDS[0])
    nir = cube.band(BANDS[1])

    cube_ndvi = (nir - red) / (nir + red)
    cube_mnth = cube_ndvi.reduce_dimension(dimension="time", reducer="mean")
    result = cube_mnth.execute()

    assert result is not None
    print("=== Test 2: NDVI mean cube ===")
    print(result)
