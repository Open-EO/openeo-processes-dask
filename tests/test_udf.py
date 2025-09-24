import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask.process_implementations.udf.udf import run_udf
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_run_udf(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    udf = """
import xarray as xr
def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    return cube + 1
"""

    output_cube = run_udf(data=input_cube, udf=udf, runtime="Python")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=input_cube + 1,
    )

    xr.testing.assert_equal(output_cube, input_cube + 1)

    # Test Issue #330 fix: verify semantic dimensions are preserved
    output_dims = list(output_cube.dims)
    generic_dims = [d for d in output_dims if str(d).startswith('dim_')]
    assert not generic_dims, f"Issue #330 regression: found generic dimensions {generic_dims}"
    
    # Should have meaningful dimension names for 4D data
    expected_semantic_dims = ['time', 'band', 'y', 'x']
    assert output_dims == expected_semantic_dims, f"Expected {expected_semantic_dims}, got {output_dims}"


@pytest.mark.parametrize("size", [(3, 4, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_run_udf_3d_dimensions(temporal_interval, bounding_box, random_raster_data):
    """Test UDF with 3D data preserves semantic dimensions (Issue #330 fix)."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],  # Single band for 3D test
        backend="dask",
    )

    udf = """
import xarray as xr
def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    # Verify semantic dimensions in UDF
    dims = list(cube.dims)
    assert 'time' in dims, f"Expected 'time' dimension, got {dims}"
    assert 'y' in dims, f"Expected 'y' dimension, got {dims}"
    assert 'x' in dims, f"Expected 'x' dimension, got {dims}"
    return cube * 2
"""

    output_cube = run_udf(data=input_cube, udf=udf, runtime="Python")
    
    # Verify dimensions are semantic
    assert list(output_cube.dims) == ['time', 'y', 'x']
