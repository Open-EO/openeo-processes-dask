import numpy as np
import openeo
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
