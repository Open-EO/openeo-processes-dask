import numpy as np
import pytest
from pyproj.crs import CRS

from openeo_processes_dask.process_implementations.cubes.resample import (
    resample_spatial,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("output_crs", [3587, 32633, 6068])
@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_resample_spatial(
    output_crs, temporal_interval, bounding_box, random_raster_data
):
    """Test to ensure CRS get changed correctly."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    output_cube = resample_spatial(data=input_cube, epsg_code=output_crs)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=False,
    )

    assert output_cube.rio.crs == CRS.from_epsg(output_crs)
    assert len(output_cube.x) == len(input_cube.x)
