import numpy as np
import pytest
from odc.geo.geobox import resolution_from_affine
from pyproj.crs import CRS

from openeo_processes_dask.process_implementations.cubes.resample import (
    resample_spatial,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize(
    "output_crs",
    [
        3587,
        "32633",
        "+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs",
        "4326",
    ],
)
@pytest.mark.parametrize("output_res", [5, 30, 60])
@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_resample_spatial(
    output_crs, output_res, temporal_interval, bounding_box, random_raster_data
):
    """Test to ensure resolution gets changed correctly."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    output_cube = resample_spatial(
        data=input_cube, projection=output_crs, resolution=output_res
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=False,
    )

    assert output_cube.odc.spatial_dims == ("y", "x")
    assert output_cube.rio.crs == CRS.from_user_input(output_crs)

    if output_crs != "4326":
        assert resolution_from_affine(output_cube.geobox.affine).x == output_res
        assert resolution_from_affine(output_cube.geobox.affine).y == -output_res

        assert min(output_cube.x) >= -180
        assert max(output_cube.x) <= 180

        assert min(output_cube.y) >= -90
        assert max(output_cube.y) <= 90
