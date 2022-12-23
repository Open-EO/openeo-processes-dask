import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask.process_implementations import merge_cubes
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_type_1(temporal_interval, bounding_box, random_raster_data):
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
    )

    cube_2 = origin_cube.copy(deep=True)
    cube_2 = cube_2.drop_sel({"bands": ["B02", "B03"]})
    cube_1 = origin_cube.drop_sel({"bands": ["B04", "B08"]})

    merged_cube = merge_cubes(cube_1, cube_2)
    xr.testing.assert_equal(merged_cube, origin_cube)
