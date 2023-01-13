import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask.exceptions import OverlapResolverMissing
from openeo_processes_dask.process_implementations import merge_cubes
from openeo_processes_dask.process_implementations.cubes.merge import (
    NEW_DIM_COORDS,
    NEW_DIM_NAME,
)
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_type_1(temporal_interval, bounding_box, random_raster_data):
    """See Example 1 from https://processes.openeo.org/#merge_cubes."""
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
    )

    cube_1 = origin_cube.drop_sel({"bands": ["B04", "B08"]})
    cube_2 = origin_cube.drop_sel({"bands": ["B02", "B03"]})

    merged_cube = merge_cubes(cube_1, cube_2)
    xr.testing.assert_equal(merged_cube, origin_cube)


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_type_2(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01", "B02", "B03"],
    )

    cube_1 = origin_cube.drop_sel({"bands": "B03"})
    cube_2 = origin_cube.drop_sel({"bands": "B01"})

    with pytest.raises(OverlapResolverMissing):
        merge_cubes(cube_1, cube_2)

    overlap_resolver = process_registry["mean"]
    merged_cube = merge_cubes(cube_1, cube_2, overlap_resolver=overlap_resolver)
    xr.testing.assert_equal(
        merged_cube.sel({"bands": "B02"}), origin_cube.sel({"bands": "B02"})
    )


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_type_3(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    # This is basically broadcasting the smaller datacube and then applying the overlap resolver.
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01", "B02", "B03"],
    )

    cube_1 = origin_cube
    cube_2 = origin_cube + 1

    # If no overlap reducer is provided, then simply concatenate along a new dimension
    merged_cube = merge_cubes(cube_1, cube_2)
    expected_result = xr.concat([cube_1, cube_2], dim=NEW_DIM_NAME).reindex(
        {NEW_DIM_NAME: NEW_DIM_COORDS}
    )
    xr.testing.assert_equal(merged_cube, expected_result)

    # If an overlap reducer is provided, then reduce per pixel
    merged_cube = merge_cubes(cube_1, cube_2, process_registry["max"])
    xr.testing.assert_equal(merged_cube, cube_1 + 1)


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_type_4(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    # This is basically broadcasting the smaller datacube and then applying the overlap resolver.
    cube_1 = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01", "B02", "B03"],
    )

    cube_2 = xr.DataArray(
        np.ones((len(cube_1["x"]), len(cube_1["y"]))),
        dims=["x", "y"],
        coords={"x": cube_1.coords["x"], "y": cube_1.coords["y"]},
    )

    with pytest.raises(OverlapResolverMissing):
        merge_cubes(cube_1, cube_2)

    overlap_resolver = process_registry["sum"]
    merged_cube = merge_cubes(cube_1, cube_2, overlap_resolver=overlap_resolver)
    xr.testing.assert_equal(merged_cube, cube_1 + 1)
