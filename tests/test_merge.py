from functools import partial

import dask
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations import merge_cubes
from openeo_processes_dask.process_implementations.cubes.merge import (
    NEW_DIM_COORDS,
    NEW_DIM_NAME,
)
from openeo_processes_dask.process_implementations.exceptions import (
    OverlapResolverMissing,
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
        bands=["B02", "B03", "B04", "--324"],
        backend="dask",
    )

    cube_1 = origin_cube.drop_sel({"bands": ["B04", "--324"]})
    cube_2 = origin_cube.drop_sel({"bands": ["B02", "B03"]})

    merged_cube = merge_cubes(cube_1, cube_2)
    assert isinstance(merged_cube.data, dask.array.Array)

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
        backend="dask",
    )

    cube_1 = origin_cube.drop_sel({"bands": "B03"})
    cube_2 = origin_cube.drop_sel({"bands": "B01"})

    with pytest.raises(OverlapResolverMissing):
        merge_cubes(cube_1, cube_2)

    overlap_resolver = partial(
        process_registry["add"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=ParameterReference(from_parameter="y"),
    )
    merged_cube = merge_cubes(cube_1, cube_2, overlap_resolver=overlap_resolver)
    assert isinstance(merged_cube.data, dask.array.Array)

    xr.testing.assert_equal(
        merged_cube.sel({"bands": "B02"}) / 2, origin_cube.sel({"bands": "B02"})
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
        backend="dask",
    )

    cube_1 = origin_cube
    cube_2 = origin_cube

    # If no overlap reducer is provided, then simply concatenate along a new dimension
    merged_cube = merge_cubes(cube_1, cube_2)
    expected_result = xr.concat([cube_1, cube_2], dim=NEW_DIM_NAME).reindex(
        {NEW_DIM_NAME: NEW_DIM_COORDS}
    )
    xr.testing.assert_equal(merged_cube, expected_result)

    # If an overlap reducer is provided, then reduce per pixel
    merged_cube = merge_cubes(
        cube_1,
        cube_2,
        partial(
            process_registry["add"].implementation,
            x=ParameterReference(from_parameter="x"),
            y=ParameterReference(from_parameter="y"),
        ),
    )
    assert isinstance(merged_cube.data, dask.array.Array)

    xr.testing.assert_equal(merged_cube, cube_1 * 2)


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
        backend="dask",
    )

    cube_2 = xr.DataArray(
        np.ones((len(cube_1["x"]), len(cube_1["y"]))),
        dims=["x", "y"],
        coords={"x": cube_1.coords["x"], "y": cube_1.coords["y"]},
    )

    with pytest.raises(OverlapResolverMissing):
        merge_cubes(cube_1, cube_2)

    overlap_resolver = partial(
        process_registry["add"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=ParameterReference(from_parameter="y"),
    )
    merged_cube_1 = merge_cubes(cube_1, cube_2, overlap_resolver=overlap_resolver)
    merged_cube_2 = merge_cubes(cube_2, cube_1, overlap_resolver=overlap_resolver)

    assert isinstance(merged_cube_1.data, dask.array.Array)
    xr.testing.assert_equal(merged_cube_1, cube_1 + 1)

    assert isinstance(merged_cube_2.data, dask.array.Array)
    xr.testing.assert_equal(merged_cube_2, cube_1 + 1)


@pytest.mark.parametrize("size", [(6, 5, 4, 1)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_conflicting_coords(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    # See https://github.com/Open-EO/openeo-processes-dask/pull/148 for why is is necessary
    # This is basically broadcasting the smaller datacube and then applying the overlap resolver.
    cube_1 = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01"],
        backend="dask",
    )
    cube_1["s2:processing_baseline"] = "05.8"
    cube_2 = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
    )
    cube_2["s2:processing_baseline"] = "05.9"

    merged_cube_1 = merge_cubes(cube_1, cube_2)

    assert isinstance(merged_cube_1.data, dask.array.Array)
