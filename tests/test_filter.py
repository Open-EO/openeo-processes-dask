import copy
import datetime
from functools import partial

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference, TemporalInterval

from openeo_processes_dask.process_implementations.cubes._filter import *
from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
    TemporalExtentEmpty,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 30, 1)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_temporal(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
    )

    temporal_interval_part = TemporalInterval.parse_obj(
        ["2018-05-15T00:00:00", "2018-06-01T00:00:00"]
    )
    output_cube = filter_temporal(data=input_cube, extent=temporal_interval_part)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )

    xr.testing.assert_equal(
        output_cube,
        input_cube.loc[dict(t=slice("2018-05-15T00:00:00", "2018-05-31T23:59:59"))],
    )

    with pytest.raises(DimensionNotAvailable):
        filter_temporal(
            data=input_cube, extent=temporal_interval_part, dimension="immissing"
        )

    with pytest.raises(TemporalExtentEmpty):
        filter_temporal(
            data=input_cube,
            extent=["2018-05-31T23:59:59", "2018-05-15T00:00:00"],
        )

    temporal_interval_open = TemporalInterval.parse_obj([None, "2018-05-03T00:00:00"])
    output_cube = filter_temporal(data=input_cube, extent=temporal_interval_open)

    xr.testing.assert_equal(
        output_cube,
        input_cube.loc[dict(t=slice("2018-05-01T00:00:00", "2018-05-02T23:59:59"))],
    )

    new_coords = list(copy.deepcopy(input_cube.coords["t"].data))
    new_coords[1] = pd.NaT
    invalid_input_cube = input_cube.assign_coords({"t": np.array(new_coords)})
    filter_temporal(invalid_input_cube, temporal_interval)


@pytest.mark.parametrize("size", [(30, 30, 30, 3)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_labels(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04"],
        backend="dask",
    )
    _process = partial(
        process_registry["eq"].implementation,
        y="B04",
        x=ParameterReference(from_parameter="x"),
    )

    output_cube = filter_labels(data=input_cube, condition=_process, dimension="bands")
    assert len(output_cube["bands"]) == 1


@pytest.mark.parametrize("size", [(1, 1, 1, 2)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_bands(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "SCL"],
        backend="dask",
    )

    output_cube = filter_bands(data=input_cube, bands=["SCL"])

    assert output_cube["bands"].values == "SCL"


@pytest.mark.parametrize("size", [(30, 30, 30, 1)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_spatial(
    temporal_interval,
    bounding_box,
    random_raster_data,
    polygon_geometry_small,
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
    )

    output_cube = filter_spatial(data=input_cube, geometries=polygon_geometry_small)

    assert len(output_cube.y) < len(input_cube.y)
    assert len(output_cube.x) < len(input_cube.x)


@pytest.mark.parametrize("size", [(30, 30, 1, 1)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_bbox(
    temporal_interval,
    bounding_box,
    random_raster_data,
    bounding_box_small,
    process_registry,
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
    )

    output_cube = filter_bbox(data=input_cube, extent=bounding_box_small)

    assert len(output_cube.y) < len(input_cube.y)
    assert len(output_cube.x) < len(input_cube.x)

    _process = partial(
        process_registry["mean"].implementation,
        ignore_nodata=True,
        data=ParameterReference(from_parameter="data"),
    )

    input_cube_no_y = reduce_dimension(data=input_cube, reducer=_process, dimension="y")
    output_cube_cube_no_y = filter_bbox(data=input_cube_no_y, extent=bounding_box_small)

    input_cube_no_x = reduce_dimension(data=input_cube, reducer=_process, dimension="x")
    output_cube_cube_no_x = filter_bbox(data=input_cube_no_x, extent=bounding_box_small)

    output_cube_cube_no_x_y = reduce_dimension(
        data=input_cube_no_x, reducer=_process, dimension="y"
    )

    assert len(output_cube.y) == len(output_cube_cube_no_x.y)
    assert len(output_cube.x) == len(output_cube_cube_no_y.x)

    with pytest.raises(DimensionNotAvailable):
        filter_bbox(data=output_cube_cube_no_x_y, extent=bounding_box_small)
