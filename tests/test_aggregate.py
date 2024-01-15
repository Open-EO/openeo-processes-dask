from functools import partial

import numpy as np
import pytest
from openeo_pg_parser_networkx.pg_schema import ParameterReference, TemporalInterval

from openeo_processes_dask.process_implementations.cubes.aggregate import (
    aggregate_spatial,
    aggregate_temporal_period,
)
from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from openeo_processes_dask.process_implementations.math import mean
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize(
    "temporal_extent,period,expected",
    [
        (["2018-05-01", "2018-05-02"], "hour", 25),
        (["2018-05-01", "2018-06-01"], "day", 32),
        (["2018-05-01", "2018-06-01"], "week", 5),
        (["2018-05-01", "2018-06-01"], "month", 2),
        (["2018-01-01", "2018-12-31"], "season", 5),
        (["2018-01-01", "2018-12-31"], "year", 1),
    ],
)
def test_aggregate_temporal_period(
    temporal_extent,
    period,
    expected,
    bounding_box,
    random_raster_data,
    process_registry,
):
    """"""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=TemporalInterval.parse_obj(temporal_extent),
        bands=["B02", "B03", "B04", "B08"],
    )

    reducer = partial(
        process_registry["mean"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube = aggregate_temporal_period(
        data=input_cube, period=period, reducer=reducer
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )

    assert len(output_cube.t) == expected
    assert isinstance(output_cube.t.values[0], np.datetime64)


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_aggregate_temporal_period_numpy_equals_dask(
    random_raster_data, bounding_box, temporal_interval, process_registry
):
    numpy_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="numpy",
    )
    dask_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    reducer = partial(
        process_registry["mean"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    func = partial(aggregate_temporal_period, reducer=reducer, period="hour")
    assert_numpy_equals_dask_numpy(
        numpy_cube=numpy_cube, dask_cube=dask_cube, func=func
    )


@pytest.mark.parametrize("size", [(30, 30, 30, 3)])
@pytest.mark.parametrize("dtype", [np.int8])
def test_aggregate_spatial(
    random_raster_data,
    bounding_box,
    temporal_interval,
    polygon_geometry_small,
    process_registry,
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04"],
        backend="dask",
    )

    reducer = "mean"

    output_cube = aggregate_spatial(
        data=input_cube, geometries=polygon_geometry_small, reducer=reducer
    )

    assert len(output_cube.dims) < len(input_cube.dims)

    _process = partial(
        process_registry["median"].implementation,
        ignore_nodata=True,
        data=ParameterReference(from_parameter="data"),
    )

    reduced_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="t")

    output_cube = aggregate_spatial(
        data=reduced_cube, geometries=polygon_geometry_small, reducer=reducer
    )

    assert len(output_cube.dims) < len(reduced_cube.dims)
