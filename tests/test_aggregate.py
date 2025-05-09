from functools import partial

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import xvec
from openeo_pg_parser_networkx.pg_schema import (
    ParameterReference,
    TemporalInterval,
    TemporalIntervals,
)

from openeo_processes_dask.process_implementations.cubes.aggregate import *
from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from openeo_processes_dask.process_implementations.math import mean
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 100, 4)])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize(
    "temporal_extent,intervals,labels, expected",
    [
        (
            ["2018-01-01T00:00:00", "2019-01-01T00:00:00"],
            [
                ["2018-01-01T12:00:00", "2018-06-01T12:00:00"],
                ["2018-07-01T12:00:00", "2018-12-01T12:00:00"],
            ],
            ["half-1", "half-2"],
            2,
        ),
        (
            ["2018-01-01T00:00:00", "2019-01-01T00:00:00"],
            [
                TemporalInterval.model_validate(
                    ["2018-01-01T12:00:00", "2018-06-01T12:00:00"]
                ),
                TemporalInterval.model_validate(
                    ["2018-07-01T12:00:00", "2018-12-01T12:00:00"]
                ),
            ],
            ["half-1", "half-2"],
            2,
        ),
        (
            ["2018-01-01T00:00:00", "2019-01-01T00:00:00"],
            TemporalIntervals.model_validate(
                [
                    ["2018-01-01T12:00:00", "2018-06-01T12:00:00"],
                    ["2018-07-01T12:00:00", "2018-12-01T12:00:00"],
                ]
            ),
            ["half-1", "half-2"],
            2,
        ),
    ],
)
def test_aggregate_temporal(
    temporal_extent,
    intervals,
    labels,
    expected,
    bounding_box,
    random_raster_data,
    process_registry,
):
    """"""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=TemporalInterval.model_validate(temporal_extent),
        bands=["B02", "B03", "B04", "B08"],
    )

    reducer = partial(
        process_registry["mean"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube = aggregate_temporal(
        data=input_cube, intervals=intervals, reducer=reducer, labels=labels
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )

    assert len(output_cube.t) == expected
    assert isinstance(output_cube.t.values[0], str)


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize(
    "temporal_extent,period,expected",
    [
        (["2018-05-01", "2018-05-02"], "hour", 25),
        (["2018-05-01", "2018-06-01"], "day", 32),
        (["2018-05-01", "2018-06-01"], "week", 5),
        (["2018-05-01", "2018-08-31"], "dekad", 12),
        (["2018-05-15", "2018-08-14"], "dekad", 10),
        (["2018-05-01", "2018-06-01"], "month", 2),
        (["2018-01-01", "2018-12-31"], "season", 5),
        (["2018-05-01", "2019-04-30"], "tropical-season", 2),
        (["2018-01-01", "2018-12-31"], "year", 1),
        (["2019-01-01", "2022-12-31"], "decade", 2),
        (["2019-01-01", "2022-12-31"], "decade-ad", 2),
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
    assert output_cube.t.values[0].dtype.type in [np.str_, np.datetime64]


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
@pytest.mark.parametrize("dtype", [np.float32])
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

    reducer = partial(
        process_registry["mean"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

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

    gdf = gpd.GeoDataFrame.from_features(polygon_geometry_small, crs="EPSG:4326")
    xmin, ymin, xmax, ymax = gdf.total_bounds

    expected_values = (
        reduced_cube.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
        .mean(["x", "y"])
        .values
    )

    assert (output_cube.values == expected_values).all()

    gdf = gpd.GeoDataFrame.from_features(polygon_geometry_small, crs="EPSG:4326")
    gdf_equi7 = gdf.to_crs(
        "+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs"
    )
    output_cube_transform = aggregate_spatial(
        data=reduced_cube, geometries=gdf_equi7, reducer=reducer
    )
    assert len(output_cube_transform.dims) == len(output_cube.dims)
    assert output_cube_transform.shape == output_cube.shape

    geometry_cube = xr.Dataset(
        data_vars={"variable": (["geometry"], np.arange(len(gdf)))},
        coords={"geometry": gdf["geometry"].values},
    ).xvec.set_geom_indexes("geometry", crs=gdf.crs)
    output_cube_transform = aggregate_spatial(
        data=reduced_cube, geometries=geometry_cube, reducer=reducer
    )
    assert len(output_cube_transform.dims) == len(output_cube.dims)
    assert output_cube_transform.shape == output_cube.shape

    polygon_geometry_small["crs"] = 4326

    output_cube = aggregate_spatial(
        data=reduced_cube, geometries=polygon_geometry_small, reducer=reducer
    )

    assert len(output_cube.dims) < len(reduced_cube.dims)

    geometry_url = "https://raw.githubusercontent.com/ValentinaHutter/polygons/master/polygons_small.json"
    output_cube = aggregate_spatial(
        data=reduced_cube, geometries=geometry_url, reducer=reducer
    )

    assert len(output_cube.geometry) == 38

    geometry = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0]]]}
    output_cube = aggregate_spatial(
        data=reduced_cube, geometries=geometry, reducer=reducer
    )

    assert np.isnan(output_cube.values).all()
