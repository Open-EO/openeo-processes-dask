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


def test_filter_bbox_vectorcube():
    """Test filter_bbox with VectorCube (GeoDataFrame)"""
    import geopandas as gpd
    from openeo_pg_parser_networkx.pg_schema import BoundingBox
    from shapely.geometry import Point

    # Create a sample VectorCube with points
    points = [
        Point(10.47, 46.15),  # Inside bbox
        Point(10.49, 46.17),  # Inside bbox
        Point(10.46, 46.11),  # Outside bbox (west of bbox)
        Point(10.51, 46.19),  # Outside bbox (east of bbox)
        Point(10.48, 46.10),  # Outside bbox (south of bbox)
    ]

    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Point_A", "Point_B", "Point_C", "Point_D", "Point_E"],
            "geometry": points,
        },
        crs="EPSG:4326",
    )

    # Define a bounding box that should filter to 2 points
    bbox = BoundingBox(
        west=10.47, east=10.50, south=46.12, north=46.18, crs="EPSG:4326"
    )

    # Apply filter_bbox
    filtered_gdf = filter_bbox(data=gdf, extent=bbox)

    # Verify results
    assert isinstance(filtered_gdf, gpd.GeoDataFrame)
    assert len(filtered_gdf) == 2  # Only 2 points should be inside
    assert set(filtered_gdf["id"]) == {1, 2}  # Points A and B
    assert filtered_gdf.crs == gdf.crs  # CRS should be preserved


def test_filter_bbox_vectorcube_crs_reprojection():
    """Test filter_bbox with VectorCube using different CRS (tests _reproject_bbox)"""
    import geopandas as gpd
    from openeo_pg_parser_networkx.pg_schema import BoundingBox
    from shapely.geometry import Point

    # Create a VectorCube in UTM Zone 32N (EPSG:32632) - covering northern Italy
    # Coordinates are in meters (UTM)
    points_utm = [
        Point(630000, 5100000),  # lon=10.680142, lat=46.041224 - Inside bbox
        Point(635000, 5105000),  # lon=10.746151, lat=46.085236 - Inside bbox
        Point(625000, 5095000),  # lon=10.614238, lat=45.997172 - Outside bbox (south)
        Point(
            640000, 5110000
        ),  # lon=10.812265, lat=46.129208 - Outside bbox (east and north)
    ]

    gdf_utm = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Point_A", "Point_B", "Point_C", "Point_D"],
            "geometry": points_utm,
        },
        crs="EPSG:32632",  # UTM Zone 32N
    )

    # Define bounding box in WGS84 (EPSG:4326) - degrees
    # This bbox should cover points 1 and 2 when reprojected
    bbox_wgs84 = BoundingBox(
        west=10.65, east=10.76, south=46.02, north=46.10, crs="EPSG:4326"
    )

    # Apply filter_bbox - should trigger CRS reprojection
    filtered_gdf = filter_bbox(data=gdf_utm, extent=bbox_wgs84)

    # Verify results
    assert isinstance(filtered_gdf, gpd.GeoDataFrame)
    assert filtered_gdf.crs == gdf_utm.crs  # CRS should remain UTM
    # Should have exactly 2 points (points 1 and 2)
    assert len(filtered_gdf) == 2
    assert set(filtered_gdf["id"].values) == {1, 2}
    assert set(filtered_gdf["name"].values) == {"Point_A", "Point_B"}


def test_filter_bbox_vectorcube_xarray_dataset():
    """Test filter_bbox with VectorCube as xarray.Dataset with geometry variable"""
    import geopandas as gpd
    import xarray as xr
    from openeo_pg_parser_networkx.pg_schema import BoundingBox
    from shapely.geometry import Point

    # Create sample geometries
    points = [
        Point(10.47, 46.15),  # Inside bbox
        Point(10.49, 46.17),  # Inside bbox
        Point(10.46, 46.11),  # Outside bbox (west)
        Point(10.51, 46.19),  # Outside bbox (east)
        Point(10.48, 46.10),  # Outside bbox (south)
    ]

    # Create a GeoSeries with geometries
    geoms = gpd.GeoSeries(points, crs="EPSG:4326")

    # Create an xarray.Dataset with geometry variable (VectorCube format)
    dataset = xr.Dataset(
        {
            "geometry": xr.DataArray(geoms, dims=["features"]),
            "id": xr.DataArray([1, 2, 3, 4, 5], dims=["features"]),
            "name": xr.DataArray(
                ["Point_A", "Point_B", "Point_C", "Point_D", "Point_E"],
                dims=["features"],
            ),
        }
    )
    # Add CRS as attribute (standard way for xarray Datasets)
    dataset.attrs["crs"] = "EPSG:4326"

    # Define a bounding box that should filter to 2 points
    bbox = BoundingBox(
        west=10.47, east=10.50, south=46.12, north=46.18, crs="EPSG:4326"
    )

    # Apply filter_bbox
    filtered_dataset = filter_bbox(data=dataset, extent=bbox)

    # Verify results
    assert isinstance(filtered_dataset, xr.Dataset)
    assert "geometry" in filtered_dataset
    # Should have exactly 2 features (points A and B)
    assert len(filtered_dataset.features) == 2
    assert set(filtered_dataset["id"].values) == {1, 2}
    assert set(filtered_dataset["name"].values) == {"Point_A", "Point_B"}
