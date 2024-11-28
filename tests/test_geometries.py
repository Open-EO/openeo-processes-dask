from copy import deepcopy
import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr
import xvec
from openeo_pg_parser_networkx.pg_schema import ParameterReference, TemporalInterval

from openeo_processes_dask.process_implementations.cubes.geometries import *
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
    UnitMismatch,
)


def test_load_geojson_point(
    geometry_point
):
    """"""
    geometry = load_geojson(geometry_point)

    assert isinstance(geometry, xr.DataArray)
    assert geometry.dims == ("geometry", "properties")
    assert hasattr(geometry["geometry"], 'crs')
    assert len(geometry["properties"]) == 2

    geometry_class = load_geojson(geometry_point, properties=["class"])
    
    assert isinstance(geometry, xr.DataArray)
    assert geometry_class.dims == ("geometry", "properties")
    assert geometry_class["properties"].values == "class"


def test_load_geojson_polygon(
    geometry_dict
):
    """"""
    geometry = load_geojson(geometry_dict)

    assert isinstance(geometry, xr.DataArray)
    assert geometry.dims == ("geometry", "properties")
    assert hasattr(geometry["geometry"], 'crs')
    assert len(geometry["properties"]) == 2

    geometry_class = load_geojson(geometry_dict, properties=["class"])
    
    assert isinstance(geometry, xr.DataArray)
    assert geometry_class.dims == ("geometry", "properties")
    assert geometry_class["properties"].values == "class"


    geometry_dict_2 = deepcopy(geometry_dict)
    geometry_dict_2["features"][0]["properties"]["class"] = [2,3,4,5,6]

    geometry_2 = load_geojson(geometry_dict_2, properties=["class"])
    
    assert isinstance(geometry_2, xr.DataArray)
    assert geometry_class.dims == ("geometry", "properties")
    assert geometry_class["properties"].values == "class"


def test_point_reproject(
        geometry_point,
        wkt_string
):
    """"""
    geometry = load_geojson(geometry_point)
    geometry_crs = geometry["geometry"].crs

    epsg_vector = vector_reproject(data = geometry, projection="EPSG:3857")
    epsg_crs = epsg_vector["geometry"].crs
    
    assert geometry_crs != epsg_crs
    
    wkt_vector = vector_reproject(data = geometry, projection=wkt_string)
    wkt_crs = wkt_vector["geometry"].crs

    assert geometry_crs != wkt_crs


def test_polygon_reproject(
        geometry_dict,
        wkt_string
):
    """"""
    geometry = load_geojson(geometry_dict)
    geometry_crs = geometry["geometry"].crs

    epsg_vector = vector_reproject(data = geometry, projection="EPSG:3857")
    epsg_crs = epsg_vector["geometry"].crs
    
    assert geometry_crs != epsg_crs
    
    wkt_vector = vector_reproject(data = geometry, projection=wkt_string)
    wkt_crs = wkt_vector["geometry"].crs

    assert geometry_crs != wkt_crs


def test_point_buffer(
        geometry_point
):
    """"""
    geometry = load_geojson(geometry_point)
    epsg_vector = vector_reproject(data = geometry, projection="EPSG:3857")

    buffered_vector = vector_buffer(geometries = epsg_vector, distance=1)
    buffered_area = shapely.area(buffered_vector["geometry"].values[0])

    assert buffered_area


def test_polygon_buffer(
        geometry_dict
):
    """"""
    geometry = load_geojson(geometry_dict)
    epsg_vector = vector_reproject(data = geometry, projection="EPSG:3857")
    geometry_area = shapely.area(epsg_vector["geometry"].values[0])

    buffered_vector = vector_buffer(geometries = epsg_vector, distance=0.5)
    buffered_area = shapely.area(buffered_vector["geometry"].values[0])
    
    assert buffered_area > geometry_area

    with pytest.raises(UnitMismatch):
        error = vector_buffer(geometries = geometry, distance=1)
