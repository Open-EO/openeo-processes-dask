import importlib
import inspect
import json
import logging

import dask_geopandas
import geopandas as gpd
import numpy as np
import pytest
from dask.distributed import Client
from geopandas.geodataframe import GeoDataFrame
from openeo_pg_parser_networkx import Process, ProcessRegistry
from openeo_pg_parser_networkx.pg_schema import (
    DEFAULT_CRS,
    BoundingBox,
    TemporalInterval,
)
from shapely.geometry import Polygon

from openeo_processes_dask.process_implementations.core import process
from openeo_processes_dask.process_implementations.data_model import VectorCube

logger = logging.getLogger(__name__)


@pytest.fixture
def dask_client():
    client = Client()
    yield client
    client.shutdown()


def _random_raster_data(size, dtype, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.integers(-100, 100, size=size)
    data = data.astype(dtype)
    return data


@pytest.fixture
def random_raster_data(size, dtype, seed=42):
    return _random_raster_data(size, dtype, seed=seed)


@pytest.fixture
def bounding_box(
    west=10.45, east=10.5, south=46.1, north=46.2, crs="EPSG:4326"
) -> BoundingBox:
    spatial_extent = {
        "west": west,
        "east": east,
        "south": south,
        "north": north,
        "crs": crs,
    }
    return BoundingBox.parse_obj(spatial_extent)


@pytest.fixture
def bounding_box_small(
    west=10.47, east=10.48, south=46.12, north=46.18, crs="EPSG:4326"
) -> BoundingBox:
    spatial_extent = {
        "west": west,
        "east": east,
        "south": south,
        "north": north,
        "crs": crs,
    }
    return BoundingBox.parse_obj(spatial_extent)


@pytest.fixture
def polygon_geometry_small(
    west=10.47, east=10.48, south=46.12, north=46.18, crs="EPSG:4326"
):
    # Bounding box coordinates
    west, east, south, north = 10.47, 10.48, 46.12, 46.18

    # Create a small polygon
    geometry = [
        Polygon(
            [(west, south), (west, north), (east, north), (east, south), (west, south)]
        )
    ]

    # Create a GeoDataFrame with a single polygon and default CRS 'wgs84'
    gdf = gpd.GeoDataFrame(geometry=geometry, crs=crs)

    geometries = gdf.to_json()
    geometries = json.loads(geometries)

    return geometries


@pytest.fixture
def temporal_interval(interval=["2018-05-01", "2018-06-01"]) -> TemporalInterval:
    return TemporalInterval.parse_obj(interval)


@pytest.fixture
def process_registry() -> ProcessRegistry:
    registry = ProcessRegistry(wrap_funcs=[process])

    standard_processes = [
        func
        for _, func in inspect.getmembers(
            importlib.import_module("openeo_processes_dask.process_implementations"),
            inspect.isfunction,
        )
    ]

    specs_module = importlib.import_module("openeo_processes_dask.specs")

    specs = {}
    for func in standard_processes:
        if hasattr(specs_module, func.__name__):
            specs[func.__name__] = getattr(specs_module, func.__name__)
        else:
            logger.warning("Process {} does not have a spec.")
        registry[func.__name__] = Process(
            spec=specs[func.__name__] if func.__name__ in specs else None,
            implementation=func,
        )

    return registry


@pytest.fixture
def vector_data_cube() -> VectorCube:
    x = np.arange(-1683723, -1683723 + 10, 1)
    y = np.arange(6689139, 6689139 + 10, 1)
    df = GeoDataFrame(
        {"geometry": gpd.points_from_xy(x, y), "value1": x + y, "value2": x * y},
        crs=DEFAULT_CRS,
    )

    dask_obj = dask_geopandas.from_geopandas(df, npartitions=1)
    return dask_obj


@pytest.fixture
def geometry_point(
    x=10.47, y=46.12, crs="EPSG:4326"
):
    # Create a small polygon
    coordinates = [x, y]
    
    geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "0",
                "type": "Feature",
                "properties": {
                    "id": "0",
                    "class": 1
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": coordinates
                }
            }
        ]
    }

    return geometry



@pytest.fixture
def geometry_dict(
    west=10.47, east=10.48, south=46.12, north=46.18, crs="EPSG:4326"
):
    # Create a small polygon
    coordinates = [
        [[west, south], [west, north], [east, north], [east, south], [west, south]]
    ]
    
    geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "0",
                "type": "Feature",
                "properties": {
                    "id": "0",
                    "class": 1
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                }
            }
        ]
    }

    return geometry


@pytest.fixture
def wkt_string():
    return 'PROJCRS["WGS 84 / Pseudo-Mercator",BASEGEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],MEMBER["World Geodetic System 1984 (G2296)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]],CONVERSION["Popular Visualisation Pseudo-Mercator",METHOD["Popular Visualisation Pseudo Mercator",ID["EPSG",1024]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["easting (X)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["northing (Y)",north,ORDER[2],LENGTHUNIT["metre",1]],USAGE[SCOPE["Web mapping and visualisation."],AREA["World between 85.06°S and 85.06°N."],BBOX[-85.06,-180,85.06,180]],ID["EPSG",3857]]'