import copy
import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
import xvec

from openeo_processes_dask.process_implementations.data_model import VectorCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
    UnitMismatch,
)

__all__ = ["load_geojson", "vector_buffer", "vector_reproject"]

logger = logging.getLogger(__name__)


def load_geojson(data: dict, properties: Optional[list[str]] = []) -> VectorCube:
    DEFAULT_CRS = "epsg:4326"

    if isinstance(data, dict):
        # Get crs from geometries
        if "features" in data:
            for feature in data["features"]:
                if "properties" not in feature:
                    feature["properties"] = {}
                elif feature["properties"] is None:
                    feature["properties"] = {}
            if isinstance(data.get("crs", {}), dict):
                DEFAULT_CRS = (
                    data.get("crs", {}).get("properties", {}).get("name", DEFAULT_CRS)
                )
            else:
                DEFAULT_CRS = int(data.get("crs", {}))
            logger.info(f"CRS in geometries: {DEFAULT_CRS}.")

        if "type" in data and data["type"] == "FeatureCollection":
            gdf = gpd.GeoDataFrame.from_features(data, crs=DEFAULT_CRS)
        elif "type" in data and data["type"] in ["Polygon"]:
            polygon = shapely.geometry.Polygon(data["coordinates"][0])
            gdf = gpd.GeoDataFrame(geometry=[polygon])
            gdf.crs = DEFAULT_CRS

    dimensions = ["geometry"]
    coordinates = {"geometry": gdf.geometry}

    if len(properties) == 0:
        if "features" in data:
            feature = data["features"][0]
            if "properties" in feature:
                property = feature["properties"]
                if len(property) == 1:
                    key = list(property.keys())[0]
                    value = list(property.values())
                    dimensions.append("properties")
                    if isinstance(value, list) and len(value) > 1:
                        values = np.zeros((len(gdf.geometry), len(value)))
                        coordinates["properties"] = np.arange(len(value))
                    elif isinstance(value, list) and len(value) == 1:
                        values = np.zeros((len(gdf.geometry), 1))
                        coordinates["properties"] = np.array([key])
                    else:
                        values = np.zeros((len(gdf.geometry), 1))
                        coordinates["properties"] = np.array([key])

                    for i, feature in enumerate(data["features"]):
                        value = feature.get("properties", {}).get(key, None)
                        values[i, :] = value
                elif len(property) > 1:
                    dimensions.append("properties")
                    keys = list(property.keys())
                    coordinates["properties"] = keys
                    values = np.zeros((len(gdf.geometry), len(keys)))
                    for i, feature in enumerate(data["features"]):
                        for j, key in enumerate(keys):
                            value = feature.get("properties", {}).get(key, None)
                            values[i, j] = value

    elif len(properties) == 1:
        property = properties[0]
        if "features" in data:
            feature = data["features"][0]
            if "properties" in feature:
                if property in feature["properties"]:
                    value = feature["properties"][property]
                    dimensions.append("properties")
                    if isinstance(value, list) and len(value) > 0:
                        values = np.zeros((len(gdf.geometry), len(value)))
                        coordinates["properties"] = np.arange(len(value))
                    elif isinstance(value, list) and len(value) == 1:
                        values = np.zeros((len(gdf.geometry), 1))
                        coordinates["properties"] = np.array([property])
                    else:
                        values = np.zeros((len(gdf.geometry), 1))
                        coordinates["properties"] = np.array([property])

                    for i, feature in enumerate(data["features"]):
                        value = feature.get("properties", {}).get(property, None)
                        values[i, :] = value
    else:
        if "features" in data:
            dimensions.append("properties")
            coordinates["properties"] = properties
            values = np.zeros((len(gdf.geometry), len(properties)))
            for i, feature in enumerate(data["features"]):
                for j, key in enumerate(properties):
                    value = feature.get("properties", {}).get(key, None)
                    values[i, j] = value

    output_vector_cube = xr.DataArray(values, coords=coordinates, dims=dimensions)
    output_vector_cube = output_vector_cube.xvec.set_geom_indexes(
        "geometry", crs=gdf.crs
    )
    return output_vector_cube


def vector_buffer(geometries: VectorCube, distance: float) -> VectorCube:
    from shapely import buffer

    geometries_copy = copy.deepcopy(geometries)

    if isinstance(geometries_copy, xr.DataArray) and "geometry" in geometries_copy.dims:
        if hasattr(geometries_copy, "xvec") and hasattr(
            geometries_copy["geometry"], "crs"
        ):
            if geometries_copy["geometry"].crs.is_geographic:
                raise UnitMismatch(
                    "The unit of the spatial reference system is not meters, but the given distance is in meters."
                )

        geometry = geometries_copy["geometry"].values.tolist()

        new_geometry = [buffer(geom, distance) for geom in geometry]

        geometries_copy["geometry"] = new_geometry

        return geometries_copy

    else:
        raise DimensionNotAvailable(f"No geometry dimension found in {geometries}")


def vector_reproject(
    data: VectorCube, projection, dimension: Optional[str] = None
) -> VectorCube:
    DEFAULT_CRS = "epsg:4326"

    data_copy = copy.deepcopy(data)

    if not dimension:
        dimension = "geometry"

    if isinstance(data, xr.DataArray) and dimension in data.dims:
        if hasattr(data, "xvec") and hasattr(data[dimension], "crs"):
            data_copy = data_copy.xvec.to_crs({dimension: projection})

            return data_copy
        else:
            data_copy = data_copy.xvec.set_geom_indexes(dimension, crs=DEFAULT_CRS)
            data_copy = data_copy.xvec.to_crs({dimension: projection})

            return data_copy
    else:
        raise DimensionNotAvailable(f"No geometry dimension found in {data}")
