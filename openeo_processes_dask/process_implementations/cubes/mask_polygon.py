import json
import logging
from typing import Any, Union

import dask.array as da
import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
import shapely
import xarray as xr
from xarray.core import dtypes

from openeo_processes_dask.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)

DEFAULT_CRS = "EPSG:4326"


logger = logging.getLogger(__name__)

__all__ = [
    "mask_polygon",
]


def mask_polygon(
    data: RasterCube,
    mask: Union[VectorCube, str],
    replacement: Any = dtypes.NA,
    inside: bool = True,
) -> RasterCube:
    y_dim = data.openeo.y_dim
    x_dim = data.openeo.x_dim
    t_dim = data.openeo.temporal_dims
    b_dim = data.openeo.band_dims

    if len(t_dim) == 0:
        t_dim = None
    else:
        t_dim = t_dim[0]
    if len(b_dim) == 0:
        b_dim = None
    else:
        b_dim = b_dim[0]

    y_dim_size = data.sizes[y_dim]
    x_dim_size = data.sizes[x_dim]

    #  Reproject vector data to match the raster data cube.
    ## Get the CRS of data cube
    try:
        data_crs = str(data.rio.crs)
    except Exception as e:
        raise Exception(f"Not possible to estimate the input data projection! {e}")

    data = data.rio.set_crs(data_crs)

    ## Reproject vector data if the input vector data is Polygon or Multi Polygon
    if "type" in mask and mask["type"] == "FeatureCollection":
        geometries = gpd.GeoDataFrame.from_features(mask, DEFAULT_CRS)
        geometries = geometries.to_crs(data_crs)
        geometries = geometries.to_json()
        geometries = json.loads(geometries)
    elif "type" in mask and mask["type"] in ["Polygon"]:
        polygon = shapely.geometry.Polygon(mask["coordinates"][0])
        geometries = gpd.GeoDataFrame(geometry=[polygon])
        geometries.crs = DEFAULT_CRS
        geometries = geometries.to_crs(data_crs)
        geometries = geometries.to_json()
        geometries = json.loads(geometries)
    else:
        raise ValueError(
            "Unsupported or missing geometry type. Expected 'Polygon' or 'FeatureCollection'."
        )

    data_dims = list(data.dims)

    # Get the Affine transformer
    transform = data.rio.transform()

    # Initialize an empty mask
    # Set the same chunk size as the input data
    data_chunks = {}
    chunks_shapes = data.chunks
    for i, d in enumerate(data_dims):
        data_chunks[d] = chunks_shapes[i][0]

    if data_dims.index(x_dim[0]) < data_dims.index(y_dim[0]):
        final_mask = da.zeros(
            (x_dim_size, y_dim_size),
            chunks={x_dim: data_chunks[x_dim], y_dim: data_chunks[y_dim]},
            dtype=bool,
        )

        dask_out_shape = da.from_array(
            (x_dim_size, y_dim_size),
            chunks={x_dim: data_chunks[x_dim], y_dim: data_chunks[y_dim]},
        )
    else:
        final_mask = da.zeros(
            (y_dim_size, x_dim_size),
            chunks={y_dim: data_chunks[y_dim], x_dim: data_chunks[x_dim]},
            dtype=bool,
        )

        dask_out_shape = da.from_array(
            (y_dim_size, x_dim_size),
            chunks={y_dim: data_chunks[y_dim], x_dim: data_chunks[x_dim]},
        )

    # CHECK IF the input single polygon or multiple Polygons
    if "type" in geometries and geometries["type"] == "FeatureCollection":
        for feature in geometries["features"]:
            polygon = shapely.geometry.Polygon(feature["geometry"]["coordinates"][0])

            # Create a GeoSeries from the geometry
            geo_series = gpd.GeoSeries(polygon)

            # Convert the GeoSeries to a GeometryArray
            geometry_array = geo_series.geometry.array

            mask = da.map_blocks(
                rasterio.features.geometry_mask,
                geometry_array,
                transform=transform,
                out_shape=dask_out_shape,
                dtype=bool,
                invert=inside,
            )
            final_mask |= mask

    elif "type" in geometries and geometries["type"] in ["Polygon"]:
        polygon = shapely.geometry.Polygon(geometries["coordinates"][0])
        geo_series = gpd.GeoSeries(polygon)

        # Convert the GeoSeries to a GeometryArray
        geometry_array = geo_series.geometry.array
        mask = da.map_blocks(
            rasterio.features.geometry_mask,
            geometry_array,
            transform=transform,
            out_shape=dask_out_shape,
            dtype=bool,
            invert=inside,
        )
        final_mask |= mask

    masked_dims = len(final_mask.shape)

    diff_axes = []
    for axis in range(len(data_dims)):
        try:
            if final_mask.shape[axis] != data.shape[axis]:
                diff_axes.append(axis)
        except:
            if len(diff_axes) < (len(data_dims) - 2):
                diff_axes.append(axis)

    final_mask = np.expand_dims(final_mask, axis=diff_axes)
    filtered_ds = data.where(final_mask, other=replacement)

    return filtered_ds
