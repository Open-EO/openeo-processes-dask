import logging
import warnings
from typing import Callable

import numpy as np
import pyproj
import rioxarray
import xarray as xr
import shapely
import rasterio
import dask.array as da
import geopandas as gpd
import json
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval
from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    BandFilterParameterMissing,
    DimensionMissing,
    DimensionNotAvailable,
    TooManyDimensions,
)

DEFAULT_CRS = 'EPSG:4326'


logger = logging.getLogger(__name__)

__all__ = ["filter_labels", "filter_temporal", "filter_bands", "filter_bbox", "filter_spatial"]


def filter_temporal(
    data: RasterCube, extent: TemporalInterval, dimension: str = None
) -> RasterCube:
    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        applicable_temporal_dimension = dimension
        if dimension not in temporal_dims:
            logger.warning(
                f"The selected dimension {dimension} exists but it is not labeled as a temporal dimension. Available temporal diemnsions are {temporal_dims}."
            )
    else:
        if not temporal_dims:
            raise DimensionNotAvailable(
                f"No temporal dimension detected on dataset. Available dimensions: {data.dims}"
            )
        if len(temporal_dims) > 1:
            raise TooManyDimensions(
                f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified."
            )
    applicable_temporal_dimension = temporal_dims[0]

    # This line raises a deprecation warning, which according to this thread
    # will never actually be deprecated:
    # https://github.com/numpy/numpy/issues/23904
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        start_time = extent[0]
        if start_time is not None:
            start_time = start_time.to_numpy()
        end_time = extent[1]
        if end_time is not None:
            end_time = extent[1].to_numpy() - np.timedelta64(1, "ms")
        # The second element is the end of the temporal interval.
        # The specified instance in time is excluded from the interval.
        # See https://processes.openeo.org/#filter_temporal

        data = data.where(~np.isnat(data[applicable_temporal_dimension]), drop=True)
        filtered = data.loc[
            {applicable_temporal_dimension: slice(start_time, end_time)}
        ]

    return filtered


def filter_labels(data: RasterCube, condition: Callable, dimension: str) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    labels = data[dimension].values
    label_mask = condition(x=labels)
    label = labels[label_mask]
    data = data.sel(**{dimension: label})
    return data


def filter_bands(data: RasterCube, bands: list[str] = None) -> RasterCube:
    if bands is None:
        raise BandFilterParameterMissing(
            "The process `filter_bands` requires the parameters `bands` to be set."
        )

    if len(data.openeo.band_dims) < 1:
        raise DimensionMissing("A band dimension is missing.")
    band_dim = data.openeo.band_dims[0]

    try:
        data = data.sel(**{band_dim: bands})
    except Exception as e:
        raise Exception(
            f"The provided bands: {bands} are not all available in the datacube. Please modify the bands parameter of filter_bands and choose among: {data[band_dim].values}."
        )
    return data



def filter_spatial(data: RasterCube, geometries)-> RasterCube:
    y_dim = data.openeo.y_dim
    x_dim = data.openeo.x_dim
    t_dim = data.openeo.temporal_dims
    b_dim = data.openeo.band_dims

    if len(t_dim)==0:
        t_dim = None
    else:
        t_dim = t_dim[0]
    if len(b_dim)==0:
        b_dim = None
    else:
        b_dim = b_dim[0]

    y_dim_size = data.sizes[y_dim]
    x_dim_size = data.sizes[x_dim]

    #  Reproject vector data to match the raster data cube.
    ## Get the CRS of data cube
    if "crs" in data.attrs:
        data_crs = data.attrs["crs"]
    else:
        raise Exception("Can't get the data projection to reproject the provided Vector Cube.")

    data = data.rio.set_crs(data_crs)
    
    ## Reproject vector data if the input vector data is Polygon or Multi Polygon
    if "type" in geometries and geometries["type"] == "FeatureCollection":
        geometries = gpd.GeoDataFrame.from_features(geometries, DEFAULT_CRS)
        geometries = geometries.to_crs(data_crs)
        geometries = geometries.to_json()
        geometries = json.loads(geometries)
    elif "type" in geometries and geometries["type"] in ["Polygon"]: 
        polygon = shapely.geometry.Polygon(geometries["coordinates"][0])
        geometries = gpd.GeoDataFrame(geometry=[polygon])
        geometries.crs = DEFAULT_CRS
        geometries = geometries.to_crs(data_crs)
        geometries = geometries.to_json()
        geometries = json.loads(geometries)
    else:
        raise ValueError("Unsupported or missing geometry type. Expected 'Polygon' or 'FeatureCollection'.")   

    data_dims = list(data.dims)

    # Get the Affine transformer
    transform = data.rio.transform()
    
    # Initialize an empty mask
    # Set the same chunk size as the input data
    data_chunks = {}
    chunks_shapes = data.chunks
    for i,d in enumerate(data_dims):
        data_chunks[d] = chunks_shapes[i][0]

    final_mask = da.zeros((y_dim_size, x_dim_size),
                          chunks={y_dim:data_chunks[y_dim],x_dim:data_chunks[x_dim]},
                          dtype=bool)
    
    dask_out_shape = da.from_array((y_dim_size, x_dim_size),
                                   chunks={y_dim:data_chunks[y_dim],x_dim:data_chunks[x_dim]})

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
                transform = transform,
                out_shape = dask_out_shape,
                dtype = bool,
                invert = True
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
            invert=True
        )
        final_mask |= mask    

    if t_dim is not None:
        final_mask = np.expand_dims(final_mask,axis=data_dims.index(t_dim))
    if b_dim is not None:
        final_mask = np.expand_dims(final_mask,axis=data_dims.index(b_dim))

    filtered_ds = data.where(final_mask)
    
    # Remove the pixels outside the polygons
    filtered_ds = filtered_ds.dropna(dim=y_dim, how="all").dropna(dim=x_dim, how="all")

    return filtered_ds


def filter_bbox(data: RasterCube, extent: BoundingBox) -> RasterCube:
    try:
        input_crs = str(data.rio.crs)
    except Exception as e:
        raise Exception(f"Not possible to estimate the input data projection! {e}")
    if not pyproj.crs.CRS(extent.crs).equals(input_crs):
        reprojected_extent = _reproject_bbox(extent, input_crs)
    else:
        reprojected_extent = extent
    y_dim = data.openeo.y_dim
    x_dim = data.openeo.x_dim

    # Check first if the data has some spatial dimensions:
    if y_dim is None and x_dim is None:
        raise DimensionNotAvailable(
            "No spatial dimensions available, can't apply filter_bbox."
        )

    if y_dim is not None:
        # Check if the coordinates are increasing or decreasing
        if len(data[y_dim]) > 1:
            if data[y_dim][0] > data[y_dim][1]:
                y_slice = slice(reprojected_extent.north, reprojected_extent.south)
            else:
                y_slice = slice(reprojected_extent.south, reprojected_extent.north)
        else:
            # We need to check if the bbox crosses this single coordinate
            # if data[y_dim][0] < reprojected_extent.north and data[y_dim][0] > reprojected_extent.south:
            #     # bbox crosses the single coordinate
            #     y_slice = data[y_dim][0]
            # else:
            #     # bbox doesn't cross the single coordinate: return empty data or error?
            raise NotImplementedError(
                f"filter_bbox can't filter data with a single coordinate on {y_dim} yet."
            )

    if x_dim is not None:
        if len(data[x_dim]) > 1:
            if data[x_dim][0] > data[x_dim][1]:
                x_slice = slice(reprojected_extent.east, reprojected_extent.west)
            else:
                x_slice = slice(reprojected_extent.west, reprojected_extent.east)
        else:
            # We need to check if the bbox crosses this single coordinate. How to do this correctly?
            # if data[x_dim][0] < reprojected_extent.east and data[x_dim][0] > reprojected_extent.west:
            #     # bbox crosses the single coordinate
            #     y_slice = data[x_dim][0]
            # else:
            #     # bbox doesn't cross the single coordinate: return empty data or error?
            raise NotImplementedError(
                f"filter_bbox can't filter data with a single coordinate on {x_dim} yet."
            )

    if y_dim is not None and x_dim is not None:
        aoi = data.loc[{y_dim: y_slice, x_dim: x_slice}]
    elif x_dim is None:
        aoi = data.loc[{y_dim: y_slice}]
    else:
        aoi = data.loc[{x_dim: x_slice}]

    return aoi


def _reproject_bbox(extent: BoundingBox, target_crs: str) -> BoundingBox:
    bbox_points = [
        [extent.south, extent.west],
        [extent.south, extent.east],
        [extent.north, extent.east],
        [extent.north, extent.west],
    ]
    if extent.crs is not None:
        source_crs = extent.crs
    else:
        source_crs = "EPSG:4326"

    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    x_reprojected = []
    y_reprojected = []
    for p in bbox_points:
        x1, y1 = p
        x2, y2 = transformer.transform(y1, x1)
        x_reprojected.append(x2)
        y_reprojected.append(y2)

    x_reprojected = np.array(x_reprojected)
    y_reprojected = np.array(y_reprojected)

    reprojected_extent = {}

    reprojected_extent = BoundingBox(
        west=x_reprojected.min(),
        east=x_reprojected.max(),
        north=y_reprojected.max(),
        south=y_reprojected.min(),
        crs=target_crs,
    )
    return reprojected_extent
