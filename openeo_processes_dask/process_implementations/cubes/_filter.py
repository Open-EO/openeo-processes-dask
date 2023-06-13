import logging
from typing import Callable

import numpy as np
import rioxarray
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval
from pyproj import CRS, Proj, Transformer, transform

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    BandFilterParameterMissing,
    DimensionMissing,
    DimensionNotAvailable,
    TooManyDimensions,
)

logger = logging.getLogger(__name__)

__all__ = ["filter_labels", "filter_temporal", "filter_bands", "filter_bbox"]


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

    start_time = extent[0]
    if start_time is not None:
        start_time = start_time.to_numpy()
    end_time = extent[1]
    if end_time is not None:
        end_time = extent[1].to_numpy() - np.timedelta64(1, "ms")
    # The second element is the end of the temporal interval.
    # The specified instance in time is excluded from the interval.
    # See https://processes.openeo.org/#filter_temporal

    filtered = data.loc[{applicable_temporal_dimension: slice(start_time, end_time)}]

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


def filter_bbox(data: RasterCube, extent: BoundingBox) -> RasterCube:
    try:
        input_crs = str(data.rio.crs)
    except Exception as e:
        raise Exception(f"Not possible to estimate the input data projection! {e}")
    trasnformed_extent = reproject_bbox(extent, input_crs)

    # Check if the coordinates are increasing or decreasing
    if len(data.y) > 1:
        if data.y[0] > data.y[1]:
            y_slice = slice(trasnformed_extent.north, trasnformed_extent.south)
        else:
            y_slice = slice(trasnformed_extent.south, trasnformed_extent.north)
    if len(data.x) > 1:
        if data.x[0] > data.x[1]:
            x_slice = slice(trasnformed_extent.east, trasnformed_extent.west)
        else:
            x_slice = slice(trasnformed_extent.west, trasnformed_extent.east)

    aoi = data.loc[{"y": y_slice, "x": x_slice}]
    return aoi


def reproject_bbox(extent: BoundingBox, output_crs: str) -> BoundingBox:
    if (
        extent is not None
        and extent.south is not None
        and extent.west is not None
        and extent.north is not None
        and extent.east is not None
    ):
        bbox_points = [
            [extent.south, extent.west],
            [extent.south, extent.east],
            [extent.north, extent.east],
            [extent.north, extent.west],
        ]
    else:
        raise Exception(f"Empty or non-valid bounding box provided! {extent}")
        return
    if extent.crs is not None:
        source_crs = extent.crs
    else:
        source_crs = "EPSG:4326"

    transformer = Transformer.from_crs(source_crs, output_crs, always_xy=True)

    x_t = []
    y_t = []
    for p in bbox_points:
        x1, y1 = p
        x2, y2 = transformer.transform(y1, x1)
        x_t.append(x2)
        y_t.append(y2)

    x_t = np.array(x_t)
    y_t = np.array(y_t)

    reprojected_extent = {}

    reprojected_extent = BoundingBox(
        west=x_t.min(), east=x_t.max(), north=y_t.max(), south=y_t.min(), crs=output_crs
    )
    return reprojected_extent
