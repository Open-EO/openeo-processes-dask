import logging
from typing import Optional, Union

import numpy as np
import odc.geo.xr
import rioxarray  # needs to be imported to set .rio accessor on xarray objects.
import xarray as xr
from odc.geo.geobox import resolution_from_affine
from pyproj.crs import CRS, CRSError

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionMissing,
    OpenEOException,
)

logger = logging.getLogger(__name__)

__all__ = ["resample_spatial", "resample_cube_spatial", "resample_cube_temporal"]

resample_methods_list = [
    "near",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
]


def resample_spatial(
    data: RasterCube,
    projection: Optional[Union[str, int]] = None,
    resolution: int = 0,
    method: str = "near",
    align: str = "upper-left",
):
    """Resamples the spatial dimensions (x,y) of the data cube to a specified resolution and/or warps the data cube to the target projection. At least resolution or projection must be specified."""

    if data.openeo.y_dim is None or data.openeo.x_dim is None:
        raise DimensionMissing(f"Spatial dimension missing for dataset: {data} ")

    methods_list = [
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
    ]

    if method not in methods_list:
        raise Exception(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(methods_list)}]"
        )

    # Assert resampling method is correct.
    if method == "near":
        method = "nearest"

    elif method not in resample_methods_list:
        raise OpenEOException(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(resample_methods_list)}]"
        )

    dim_order = data.dims

    data_cp = data.transpose(..., data.openeo.y_dim, data.openeo.x_dim)

    if projection is None:
        projection = data_cp.rio.crs

    try:
        projection = CRS.from_user_input(projection)
    except CRSError as e:
        raise CRSError(
            f"Provided projection string: '{projection}' can not be parsed to CRS."
        ) from e

    if resolution == 0:
        resolution = resolution_from_affine(data_cp.odc.geobox.affine).x

    reprojected = data_cp.odc.reproject(
        how=projection, resolution=resolution, resampling=method
    )

    if reprojected.openeo.x_dim != data.openeo.x_dim:
        reprojected = reprojected.rename({reprojected.openeo.x_dim: data.openeo.x_dim})

    if reprojected.openeo.y_dim != data.openeo.y_dim:
        reprojected = reprojected.rename({reprojected.openeo.y_dim: data.openeo.y_dim})

    reprojected = reprojected.transpose(*dim_order)

    reprojected.attrs["crs"] = data_cp.rio.crs

    return reprojected


def resample_cube_spatial(
    data: RasterCube, target: RasterCube, method="near", options=None
) -> RasterCube:
    if target.openeo.y_dim is None or target.openeo.x_dim is None:
        raise DimensionMissing(
            f"Spatial dimension missing for target dataset: {target} "
        )

    target_resolution, target_crs = None, None
    if hasattr(target, "rio"):
        if hasattr(target.rio, "resolution"):
            if type(target.rio.resolution()) in [tuple, list]:
                target_resolution = target.rio.resolution()[0]
            else:
                target_resolution = target.rio.resolution()
        if hasattr(target.rio, "crs"):
            target_crs = target.rio.crs
    if not target_crs:
        raise OpenEOException(f"Projection not found in target dataset: {target} ")
    if not target_resolution:
        raise OpenEOException(f"Resolution not found in target dataset: {target} ")

    resampled_data = resample_spatial(
        data=data, projection=target_crs, resolution=target_resolution, method=method
    )

    return resampled_data


def resample_cube_temporal(data, target, dimension=None, valid_within=None):
    if dimension is None:
        if len(data.openeo.temporal_dims) > 0:
            dimension = data.openeo.temporal_dims[0]
        else:
            raise Exception("DimensionNotAvailable")
    if dimension not in data.dims:
        raise Exception("DimensionNotAvailable")
    if dimension not in target.dims:
        if len(target.openeo.temporal_dims) > 0:
            target_time = target.openeo.temporal_dims[0]
        else:
            raise Exception("DimensionNotAvailable")
        target = target.rename({target_time: dimension})
    index = []
    for d in target[dimension].values:
        difference = np.abs(d - data[dimension].values)
        nearest = np.argwhere(difference == np.min(difference))
        # The rare case of ties is resolved by choosing the earlier timestamps. (index 0)
        if np.shape(nearest) == (2, 1):
            nearest = nearest[0]
        if np.shape(nearest) == (1, 2):
            nearest = nearest[:, 0]
        index.append(int(nearest))
    times_at_target_time = data[dimension].values[index]
    new_data = data.loc[{dimension: times_at_target_time}]
    filter_values = new_data[dimension].values
    new_data[dimension] = target[dimension].values
    # valid_within
    if valid_within is None:
        new_data = new_data
    else:
        minimum = np.timedelta64(valid_within, "D")
        filter_valid = np.abs(filter_values - new_data[dimension].values) <= minimum
        times_valid = new_data[dimension].values[filter_valid]
        valid_data = new_data.loc[{dimension: times_valid}]
        filter_nan = np.abs(filter_values - new_data[dimension].values) > minimum
        times_nan = new_data[dimension].values[filter_nan]
        nan_data = new_data.loc[{dimension: times_nan}] * np.nan
        combined = xr.concat([valid_data, nan_data], dim=dimension)
        new_data = combined.sortby(dimension)
    new_data.attrs = data.attrs
    return new_data
