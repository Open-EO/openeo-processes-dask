import logging
from typing import Optional, Union

import odc.geo.xr
from odc.geo.geobox import resolution_from_affine
from pyproj.crs import CRS, CRSError

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import OpenEOException

logger = logging.getLogger(__name__)

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
):
    """Resamples the spatial dimensions (x,y) of the data cube to a specified resolution and/or warps the data cube to the target projection. At least resolution or projection must be specified."""

    # Assert resampling method is correct.
    if method == "near":
        method = "nearest"

    elif method not in resample_methods_list:
        raise OpenEOException(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(resample_methods_list)}]"
        )

    # Re-order, this is specifically done for odc reproject
    data_cp = data.transpose(
        data.openeo.band_dims[0],
        data.openeo.temporal_dims[0],
        data.openeo.y_dim,
        data.openeo.x_dim,
    )

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

    if "longitude" in reprojected.dims and "x" in data.dims:
        reprojected = reprojected.rename({"longitude": "x"})

    if "latitude" in reprojected.dims and "y" in data.dims:
        reprojected = reprojected.rename({"latitude": "y"})

    reprojected.attrs["crs"] = data_cp.rio.crs

    return reprojected
