import logging
from typing import Union

import odc.geo
from odc.geo.geobox import resolution_from_affine
from pyproj.crs import CRS, CRSError

from openeo_processes_dask.process_implementations.data_model import RasterCube

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
    projection: Union[str, int] = None,
    resolution: Union[str, int] = None,
    method: str = "near",
    align: str = "upper-left",
):
    """Resamples the spatial dimensions (x,y) of the data cube to a specified resolution and/or warps the data cube to the target projection. At least resolution or projection must be specified."""

    if align == "upper-left":
        logging.warning("Warning: align parameter is unused by current implementation.")

    # This is necessary, because the resampling methods in rasterio are used
    # Reference here: https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
    if method == "near":
        method = "nearest"

    elif method not in resample_methods_list:
        raise Exception(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(resample_methods_list)}]"
        )

    # Re-order, this is specifically done for odc reproject
    data_cp = data.transpose("bands", "t", "y", "x")

    if not projection:
        projection = data_cp.rio.crs

    try:
        projection = CRS(projection)
    except CRSError as e:
        raise CRSError(f"{projection} Can not be parsed to CRS.")

    if not resolution:
        resolution = resolution_from_affine(data_cp.geobox.affine).x

    reprojected = data_cp.odc.reproject(
        how=projection, resolution=resolution, resampling=method
    )

    if "longitude" in reprojected.dims:
        reprojected = reprojected.rename({"longitude": "x"})

    if "latitude" in reprojected.dims:
        reprojected = reprojected.rename({"latitude": "y"})

    reprojected.attrs["crs"] = data_cp.rio.crs

    # Undo odc specific re-ordering.
    reprojected = reprojected.transpose(*data.dims)

    return reprojected
