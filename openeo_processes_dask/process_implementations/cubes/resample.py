from typing import Union

import odc.algo
from odc.geo.geobox import resolution_from_affine
from pyproj import Transformer
from pyproj.crs import CRS, CRSError

from openeo_processes_dask.process_implementations.cubes.utils import (
    detect_changing_unit,
    prepare_geobox,
)
from openeo_processes_dask.process_implementations.data_model import RasterCube

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
    epsg_code: Union[str, int] = None,
    resolution: int = None,
    method: str = "near",
    align: str = "upper-left",
):
    """Resample the input rastercube by the provided epsg_code. Currently not accepting resolution resampling."""

    # Assert resampling method is correct.
    if method == "near":
        method = "nearest"

    elif method not in resample_methods_list:
        raise Exception(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(resample_methods_list)}]"
        )

    # Re-order, this is specifically done for xr_reproject
    data = data.transpose("bands", "t", "y", "x")

    # Do reprojection first, and then resampling
    if epsg_code:
        try:
            dst_crs = CRS.from_epsg(epsg_code)
        except CRSError:
            raise Exception(
                f"epsg_code parameter {epsg_code} is not a valid epsg code."
            )

        # Get original crs and resolution from dataset.
        src_crs = CRS(data.rio.crs)

        transformer = Transformer.from_crs(
            src_crs,
            dst_crs,
            always_xy=True,
        )
        # For the affine, we want to know the x and y coordinate for the upper-left pixel.
        top_left_pixel = data.isel(x=[0], y=[0])

        # Transform the upper left pixel coords to the destination crs.
        new_x, new_y = transformer.transform(
            top_left_pixel.x.values, top_left_pixel.y.values
        )

        # If resolution has not been provided, it must be inferred from old resolution.
        # First value of affine is pixel width. Use it as "resolution".
        src_res = resolution_from_affine(data.affine).x
        dst_res = detect_changing_unit(
            src_crs=src_crs,
            dst_crs=dst_crs,
            # First value of affine is pixel width, which we use as 'resolution'.
            src_res=src_res,
        )

        dst_geobox = prepare_geobox(data, dst_crs, dst_res, src_res, new_x[0], new_y[0])
        data = odc.algo.xr_reproject(src=data, geobox=dst_geobox)
        data.attrs["crs"] = dst_crs

    # And if resampling was requested
    if resolution:
        dst_res = resolution
        # Get src_crs seperately incase reprojection was carried out.
        src_crs = CRS(data.rio.crs)
        # Get top left pixel seperately incase reprojection was carried out
        top_left_pixel = data.isel(x=[0], y=[0])
        src_res = resolution_from_affine(data.affine).x

        dst_geobox = prepare_geobox(
            data,
            src_crs,
            dst_res,
            src_res,
            top_left_pixel.x.values[0],
            top_left_pixel.y.values[0],
            scale=True,
        )

        data = odc.algo.xr_reproject(src=data, geobox=dst_geobox)

    reprojected = data

    if "longitude" in reprojected.dims:
        reprojected = reprojected.rename({"longitude": data.openeo.x_dim})

    if "latitude" in reprojected.dims:
        reprojected = reprojected.rename({"latitude": data.openeo.y_dim})

    reprojected.attrs["crs"] = data.rio.crs
    return reprojected
