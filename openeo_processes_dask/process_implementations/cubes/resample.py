import odc.algo

from affine import Affine
from enum import Enum
from datacube.utils.geometry import GeoBox
from math import pi
from pyproj import Transformer
from pyproj.crs import CRS, CRSError

from typing import Union, Optional

from openeo_processes_dask.process_implementations.cubes.utils import detect_changing_unit
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["resample_spatial"]


def resample_spatial(
        data: RasterCube,
        epsg_code: Union[str, int],
        resolution: int = 0,
        method: str = 'near',
        align: str = "upper-left"):
    """ Resample the input rastercube by the provided epsg_code. Currently not accepting resolution resampling. """
    # Re-order
    data = data.transpose("bands", "t", "y", "x")

    src_crs = CRS(data.rio.crs)

    try:
        dst_crs = CRS.from_epsg(epsg_code)
    except CRSError:
        raise Exception(f"epsg_code parameter {epsg_code} is not a valid epsg code.")

    transformer = Transformer.from_crs(
        src_crs, 
        dst_crs,
        always_xy=True,
    )

    # For the affine, we want to know the x and y coordinate for the upper-left pixel.
    tmp_left_pixel = data.isel(
        x=[0],
        y=[0]
    )
    tmp_left_pixel.x.values, tmp_left_pixel.y.values
    # Transform the upper left pixel coords to the destination crs.
    new_x, new_y = transformer.transform(
        tmp_left_pixel.x.values,
        tmp_left_pixel.y.values
    )

    resolution = detect_changing_unit(
        src_crs=src_crs,
        dst_crs= dst_crs,
        # First value of affine is pixel width.
        src_res=data.affine[0]
    )

    # Docs for geotransform
    # https://gdal.org/tutorials/geotransforms_tut.html
    new_affine = Affine(
        resolution,
        0,
        new_x[0],
        0,
        # Negative pixel width used for pixel height
        -resolution,
        new_y[0]
    )

    dst_geo = GeoBox(
        width=len(data.x),
        height=len(data.y),
        crs=dst_crs,
        affine=new_affine
    )

    reprojected =  odc.algo.xr_reproject(
        src=data,
        geobox=dst_geo,
    )
    if "longitude" in reprojected.dims:
        reprojected = reprojected.rename(
            {"longitude": data.openeo.x_dim}
        )

    if "latitude" in reprojected.dims:
        reprojected = reprojected.rename(
            {"latitude": data.openeo.y_dim}
        )  
    
    reprojected.attrs["crs"] = dst_crs
    return reprojected