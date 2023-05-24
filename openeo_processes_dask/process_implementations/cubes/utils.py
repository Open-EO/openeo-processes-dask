try:
    import dask
except ImportError:
    dask = None
import math
from enum import Enum

from affine import Affine
from datacube.utils.geometry import GeoBox
from odc.geo.geobox import resolution_from_affine
from xarray.core.duck_array_ops import isnull as xr_isnull


def _has_dask():
    return dask is not None


def _is_dask_array(arr):
    return _has_dask() and isinstance(arr, dask.array.Array)


def isnull(data):
    if _is_dask_array(data):
        return dask.array.map_blocks(xr_isnull, data)
    else:
        return xr_isnull(data)


def notnull(data):
    return ~isnull(data)


class Unit(Enum):
    metre = "metre"
    degree = "degree"


def approx_metres_2_degrees(length_metres=10):
    """Angle of rotation calculated off the earths equatorial radius in metres.
    Ref: https://en.ans.wiki/5729/how-to-convert-meters-to-degrees/"""
    earth_equator_radius_metres = 6378160
    return ((180 * length_metres) / (earth_equator_radius_metres * math.pi)) * 10


def approx_degrees_2_metres(angle_in_degrees=0.0009):
    """Calculated the metres travelled for an angle of rotation on the earths equatorial radius in metres.
    Ref: https://en.ans.wiki/5728/how-to-convert-degrees-to-meters/"""
    earth_equator_radius_metres = 6378160
    return ((earth_equator_radius_metres * math.pi * angle_in_degrees) / 180) / 10


def detect_changing_unit(src_crs, dst_crs, src_res=None):
    """Is there a unit change between the src and dst CRS. If so, return the approximate value for the new resolution."""

    src_unit = Unit(src_crs.axis_info[0].unit_name)
    dst_unit = Unit(dst_crs.axis_info[0].unit_name)

    # If not dst_res has been set, check the src_unit and dst_unit to
    # determine whether the src_res needs to be converted.
    if src_unit != dst_unit:
        if dst_unit == Unit.metre:
            print("deg 2 met")
            return approx_degrees_2_metres(src_res)
        if dst_unit == Unit.degree:
            print("met 2 deg")
            return approx_metres_2_degrees(src_res)
    return src_res


def prepare_geobox(
    data, dst_crs, dst_res, src_res, new_top_left_x, new_top_left_y, scale=False
):
    """Get the destination geobox ready for the transformation."""
    # Docs for geotransform
    # https://gdal.org/tutorials/geotransforms_tut.html

    new_affine = Affine(
        dst_res,
        0,
        new_top_left_x,
        0,
        # Negative pixel width used for pixel height
        -dst_res,
        new_top_left_y,
    )

    if scale:
        updated_resolution = resolution_from_affine(new_affine)

        x_min, x_max = min(data.x.values), max(data.x.values)
        y_min, y_max = min(data.y.values), max(data.y.values)

        x_length = math.ceil(abs(x_min - x_max) / updated_resolution.x)
        y_length = math.ceil(abs(y_min - y_max) / -updated_resolution.y)

    else:
        x_length = len(data.x)
        y_length = len(data.y)

    dst_geo = GeoBox(width=x_length, height=y_length, crs=dst_crs, affine=new_affine)
    return dst_geo
