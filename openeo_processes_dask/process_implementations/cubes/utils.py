try:
    import dask
except ImportError:
    dask = None
from enum import Enum
from math import pi

def _has_dask():
    return dask is not None


def _is_dask_array(arr):
    return _has_dask() and isinstance(arr, dask.array.Array)


class Unit(Enum):
    metre = 'metre'
    degree = 'degree'

def approx_metres_2_degrees(length_metres = 10):
    """ Angle of rotation calculated off the earths equatorial radius in metres. 
        Ref: https://en.ans.wiki/5729/how-to-convert-meters-to-degrees/ """
    earth_equator_radius_metres = 6378160
    return (180 * length_metres ) / (earth_equator_radius_metres * pi)

def approx_degrees_2_metres(angle_in_degrees = 0.0009):
    """ Calculated the metres travelled for an angle of rotation on the earths equatorial radius in metres.
        Ref: https://en.ans.wiki/5728/how-to-convert-degrees-to-meters/ """
    earth_equator_radius_metres = 6378160
    return (earth_equator_radius_metres * pi * angle_in_degrees) / 180

def detect_changing_unit(src_crs, dst_crs, src_res):
    """ Is there a unit change between the src and dst CRS. If so, return the approximate value for the new resolution. """

    src_unit = Unit(src_crs.axis_info[0].unit_name)
    dst_unit = Unit(dst_crs.axis_info[0].unit_name)

    if src_unit != dst_unit:
        if dst_unit == Unit.metre:
            return approx_degrees_2_metres(src_res)
        if dst_unit == Unit.degree:
            return approx_metres_2_degrees(src_res)
    return src_res