try:
    import dask
except ImportError:
    dask = None
import math

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


def approx_metres_2_degrees(length_metres=10):
    """Angle of rotation calculated off the earths equatorial radius in metres.
    Ref: https://en.ans.wiki/5729/how-to-convert-meters-to-degrees/"""
    earth_equator_radius_metres = 6378160
    return ((180 * length_metres) / (earth_equator_radius_metres * math.pi)) * 10
