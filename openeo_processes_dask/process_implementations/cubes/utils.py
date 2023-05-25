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
