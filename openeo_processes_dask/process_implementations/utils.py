import dask.array as da
import numpy as np
import xarray as xr


def get_scalar_type(obj):
    if np.isscalar(obj):
        return np.obj2sctype(type(obj))
    if hasattr(obj, "dtype"):
        return obj.dtype
    return np.object_
