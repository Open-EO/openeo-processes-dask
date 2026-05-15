import dask.array as da
import numpy as np
import xarray as xr


def get_scalar_type(obj):
    if np.isscalar(obj):
        return np.dtype(type(obj)).type
    if hasattr(obj, "dtype"):
        return np.dtype(obj.dtype).type
    return np.object_
