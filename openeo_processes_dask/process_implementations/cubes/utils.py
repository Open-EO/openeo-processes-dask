try:
    import dask

except ImportError:
    dask = None


def _has_dask():
    return dask is not None


def _is_dask_array(arr):
    return _has_dask() and isinstance(arr, dask.array.Array)
