import logging
from functools import wraps

try:
    import dask

except ImportError:
    dask = None

from openeo_processes_dask.process_implementations.data_model import RasterCube

logger = logging.getLogger(__name__)


def _normalise_output_datacube(data) -> RasterCube:
    # Order dimensions for rioxarray
    data = data.transpose("bands", "t", "z", "y", "x", missing_dims="ignore")

    if "origin" not in data.attrs.keys():
        data.attrs["origin"] = "odc"

    if not hasattr(data, "crs"):
        if hasattr(data, "rio"):
            data.attrs["crs"] = data.rio.crs
        else:
            raise AttributeError(
                "No CRS could be determined for gridding processing output."
            )

    return data


def normalise_output_datacube(f):
    """Decorator to ensure certain properties on Rastercubes."""

    @wraps(f)
    def wrapper(*args, **kwargs) -> RasterCube:
        data = f(*args, **kwargs)  # type: RasterCube
        return _normalise_output_datacube(data)

    return wrapper


def _has_dask():
    return dask is not None


def _is_dask_array(arr):
    return _has_dask() and isinstance(arr, dask.array.Array)
