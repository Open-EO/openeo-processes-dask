from functools import wraps
from openeo_processes_dask.process_implementations.data_model import RasterCube
import logging


logger = logging.getLogger(__name__)

RENAME_DIMS =  {"time": "t", "Time": "t", "Bands": "bands", "band": "bands", "latitude": "x", "longitude": "y", "lat": "x", "lon": "y"}


def _normalise_output_datacube(data) -> RasterCube:
    # Rename time dimension
    for old_dim_name, new_dim_name in RENAME_DIMS.items():
        try:
            data = data.rename(new_name_or_name_dict={old_dim_name: new_dim_name})
        except ValueError as e:
            pass

    # Order dimensions for rioxarray
    data = data.transpose('bands', 't', 'z', 'y', 'x', missing_dims="ignore")

    if 'origin' not in data.attrs.keys():
        data.attrs['origin'] = 'odc'

    if not hasattr(data, 'crs'):
        if hasattr(data, 'rio'):
            data.attrs['crs'] = data.rio.crs
        else:
            raise AttributeError("No CRS could be determined for gridding processing output.")

    return data

def normalise_output_datacube(f):
    """Decorator to ensure certain properties on Rastercubes."""
    @wraps(f)
    def wrapper(*args, **kwargs) -> RasterCube:
        data = f(*args, **kwargs)  # type: RasterCube
        return _normalise_output_datacube(data)
    return wrapper