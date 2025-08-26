from typing import Optional, Union

import dask.array as da
import xarray as xr
from openeo.udf import UdfData
from openeo.udf.run_code import run_udf_code
from openeo.udf.xarraydatacube import XarrayDataCube

from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["run_udf"]


def run_udf(
    data: Union[RasterCube, da.Array], udf: str, runtime: str, context: Optional[dict] = None
) -> RasterCube:
    # Preserve dimension names and coordinates if input is already an xr.DataArray
    if isinstance(data, xr.DataArray):
        # Input is already a proper xr.DataArray (RasterCube), preserve its structure
        data_cube = XarrayDataCube(data)
    else:
        # Input is a dask/numpy array, convert to xr.DataArray (will have generic dims)
        data_cube = XarrayDataCube(xr.DataArray(data))
    
    udf_data = UdfData(datacube_list=[data_cube], user_context=context)
    result = run_udf_code(code=udf, data=udf_data)
    cubes = result.get_datacube_list()
    if len(cubes) != 1:
        raise ValueError(
            f"The provided UDF should return one datacube, but got: {result}"
        )
    result_array: xr.DataArray = cubes[0].array
    return result_array
