"""
Native UDF implementation for openeo-processes-dask.

This module replaces the previous openeo-python-client dependency
with a native implementation that preserves dimension names (fixes Issue #330).
"""

from typing import Optional

import dask.array as da
import xarray as xr

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.udf.native_udf import (
    run_udf as native_run_udf
)

__all__ = ["run_udf"]


def run_udf(
    data: da.Array, udf: str, runtime: str, context: Optional[dict] = None
) -> RasterCube:
    """
    Execute UDF code on the provided data.
    
    This implementation fixes Issue #330 by preserving semantic dimension names
    (e.g., 'time', 'x', 'y') instead of converting them to generic names
    (e.g., 'dim_0', 'dim_1', 'dim_2').
    
    Args:
        data: Input dask array data
        udf: UDF code string containing apply_datacube or apply_hypercube function
        runtime: Runtime environment ("Python" supported)
        context: Optional context dictionary passed to UDF
        
    Returns:
        RasterCube with preserved dimension names
        
    Raises:
        ValueError: If runtime is not supported
        UdfExecutionError: If UDF execution fails
    """
    # Use native implementation (no openeo-python-client dependency!)
    result = native_run_udf(data, udf, runtime, context)
    return result
