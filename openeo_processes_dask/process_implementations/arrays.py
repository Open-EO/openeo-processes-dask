from typing import Union, Optional, Callable
import xarray as xr

from openeo_processes_dask.exceptions import (
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
)
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["array_element", "array_filter", "count"]


def array_element(
    data: Union[xr.Dataset, xr.DataArray],
    index: Optional[int] = None,
    label: Optional[str] = None,
    return_nodata: Optional[bool] = False,
    dimension: Optional[str] = None,
    **kwargs
):

    if index is None and label is None:
        raise ArrayElementParameterMissing(
            "The process `array_element` requires either the `index` or `labels` parameter to be set."
        )

    if index is not None and label is not None:
        raise ArrayElementParameterConflict(
            "The process `array_element` only allows that either the `index` or the `labels` parameter is set."
        )

    if label is not None:
        element = data.sel({dimension: label})
        return element

    if index is not None:
        element = data.isel({dimension: int(index)})
        return element

    raise ValueError("Shouldn't have come here!")


def array_filter(data: RasterCube, condition: Callable, **kwargs):
    mask = condition(x=data, **kwargs)
    data = data[mask]
    return data


def count(data: RasterCube, condition: Callable, **kwargs):
    data = condition(x=data, **kwargs)
    if "dimension" in kwargs:
        if kwargs["dimension"] == "t":
            kwargs["dimension"] = "time"
        data = data.sum(dim=kwargs["dimension"])
    else:
        data = data.sum()
    return data
