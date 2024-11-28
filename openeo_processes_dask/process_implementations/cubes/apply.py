from typing import Callable, Optional, Union

import numpy as np
import scipy.ndimage
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from openeo_processes_dask.process_implementations.cubes.mask_polygon import (
    mask_polygon,
)
from openeo_processes_dask.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
    KernelDimensionsUneven,
)

__all__ = ["apply", "apply_dimension", "apply_kernel"]


def apply(
    data: RasterCube, process: Callable, context: Optional[dict] = None
) -> RasterCube:
    positional_parameters = {"x": 0}
    named_parameters = {"context": context}
    result = xr.apply_ufunc(
        process,
        data,
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
        },
    )
    return result


def apply_dimension(
    data: RasterCube,
    process: Callable,
    dimension: str,
    target_dimension: Optional[str] = None,
    context: Optional[dict] = None,
) -> RasterCube:
    if context is None:
        context = {}

    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    keepdims = False
    is_new_dim_added = target_dimension is not None
    if is_new_dim_added:
        keepdims = True

    if target_dimension is None:
        target_dimension = dimension

    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    # This transpose (and back later) is needed because apply_ufunc automatically moves
    # input_core_dimensions to the last axes
    reordered_data = data.transpose(..., dimension)

    result = xr.apply_ufunc(
        process,
        reordered_data,
        input_core_dims=[[dimension]],
        output_core_dims=[[dimension]],
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
            "axis": reordered_data.get_axis_num(dimension),
            "keepdims": keepdims,
            "source_transposed_axis": data.get_axis_num(dimension),
            "context": context,
        },
        exclude_dims={dimension},
    )

    reordered_result = result.transpose(*data.dims, ...)

    if dimension in reordered_result.dims:
        result_len = len(reordered_result[dimension])
    else:
        result_len = 1

    # Case 1: target_dimension is not defined/ is source dimension
    if dimension == target_dimension:
        # dimension labels preserved
        # if the number of source dimension's values is equal to the number of computed values
        if len(reordered_data[dimension]) == result_len:
            reordered_result[dimension] == reordered_data[dimension].values
        else:
            reordered_result[dimension] = np.arange(result_len)
    elif target_dimension in reordered_result.dims:
        # source dimension is not target dimension
        # target dimension exists with a single label only
        if len(reordered_result[target_dimension]) == 1:
            reordered_result = reordered_result.drop_vars(target_dimension).squeeze(
                target_dimension
            )
            reordered_result = reordered_result.rename({dimension: target_dimension})
            reordered_result[dimension] = np.arange(result_len)
        else:
            raise Exception(
                f"Cannot rename dimension {dimension} to {target_dimension} as {target_dimension} already exists in dataset and contains more than one label: {reordered_result[target_dimension]}. See process definition. "
            )
    else:
        # source dimension is not the target dimension and the latter does not exist
        reordered_result = reordered_result.rename({dimension: target_dimension})
        reordered_result[target_dimension] = np.arange(result_len)

    if data.rio.crs is not None:
        try:
            reordered_result.rio.write_crs(data.rio.crs, inplace=True)
        except ValueError:
            pass

    return reordered_result


def apply_kernel(
    data: RasterCube,
    kernel: np.ndarray,
    factor: Optional[float] = 1,
    border: Union[float, str, None] = 0,
    replace_invalid: Optional[float] = 0,
) -> RasterCube:
    kernel = np.asarray(kernel)
    if any(dim % 2 == 0 for dim in kernel.shape):
        raise KernelDimensionsUneven(
            "Each dimension of the kernel must have an uneven number of elements."
        )

    def convolve(data, kernel, mode="constant", cval=0, fill_value=0):
        dims = data.openeo.spatial_dims
        convolved = lambda data: scipy.ndimage.convolve(
            data, kernel, mode=mode, cval=cval
        )

        data_masked = data.fillna(fill_value)

        return xr.apply_ufunc(
            convolved,
            data_masked,
            vectorize=True,
            dask="parallelized",
            input_core_dims=[dims],
            output_core_dims=[dims],
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        ).transpose(*data.dims)

    openeo_scipy_modes = {
        "replicate": "nearest",
        "reflect": "reflect",
        "reflect_pixel": "mirror",
        "wrap": "wrap",
    }
    if isinstance(border, int) or isinstance(border, float):
        mode = "constant"
        cval = border
    else:
        mode = openeo_scipy_modes[border]
        cval = 0

    return convolve(data, kernel, mode, cval, replace_invalid) * factor


def apply_polygon(
    data: RasterCube,
    polygons: Union[VectorCube, dict],
    process: Callable,
    mask_value: Optional[Union[int, float, str, None]] = None,
    context: Optional[dict] = None,
) -> RasterCube:
    if isinstance(polygons, dict) and polygons.get("type") == "FeatureCollection":
        polygon_geometries = [
            shape(feature["geometry"]) for feature in polygons["features"]
        ]
    elif isinstance(polygons, dict) and polygons.get("type") in [
        "Polygon",
        "MultiPolygon",
    ]:
        polygon_geometries = [shape(polygons)]
    else:
        raise ValueError(
            "Unsupported polygons format. Expected GeoJSON-like FeatureCollection or Polygon."
        )

    unified_polygon = unary_union(polygon_geometries)

    if isinstance(unified_polygon, MultiPolygon) and len(unified_polygon.geoms) < len(
        polygon_geometries
    ):
        raise Exception("GeometriesOverlap")

    masked_data = mask_polygon(data, polygons, replacement=np.nan)

    processed_data = apply(masked_data, process, context=context)

    result = mask_polygon(processed_data, polygons, replacement=mask_value)

    return result
