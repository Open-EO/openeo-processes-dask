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
        keep_attrs=True,
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

    # Store original dimension names to preserve semantic dimensions (fixes Issue #330)
    original_dims = list(data.dims)
    original_coords = {dim: data.coords[dim] for dim in data.dims if dim in data.coords}

    # This transpose (and back later) is needed because apply_ufunc automatically moves
    # input_core_dimensions to the last axes
    reordered_data = data.transpose(..., dimension)

    # ISSUE #330 FIX: Keep exclude_dims for functionality but restore semantic names after
    # The exclude_dims parameter is needed for dimension changes, but causes generic names
    # We'll restore semantic dimension names after apply_ufunc returns
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
        exclude_dims={
            dimension
        },  # Needed for dimension changes, but we fix names below
    )

    # ISSUE #330 FIX: Restore semantic dimension names that were converted to generic names
    # xarray's exclude_dims causes dimension names like 'time', 'x', 'y' to become 'dim_0', 'dim_1', 'dim_2'
    # We restore the original semantic names here
    if any(dim.startswith("dim_") for dim in result.dims):
        # Build mapping from generic to semantic dimension names
        dim_mapping = {}
        generic_dims = [d for d in result.dims if d.startswith("dim_")]

        # Map each generic dimension back to its semantic name
        for generic_dim in generic_dims:
            # Extract the dimension index from 'dim_0', 'dim_1', etc.
            dim_idx = int(generic_dim.split("_")[1])
            if dim_idx < len(original_dims):
                semantic_name = original_dims[dim_idx]
                # Only map if the semantic name isn't already in use
                if semantic_name not in result.dims or semantic_name == generic_dim:
                    dim_mapping[generic_dim] = semantic_name

        if dim_mapping:
            result = result.rename(dim_mapping)

            # Restore original coordinates for renamed dimensions
            for generic_dim, semantic_dim in dim_mapping.items():
                if semantic_dim in original_coords and semantic_dim in result.dims:
                    if semantic_dim not in result.coords or len(
                        result.coords[semantic_dim]
                    ) != len(original_coords[semantic_dim]):
                        try:
                            result = result.assign_coords(
                                {semantic_dim: original_coords[semantic_dim]}
                            )
                        except (ValueError, KeyError):
                            # Coordinate assignment might fail if dimensions changed, that's OK
                            pass

    # Restore original dimension order
    reordered_result = result.transpose(
        *[d for d in data.dims if d in result.dims], ...
    )

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
