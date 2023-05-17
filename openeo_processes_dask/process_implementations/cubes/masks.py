import logging
from typing import Callable

from openeo_processes_dask.process_implementations.exceptions import (
    DimensionLabelCountMismatch,
    DimensionMismatch,
    LabelMismatch,
)

logger = logging.getLogger(__name__)

__all__ = ["mask"]


def mask(data: RasterCube, mask: RasterCube, replacement=None) -> RasterCube:
    # Check if spatial dimensions have the same name
    data_spatial_dims = data.openeo.spatial_dims
    mask_spatial_dims = mask.openeo.spatial_dims
    if not set(data_spatial_dims) == set(mask_spatial_dims):
        raise DimensionMismatch(
            f"data and mask spatial dimensions do no match: data has spatial dimensions ({data_spatial_dims}) and mask {mask_spatial_dims}"
        )
    # Check if spatial labels correspond
    for n in data.openeo.spatial_dims:
        data_spatial_labels = data[n].values
        mask_spatial_labels = mask[n].values
        data_n_labels = len(data_spatial_labels)
        mask_n_labels = len(mask_spatial_labels)

        if not data_n_labels == mask_n_labels:
            raise DimensionLabelCountMismatch(
                f"data and mask spatial dimension labels don't match: data has ({data_n_labels}) labels and mask has {mask_n_labels} for dimension {n}."
            )
        if not all(data_spatial_labels == mask_spatial_labels):
            raise LabelMismatch(
                f"data and mask spatial dimension labels don't match for dimension {n}, i.e. the coordinate values are different."
            )
    # Check if temporal dimensions are present and check the names
    data_temporal_dims = data.openeo.temporal_dims
    mask_temporal_dims = mask.openeo.temporal_dims

    if not set(data_temporal_dims) == set(mask_temporal_dims):
        # To continue with a valid case, mask shouldn't have a temporal dimension, so that the mask will be applied to all the temporal labels
        if len(mask_temporal_dims) != 0:
            raise DimensionMismatch(
                f"data and mask temporal dimensions do no match: data has temporal dimensions ({data_temporal_dims}) and mask {mask_temporal_dims}"
            )

        #

    # if data.openeo.band_dims[0] == "bands"
    # if data.openeo.temporal_dims[0] == mask.openeo.temporal_dims[0]
    assert output_cube.openeo.spatial_dims == ("y", "x")
    assert output_cube.openeo.other_dims[0] == "other"

    maskSource = node.arguments["mask"]["from_node"]
    dataSource = node.arguments["data"]["from_node"]
    # If the mask has a variable dimension, it will keep only the values of the input with the same variable name.
    # Solution is to take the min over the variable dim to drop that dimension. (Problems if there are more than 1 band/variable)
    if (
        "variable" in self.partialResults[maskSource].dims
        and len(self.partialResults[maskSource]["variable"]) == 1
    ):
        mask = self.partialResults[maskSource].min(dim="variable")
    else:
        mask = self.partialResults[maskSource]
    self.partialResults[node.id] = self.partialResults[dataSource].where(
        np.logical_not(mask)
    )
    if "replacement" in node.arguments and node.arguments["replacement"] is not None:
        burnValue = node.arguments["replacement"]
        if isinstance(burnValue, int) or isinstance(burnValue, float):
            self.partialResults[node.id] = self.partialResults[node.id].fillna(
                burnValue
            )  # Replace the na with the burnValue

    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    labels = data[dimension].values
    label_mask = condition(x=labels)
    label = labels[label_mask]
    data = data.sel(**{dimension: label})
    return data
