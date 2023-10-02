import logging
from typing import Callable

import numpy as np

from openeo_processes_dask.process_implementations.cubes.utils import notnull
from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionLabelCountMismatch,
    DimensionMismatch,
    LabelMismatch,
)
from openeo_processes_dask.process_implementations.logic import _not

logger = logging.getLogger(__name__)

__all__ = ["mask"]


def mask(data: RasterCube, mask: RasterCube, replacement=None) -> RasterCube:
    if replacement is None:
        replacement = np.nan

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

    return data.where(_not(mask), replacement)
