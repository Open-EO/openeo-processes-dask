import logging
from typing import Callable

import numpy as np

from openeo_processes_dask.process_implementations.cubes.resample import (
    resample_cube_spatial,
)
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

    data_band_dims = data.openeo.band_dims
    mask_band_dims = mask.openeo.band_dims
    # Check if temporal dimensions are present and check the names
    data_temporal_dims = data.openeo.temporal_dims
    mask_temporal_dims = mask.openeo.temporal_dims

    check_temporal_labels = True
    if not set(data_temporal_dims) == set(mask_temporal_dims):
        check_temporal_labels = False
        # To continue with a valid case, mask shouldn't have a temporal dimension, so that the mask will be applied to all the temporal labels
        if len(mask_temporal_dims) != 0:
            raise DimensionMismatch(
                f"data and mask temporal dimensions do no match: data has temporal dimensions ({data_temporal_dims}) and mask {mask_temporal_dims}."
            )
    if check_temporal_labels:
        # Check if temporal labels correspond
        for n in data_temporal_dims:
            data_temporal_labels = data[n].values
            mask_temporal_labels = mask[n].values
            data_n_labels = len(data_temporal_labels)
            mask_n_labels = len(mask_temporal_labels)

            if not data_n_labels == mask_n_labels:
                raise DimensionLabelCountMismatch(
                    f"data and mask temporal dimensions do no match: data has {data_n_labels} temporal dimensions labels and mask {mask_n_labels}."
                )
            elif not all(data_temporal_labels == mask_temporal_labels):
                raise LabelMismatch(
                    f"data and mask temporal dimension labels don't match for dimension {n}."
                )

    # From the process definition: https://processes.openeo.org/#mask
    # The data cubes have to be compatible except that the horizontal spatial dimensions (axes x and y) will be aligned implicitly by resample_cube_spatial.
    apply_resample_cube_spatial = False

    # Check if spatial dimensions have the same name
    data_spatial_dims = data.openeo.spatial_dims
    mask_spatial_dims = mask.openeo.spatial_dims
    if not set(data_spatial_dims) == set(mask_spatial_dims):
        raise DimensionMismatch(
            f"data and mask spatial dimensions do no match: data has spatial dimensions ({data_spatial_dims}) and mask {mask_spatial_dims}"
        )

    # Check if spatial labels correspond
    for n in data_spatial_dims:
        data_spatial_labels = data[n].values
        mask_spatial_labels = mask[n].values
        data_n_labels = len(data_spatial_labels)
        mask_n_labels = len(mask_spatial_labels)

        if not data_n_labels == mask_n_labels:
            apply_resample_cube_spatial = True
            logger.info(
                f"data and mask spatial dimension labels don't match: data has ({data_n_labels}) labels and mask has {mask_n_labels} for dimension {n}."
            )
        elif not all(data_spatial_labels == mask_spatial_labels):
            apply_resample_cube_spatial = True
            logger.info(
                f"data and mask spatial dimension labels don't match for dimension {n}, i.e. the coordinate values are different."
            )

    if apply_resample_cube_spatial:
        logger.info(f"mask is aligned to data using resample_cube_spatial.")
        mask = resample_cube_spatial(data=mask, target=data)

    original_dim_order = data.dims
    # If bands dimension in data but not in mask, ensure that it comes first and all the other dimensions at the end
    if len(data_band_dims) != 0 and len(mask_band_dims) == 0:
        required_dim_order = (
            data_band_dims[0] if len(data_band_dims) > 0 else (),
            data_temporal_dims[0] if len(data_temporal_dims) > 0 else (),
            data.openeo.y_dim,
            data.openeo.x_dim,
        )
        data = data.transpose(*required_dim_order, missing_dims="ignore")
        mask = mask.transpose(*required_dim_order, missing_dims="ignore")

    elif len(data_temporal_dims) != 0 and len(mask_temporal_dims) == 0:
        required_dim_order = (
            data_temporal_dims[0] if len(data_temporal_dims) > 0 else (),
            data_band_dims[0] if len(data_band_dims) > 0 else (),
            data.openeo.y_dim,
            data.openeo.x_dim,
        )
        data = data.transpose(*required_dim_order, missing_dims="ignore")
        mask = mask.transpose(*required_dim_order, missing_dims="ignore")

    data = data.where(_not(mask), replacement)

    if len(data_band_dims) != 0 and len(mask_band_dims) == 0:
        # Order axes back to how they were before
        data = data.transpose(*original_dim_order)

    return data
