from typing import Any

import numpy as np
import xarray as xr

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    BandExists,
    DimensionAmbiguous,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from openeo_processes_dask.process_implementations.math import normalized_difference

__all__ = ["ndvi"]


def _dummy_value(dtype: np.dtype) -> Any:
    """A sensible 'missing' fill value for a given numpy dtype."""
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.datetime64):
        return np.datetime64("NaT")
    if np.issubdtype(dtype, np.timedelta64):
        return np.timedelta64("NaT")
    if np.issubdtype(dtype, np.integer):
        return -1  # ints can't hold NaN
    if np.issubdtype(dtype, np.bool_):
        return False
    return ""  # str / object


def _add_missing_coords(
    target: xr.DataArray, reference: xr.DataArray, dim: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Adds missing secondary coordinate to dimension dim in 'reference' to 'target'
    Returns the target DataArray with added coordiantes, and the reference with
    potentially altered coordinate data type
    :param target: DataArray to which secondary coordinate is added
    :param reference: DataArray to be checked for secondary coordiantes to dimension dim
    :param dim: Name of dimension to check for secondary coordiante in dim
    :return: DataArray `target` with potentially added coordinate dim,
        DataArray `reference` with potentially altered coordinate dtype
    """
    missing = {}

    # get all secondary coordinates to dim in reference DataArray
    secondary_coord_to_dim = [
        (name, coord)
        for name, coord in reference.coords.items()
        if name != dim and coord.dims == (dim,)
    ]

    for name, coord in secondary_coord_to_dim:
        # special case: dim is "common_name"
        if coord.dtype.kind == "U" and name == "common_name":
            fill = "NDVI"

            # convert coord dtype to U<4 to fit "NDVI", if it is less than 4 characters
            if coord.dtype.kind == "U" and int(coord.dtype.str[2:]) < 4:
                reference = reference.assign_coords(common_name=coord.astype("<U4"))

        else:
            fill = _dummy_value(coord.dtype)

        missing[name] = (dim, np.array([fill]))

    return target.assign_coords(**missing) if missing else target, reference


def ndvi(
    data: RasterCube, nir: str = "nir", red: str = "red", target_band: str | None = None
) -> xr.DataArray:
    if len(data.openeo.band_dims) == 0:
        raise DimensionAmbiguous(
            "Dimension of type `bands` is not available or is ambiguous."
        )
    band_dim = data.openeo.band_dims[0]
    available_bands = data.coords[band_dim]

    if nir not in available_bands or red not in available_bands:
        try:
            data = data.set_xindex("common_name")
        except (ValueError, KeyError):
            pass

        if (
            nir not in available_bands
            and "common_name" in data.xindexes._coord_name_id.keys()
            and nir not in data.coords["common_name"].data
        ):
            raise NirBandAmbiguous(
                "The NIR band can't be resolved, please specify the specific NIR band name."
            )
        elif (
            red not in available_bands
            and "common_name" in data.xindexes._coord_name_id.keys()
            and red not in data.coords["common_name"].data
        ):
            raise RedBandAmbiguous(
                "The Red band can't be resolved, please specify the specific Red band name."
            )

    nir_band_dim = "common_name" if nir not in available_bands else band_dim
    red_band_dim = "common_name" if red not in available_bands else band_dim

    nir_band = data.sel({nir_band_dim: nir})
    red_band = data.sel({red_band_dim: red})

    nd = normalized_difference(nir_band, red_band)
    if target_band is not None:
        if target_band in data.coords:
            raise BandExists("A band with the specified target name exists.")
        nd = nd.expand_dims(band_dim).assign_coords({band_dim: [target_band]})

        # add potentially missing coords from data to nd, so that xr.concat works
        nd, data = _add_missing_coords(nd, data, band_dim)
        nd = xr.concat([data, nd], dim=band_dim)

    nd.attrs = data.attrs
    return nd
