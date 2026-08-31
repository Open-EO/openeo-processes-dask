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


def _dummy_value(dtype):
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


def _add_missing_coords(target, reference, dim):
    """
    Ensure `target` has every coordinate that `reference` has.
    Missing coordinates are created with a dummy value
    """
    missing = {}
    for name, coord in reference.coords.items():
        if name in target.coords:
            continue

        fill = _dummy_value(coord.dtype)

        if coord.dims == ():  # scalar coordinate
            missing[name] = ((), np.array(fill, dtype=coord.dtype))

        elif coord.dims == (dim,):  # coordinate along concat dim
            missing[name] = (dim, np.full(target.sizes[dim], fill, dtype=coord.dtype))

        elif all(d in target.dims for d in coord.dims):  # other dims
            shape = tuple(target.sizes[d] for d in coord.dims)
            missing[name] = (coord.dims, np.full(shape, fill, dtype=coord.dtype))
        # else: dims don't exist in target -> can't sensibly create it, skip

    return target.assign_coords(**missing) if missing else target


def ndvi(data: RasterCube, nir="nir", red="red", target_band=None):
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
        nd = _add_missing_coords(nd, data, band_dim)
        nd = xr.concat([data, nd], dim=band_dim)

    nd.attrs = data.attrs
    return nd
