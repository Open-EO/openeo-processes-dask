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
        nd = xr.merge([data, nd])
    nd.attrs = data.attrs
    return nd
