from typing import Optional, Tuple

import odc.geo.xr  # Required for the .geo accessor on xarrays.
import xarray as xr


@xr.register_dataarray_accessor("openeo")
class OpenEOExtensionDa:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def spatial_dims(self) -> Optional[tuple[str, str]]:
        spatial_dims = self._obj.odc.spatial_dims
        return spatial_dims

    @property
    def x_dim(self) -> Optional[str]:
        spatial_dims = self._obj.odc.spatial_dims
        if spatial_dims is not None:
            return spatial_dims[1]
        return None

    @property
    def y_dim(self) -> Optional[str]:
        spatial_dims = self._obj.odc.spatial_dims
        if spatial_dims is not None:
            return spatial_dims[0]
        return None

    @property
    def z_dim(self):
        raise NotImplementedError()

    @property
    def temporal_dims(self) -> Optional[list[str]]:
        """Find and return all temporal dimensions of the datacube as a list."""
        guesses = [
            "time",
            "t",
            "year",
            "quarter",
            "month",
            "week",
            "day",
            "hour",
            "second",
        ]

        dims = {str(dim).casefold(): str(dim) for dim in self._obj.dims}

        temporal_dims = []
        for guess in guesses:
            if guess in dims:
                temporal_dims.append(dims[guess])

        return temporal_dims if temporal_dims else None

    @property
    def band_dims(self) -> Optional[list[str]]:
        guesses = ["b", "bands"]

        dims = {str(dim).casefold(): str(dim) for dim in self._obj.dims}

        bands_dims = []
        for guess in guesses:
            if guess in dims:
                bands_dims.append(dims[guess])

        return bands_dims if bands_dims else None
