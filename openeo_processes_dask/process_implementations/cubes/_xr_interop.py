from typing import Optional

import odc.geo.xr  # Required for the .geo accessor on xarrays.
import xarray as xr

from openeo_processes_dask.exceptions import DimensionNotAvailable


@xr.register_dataarray_accessor("openeo")
class OpenEOExtensionDa:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

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
    def temporal_dims(self) -> Optional[list]:
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

    def bands_dim(self):
        guesses = ["b", "bands"]

        dims = {str(dim) for dim in self._obj.dims}

        for guess in guesses:
            if dims.issuperset(guess):
                return guess

        raise DimensionNotAvailable(
            f"Unable to identify bands dimension on datacube. Available dimensions: {self._obj.dims}"
        )
