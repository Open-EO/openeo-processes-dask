import odc.geo.xr  # Required for the .geo accessor on xarrays.
import xarray as xr

from openeo_processes_dask.exceptions import DimensionNotAvailable


@xr.register_dataarray_accessor("openeo")
class OpenEOExtensionDa:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def x_dim(self):
        spatial_dims = self._obj.odc.spatial_dims
        return spatial_dims[1]

    @property
    def y_dim(self):
        spatial_dims = self._obj.odc.spatial_dims
        return spatial_dims[0]

    @property
    def z_dim(self):
        return NotImplemented()

    @property
    def time_dim(self):
        guesses = ["time", "t"]

        dims = {str(dim) for dim in self._obj.dims}

        for guess in guesses:
            if dims.issuperset(guess):
                return guess

        raise DimensionNotAvailable(
            f"Datacube has no temporal dimension. Available dimensions: {self._obj.dims}"
        )

    def bands_dim(self):
        guesses = ["b", "bands"]

        dims = {str(dim) for dim in self._obj.dims}

        for guess in guesses:
            if dims.issuperset(guess):
                return guess

        raise DimensionNotAvailable(
            f"Datacube has no bands dimensions. Available dimensions: {self._obj.dims}"
        )
