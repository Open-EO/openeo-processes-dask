from typing import Optional

import odc.geo.xr  # Required for the .geo accessor on xarrays.
import xarray as xr


@xr.register_dataarray_accessor("openeo")
class OpenEOExtensionDa:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._spatial_guesses = [
            "y",
            "Y",
            "x",
            "X",
            "lat",
            "latitude",
            "lon",
            "longitude",
        ]
        self._x_guesses = ["x", "X", "lon", "longitude"]
        self._y_guesses = ["y", "Y", "lat", "latitude"]
        self._temporal_guesses = [
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
        self._bands_guesses = ["b", "bands", "band"]
        self._other_guesses = []

    @property
    def spatial_dims(self) -> tuple[str, str]:
        """Find and return all spatial dimensions of the datacube as a list."""
        spatial_dims = self._obj.odc.spatial_dims
        return spatial_dims if spatial_dims is not None else tuple()

    @property
    def x_dim(self) -> Optional[str]:
        dims = {str(dim).casefold(): str(dim) for dim in self._obj.dims}

        for guess in self._x_guesses:
            if guess in dims:
                return dims[guess]
        return None

    @property
    def y_dim(self) -> Optional[str]:
        dims = {str(dim).casefold(): str(dim) for dim in self._obj.dims}

        for guess in self._y_guesses:
            if guess in dims:
                return dims[guess]
        return None

    @property
    def z_dim(self):
        raise NotImplementedError()

    @property
    def temporal_dims(self) -> tuple[str]:
        """Find and return all temporal dimensions of the datacube as a list."""
        dims = {str(dim).casefold(): str(dim) for dim in self._obj.dims}

        temporal_dims = []
        for guess in self._temporal_guesses:
            if guess in dims:
                temporal_dims.append(dims[guess])

        return tuple(temporal_dims)

    @property
    def band_dims(self) -> tuple[str]:
        """Find and return all bands dimensions of the datacube as a list."""
        dims = {str(dim).casefold(): str(dim) for dim in self._obj.dims}

        bands_dims = []
        for guess in self._bands_guesses:
            if guess in dims:
                bands_dims.append(dims[guess])

        return tuple(bands_dims)

    @property
    def other_dims(self) -> tuple[str]:
        """Find and return any dimensions with type other as s list."""
        dims = {str(dim).casefold(): str(dim) for dim in self._obj.dims}

        other_dims = []
        for guess in self._other_guesses:
            if guess in dims:
                other_dims.append(dims[guess])

        return tuple(other_dims)

    def add_dim_to_guesses(self, name: str, type: str) -> None:
        """Add dimension name to the list of guesses when calling add_dimension."""

        if type == "spatial":
            self._spatial_guesses.append(name)
        elif type == "temporal":
            self._temporal_guesses.append(name)
        elif type == "bands":
            self._bands_guesses.append(name)
        elif type == "other":
            self._other_guesses.append(name)

        return None
