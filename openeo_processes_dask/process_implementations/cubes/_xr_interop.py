from typing import Optional

import odc.geo.xr  # Required for the .geo accessor on xarrays.
import xarray as xr

TEMPORAL_GUESSES = [
    "DATE",
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
X_GUESSES = ["x", "lon", "longitude"]
Y_GUESSES = ["y", "lat", "latitude"]
BANDS_GUESSES = ["b", "bands", "band"]


@xr.register_dataarray_accessor("openeo")
class OpenEOExtensionDa:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._spatial_dims = self._guess_dims_for_type(
            X_GUESSES
        ) + self._guess_dims_for_type(Y_GUESSES)
        self._temporal_dims = self._guess_dims_for_type(TEMPORAL_GUESSES)
        self._bands_dims = self._guess_dims_for_type(BANDS_GUESSES)
        self._other_dims = [
            dim
            for dim in self._obj.dims
            if dim not in self._spatial_dims + self._temporal_dims + self._bands_dims
        ]

    @property
    def _lowercase_dims(self):
        return [str(dim).casefold() for dim in self._obj.dims]

    def _guess_dims_for_type(self, guesses):
        found_dims = []
        datacube_dims = self._lowercase_dims
        for guess in guesses:
            if guess in datacube_dims:
                i = datacube_dims.index(guess)
                found_dims.append(self._obj.dims[i])
        return found_dims

    def _get_existing_dims_and_pop_missing(self, expected_dims):
        existing_dims = []
        for i, dim in enumerate(expected_dims):
            if dim in self._obj.dims:
                existing_dims.append(dim)
            else:
                expected_dims.pop(i)
        return existing_dims

    @property
    def spatial_dims(self) -> tuple[str]:
        """Find and return all spatial dimensions of the datacube as a tuple."""
        return tuple(self._get_existing_dims_and_pop_missing(self._spatial_dims))

    @property
    def temporal_dims(self) -> tuple[str]:
        """Find and return all temporal dimensions of the datacube as a list."""
        return tuple(self._get_existing_dims_and_pop_missing(self._temporal_dims))

    @property
    def band_dims(self) -> tuple[str]:
        """Find and return all bands dimensions of the datacube as a list."""
        return tuple(self._get_existing_dims_and_pop_missing(self._bands_dims))

    @property
    def other_dims(self) -> tuple[str]:
        """Find and return any dimensions with type other as s list."""
        return tuple(self._get_existing_dims_and_pop_missing(self._other_dims))

    @property
    def x_dim(self) -> Optional[str]:
        return next(
            iter(
                [
                    dim
                    for dim in self.spatial_dims
                    if str(dim).casefold() in X_GUESSES and dim in self._obj.dims
                ]
            ),
            None,
        )

    @property
    def y_dim(self) -> Optional[str]:
        return next(
            iter(
                [
                    dim
                    for dim in self.spatial_dims
                    if str(dim).casefold() in Y_GUESSES and dim in self._obj.dims
                ]
            ),
            None,
        )

    @property
    def z_dim(self):
        raise NotImplementedError()

    def add_dim_type(self, name: str, type: str) -> None:
        """Add dimension name to the list of guesses when calling add_dimension."""

        if name not in self._obj.dims:
            raise ValueError("Trying to add a dimension that doesn't exist")

        if type == "spatial":
            self._spatial_dims.append(name)
        elif type == "temporal":
            self._temporal_dims.append(name)
        elif type == "bands":
            self._bands_dims.append(name)
        elif type == "other":
            self._other_dims.append(name)
        else:
            raise ValueError(f"Type {type} is not understood")
