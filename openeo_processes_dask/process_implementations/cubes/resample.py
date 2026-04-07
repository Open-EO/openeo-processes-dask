import logging
from typing import Dict, Optional, Union

import numpy as np
import odc.geo.xr
import rioxarray  # needs to be imported to set .rio accessor on xarray objects.
import xarray as xr
from odc.geo.geobox import resolution_from_affine
from pyproj import Transformer
from pyproj.crs import CRS, CRSError

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionMissing,
    OpenEOException,
)

logger = logging.getLogger(__name__)

__all__ = ["resample_spatial", "resample_cube_spatial", "resample_cube_temporal"]

resample_methods_list = [
    "near",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "geocode",  # new-addition which uses xcube
]


def _find_lon_lat_vars(ds: xr.Dataset) -> tuple[str, str]:
    """
    Return names of lon/lat variables in ds.
    Accepts both (lon, lat) and (longitude, latitude) as coords or data_vars.
    """
    candidates = [
        ("lon", "lat"),
        ("longitude", "latitude"),
    ]
    for lon_name, lat_name in candidates:
        if (lon_name in ds.coords or lon_name in ds.data_vars) and (
            lat_name in ds.coords or lat_name in ds.data_vars
        ):
            return lon_name, lat_name
    raise OpenEOException(
        'method="geocode" requires 2D lon/lat layers present as variables or coordinates '
        "(expected names: lon/lat or longitude/latitude)."
    )


def _default_interp_methods_from_dtypes(ds: xr.Dataset) -> dict[str, str]:
    """
    xcube supports 'nearest' and 'bilinear' (among others), and applies them per variable.
    We default:
      - integer/flags -> nearest
      - float -> bilinear
    """
    methods: dict[str, str] = {}
    for var_name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.integer) or np.issubdtype(da.dtype, np.bool_):
            methods[var_name] = "nearest"
        else:
            methods[var_name] = "bilinear"
    return methods


def _build_target_gm_from_lonlat_bbox(
    lon2d: xr.DataArray,
    lat2d: xr.DataArray,
    target_crs: CRS,
    resolution: float,
    tile_size: int = 1024,
):
    """
    Build a regular xcube GridMapping using the lon/lat bbox as extent.
    If target_crs != EPSG:4326, bbox is transformed to target CRS.
    """
    # Lazy reductions (works with dask-backed lon/lat)
    west = float(lon2d.min().compute())
    east = float(lon2d.max().compute())
    south = float(lat2d.min().compute())
    north = float(lat2d.max().compute())

    # Transform bounds from EPSG:4326 into target CRS if needed
    src_crs = CRS.from_epsg(4326)
    if not target_crs.equals(src_crs) and not (
        target_crs.is_geographic and src_crs.is_geographic
    ):
        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
        west, south, east, north = transformer.transform_bounds(
            west, south, east, north
        )

    # Import here so the module still works even if xcube-resampling isn't installed
    from xcube_resampling.gridmapping import GridMapping

    return GridMapping.regular_from_bbox(
        bbox=(west, south, east, north),
        xy_res=resolution,
        crs=target_crs,
        tile_size=tile_size,
        is_j_axis_up=False,
    )


def resample_spatial(
    data: RasterCube,
    projection: Optional[Union[str, int]] = None,
    resolution: int = 0,
    method: str = "near",
    align: str = "upper-left",
):
    """Resamples the spatial dimensions (x,y) of the data cube to a specified resolution and/or warps the data cube to the target projection. At least resolution or projection must be specified."""

    if data.openeo.y_dim is None or data.openeo.x_dim is None:
        raise DimensionMissing(f"Spatial dimension missing for dataset: {data} ")

    if method not in resample_methods_list:
        raise Exception(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(resample_methods_list)}]"
        )

    dim_order = data.dims
    data_cp = data.transpose(..., data.openeo.y_dim, data.openeo.x_dim)

    if method == "geocode":
        # Require resolution for geocode method
        if resolution == 0:
            raise OpenEOException(
                'method="geocode" requires a non-zero resolution parameter. '
                "Please specify the output resolution in the target CRS units."
            )

        # If projection not specified, default to EPSG:4326
        if projection is None:
            projection = 4326
            logger.info(
                'method="geocode": projection not specified, defaulting to EPSG:4326'
            )

        try:
            from xcube_resampling.gridmapping import GridMapping
            from xcube_resampling.rectify import rectify_dataset
            from xcube_resampling.spatial import resample_in_space
        except Exception as e:
            raise OpenEOException(
                'method="geocode" requires the optional dependency "xcube-resampling" to be installed.'
            ) from e

        y_dim = data.openeo.y_dim
        x_dim = data.openeo.x_dim
        band_dim = getattr(data.openeo, "band_dim", None) or "bands"

        def _bands_da_to_vars_ds(da: xr.DataArray, band_dim_: str) -> xr.Dataset:
            if band_dim_ not in da.dims:
                return da.to_dataset(name=(da.name or "data"))

            if band_dim_ not in da.coords:
                raise OpenEOException(
                    f'For method="geocode", band dimension {band_dim_!r} must have labels as a coordinate.'
                )

            labels = [str(v) for v in da[band_dim_].values.tolist()]
            vars_dict: dict[str, xr.DataArray] = {}
            for lbl in labels:
                vars_dict[lbl] = da.sel({band_dim_: lbl}).drop_vars(
                    band_dim_, errors="ignore"
                )

            coords = {k: v for k, v in da.coords.items() if k != band_dim_}
            ds = xr.Dataset(vars_dict, coords=coords)
            ds.attrs.update(getattr(da, "attrs", {}))
            return ds

        def _coerce_lonlat_to_coords_and_normalize(
            ds: xr.Dataset, lon_name: str, lat_name: str
        ) -> xr.Dataset:
            """
            Make sure lon/lat are coords and normalize names to 'longitude'/'latitude'.
            Ensures there are no duplicate lon/lat vars lingering in data_vars.
            """
            # move to coords if they are variables
            if lon_name in ds.data_vars:
                ds = ds.set_coords(lon_name)
            if lat_name in ds.data_vars:
                ds = ds.set_coords(lat_name)

            # normalize naming
            rename = {}
            if lon_name == "lon":
                rename["lon"] = "longitude"
            if lat_name == "lat":
                rename["lat"] = "latitude"
            if rename:
                ds = ds.rename(rename)

            return ds

        def _extract_lonlat_and_payload(
            obj: xr.DataArray | xr.Dataset,
            y_dim_: str,
            x_dim_: str,
            band_dim_: str,
        ) -> tuple[xr.Dataset, xr.DataArray, xr.DataArray, list[str], bool]:
            """
            Return payload_ds (ONLY payload vars), lon2d/lat2d as coords, payload_vars, input_was_dataarray.
            Supports:
              - Dataset with lon/lat or longitude/latitude as coords/vars
              - DataArray with lon/lat or longitude/latitude as coords/vars
              - DataArray where lon/lat or longitude/latitude are stored as bands
            """
            is_da = isinstance(obj, xr.DataArray)

            if is_da:
                da = obj

                # Case: lon/lat stored as band labels (your combined.nc case)
                if band_dim_ in da.dims and band_dim_ in da.coords:
                    band_labels = [str(b) for b in da[band_dim_].values.tolist()]
                    band_set = set(band_labels)

                    lat_band = (
                        "latitude"
                        if "latitude" in band_set
                        else ("lat" if "lat" in band_set else None)
                    )
                    lon_band = (
                        "longitude"
                        if "longitude" in band_set
                        else ("lon" if "lon" in band_set else None)
                    )

                    if lat_band and lon_band:
                        lat2d = da.sel({band_dim_: lat_band}).drop_vars(
                            band_dim_, errors="ignore"
                        )
                        lon2d = da.sel({band_dim_: lon_band}).drop_vars(
                            band_dim_, errors="ignore"
                        )

                        payload_band_labels = [
                            b for b in band_labels if b not in (lat_band, lon_band)
                        ]
                        if not payload_band_labels:
                            raise OpenEOException(
                                'method="geocode": bands contained only lon/lat; no payload bands to rectify.'
                            )

                        da_payload = da.sel({band_dim_: payload_band_labels})
                        payload_ds = _bands_da_to_vars_ds(
                            da_payload, band_dim_=band_dim_
                        )

                        # attach as coords under canonical names
                        payload_ds = payload_ds.assign_coords(
                            {"longitude": lon2d, "latitude": lat2d}
                        )
                        payload_ds = payload_ds.set_coords(["longitude", "latitude"])

                        payload_vars = list(payload_ds.data_vars)
                        return (
                            payload_ds,
                            payload_ds["longitude"],
                            payload_ds["latitude"],
                            payload_vars,
                            True,
                        )

                # General DataArray case: convert to dataset, then find lon/lat
                ds_full = (
                    _bands_da_to_vars_ds(da, band_dim_=band_dim_)
                    if band_dim_ in da.dims
                    else da.to_dataset(name=(da.name or "data"))
                )
                lon_name, lat_name = _find_lon_lat_vars(ds_full)
                ds_full = _coerce_lonlat_to_coords_and_normalize(
                    ds_full, lon_name, lat_name
                )

                # payload vars exclude lon/lat + spatial_ref
                payload_vars = [
                    v
                    for v in ds_full.data_vars
                    if v not in ("longitude", "latitude", "spatial_ref")
                ]
                if not payload_vars:
                    raise OpenEOException(
                        'method="geocode": no payload variables found (only lon/lat present?).'
                    )

                payload_ds = ds_full[payload_vars].assign_coords(
                    {"longitude": ds_full["longitude"], "latitude": ds_full["latitude"]}
                )
                payload_ds = payload_ds.set_coords(["longitude", "latitude"])
                return (
                    payload_ds,
                    payload_ds["longitude"],
                    payload_ds["latitude"],
                    payload_vars,
                    True,
                )

            # Dataset case
            ds_full = obj
            lon_name, lat_name = _find_lon_lat_vars(ds_full)
            ds_full = _coerce_lonlat_to_coords_and_normalize(
                ds_full, lon_name, lat_name
            )

            payload_vars = [
                v
                for v in ds_full.data_vars
                if v not in ("longitude", "latitude", "spatial_ref")
            ]
            if not payload_vars:
                raise OpenEOException(
                    'method="geocode": no payload variables found (only lon/lat present?).'
                )

            payload_ds = ds_full[payload_vars].assign_coords(
                {"longitude": ds_full["longitude"], "latitude": ds_full["latitude"]}
            )
            payload_ds = payload_ds.set_coords(["longitude", "latitude"])
            return (
                payload_ds,
                payload_ds["longitude"],
                payload_ds["latitude"],
                payload_vars,
                False,
            )

        def _stack_extra_dims_for_xcube(
            ds: xr.Dataset, vars_: list[str]
        ) -> tuple[xr.Dataset, bool, list[str]]:
            plane_dim_ = "__plane__"
            v0 = vars_[0]
            extra = [d for d in ds[v0].dims if d not in (y_dim, x_dim)]
            stacked_ = False

            # Always enforce spatial dims at the end
            if len(extra) == 0:
                for v in vars_:
                    ds[v] = ds[v].transpose(y_dim, x_dim)
                return ds, False, extra

            if len(extra) == 1:
                for v in vars_:
                    ds[v] = ds[v].transpose(extra[0], y_dim, x_dim)
                return ds, False, extra

            stacked_ = True
            stacked_vars = {}
            for v in vars_:
                da = ds[v].transpose(*extra, y_dim, x_dim)
                stacked_vars[v] = da.stack({plane_dim_: extra})
            ds = ds.assign(stacked_vars)
            return ds, stacked_, extra

        def _unstack_after_xcube(
            da: xr.DataArray, extra_dims: list[str]
        ) -> xr.DataArray:
            plane_dim_ = "__plane__"
            if plane_dim_ in da.dims:
                da = da.unstack(plane_dim_)
                da = da.transpose(*extra_dims, y_dim, x_dim)
            return da

        def _force_xcube_use_2d_xy(
            source_payload: xr.Dataset, y_dim_: str, x_dim_: str
        ) -> xr.Dataset:
            """
            Critical: avoid xcube preferring 1D x(x)/y(y) index coords.
            Provide 2D x(y,x), y(y,x) from lon/lat.
            """
            lon2d_ = source_payload.coords.get("longitude")
            lat2d_ = source_payload.coords.get("latitude")
            if lon2d_ is None or lat2d_ is None:
                raise OpenEOException(
                    'method="geocode": missing longitude/latitude coords.'
                )

            if lon2d_.dims != (y_dim_, x_dim_) or lat2d_.dims != (y_dim_, x_dim_):
                raise OpenEOException(
                    f'method="geocode": longitude/latitude must have dims ({y_dim_},{x_dim_}); got {lon2d_.dims} and {lat2d_.dims}.'
                )

            # drop 1D index coords if present
            drop_names = []
            if "x" in source_payload.coords and source_payload["x"].dims == (x_dim_,):
                drop_names.append("x")
            if "y" in source_payload.coords and source_payload["y"].dims == (y_dim_,):
                drop_names.append("y")
            if drop_names:
                source_payload = source_payload.drop_vars(drop_names)

            # assign 2D x/y coords for xcube GridMapping detection
            return source_payload.assign_coords(
                {
                    "x": lon2d_.astype(np.float64),
                    "y": lat2d_.astype(np.float64),
                }
            )

        def _rename_output_spatial_dims_to_cube(
            ds_or_da: xr.Dataset | xr.DataArray,
        ) -> xr.Dataset | xr.DataArray:
            """
            xcube often outputs dims named ('lat','lon') or ('latitude','longitude').
            Rename them back to the cube's y_dim/x_dim so transpose(*dim_order) works.
            """
            rename = {}
            if "lat" in ds_or_da.dims and y_dim not in ds_or_da.dims:
                rename["lat"] = y_dim
            if "lon" in ds_or_da.dims and x_dim not in ds_or_da.dims:
                rename["lon"] = x_dim
            if "latitude" in ds_or_da.dims and y_dim not in ds_or_da.dims:
                rename["latitude"] = y_dim
            if "longitude" in ds_or_da.dims and x_dim not in ds_or_da.dims:
                rename["longitude"] = x_dim
            return ds_or_da.rename(rename) if rename else ds_or_da

        dim_order = data.dims
        data_cp = data.transpose(..., y_dim, x_dim)

        (
            payload_ds,
            lon2d,
            lat2d,
            payload_vars,
            input_was_dataarray,
        ) = _extract_lonlat_and_payload(
            data_cp, y_dim_=y_dim, x_dim_=x_dim, band_dim_=band_dim
        )

        # Make absolutely sure lon/lat are coords only and NOT data_vars
        payload_ds = payload_ds.set_coords(["longitude", "latitude"])
        payload_ds = payload_ds[payload_vars]  # enforce payload-only variables
        payload_ds = payload_ds.assign_coords({"longitude": lon2d, "latitude": lat2d})

        if lon2d.ndim != 2 or lat2d.ndim != 2:
            raise OpenEOException(
                f'method="geocode" requires 2D lon/lat aligned with (y,x); got shapes {lon2d.shape} and {lat2d.shape}.'
            )

        interp_methods = _default_interp_methods_from_dtypes(payload_ds)

        # dim prep for xcube
        payload_ds2, stacked, extra_dims = _stack_extra_dims_for_xcube(
            payload_ds, payload_vars
        )

        # critical fix: enforce 2D x/y coords
        payload_ds2 = _force_xcube_use_2d_xy(payload_ds2, y_dim_=y_dim, x_dim_=x_dim)

        user_passed_projection = projection is not None
        user_passed_resolution = resolution not in (0, None)

        if not user_passed_projection and not user_passed_resolution:
            out_ds = rectify_dataset(
                payload_ds2,
                interp_methods=interp_methods,
                tile_size=1024,
            )
        else:
            if projection is None:
                target_crs = CRS.from_epsg(4326)
            else:
                try:
                    target_crs = CRS.from_user_input(projection)
                except CRSError as e:
                    raise CRSError(
                        f"Provided projection string: '{projection}' can not be parsed to CRS."
                    ) from e

            if not user_passed_resolution:
                raise OpenEOException(
                    'method="geocode": if "projection" is provided explicitly, you must also provide a non-zero "resolution".'
                )

            target_gm = _build_target_gm_from_lonlat_bbox(
                lon2d=lon2d,
                lat2d=lat2d,
                target_crs=target_crs,
                resolution=float(resolution),
                tile_size=1024,
            )

            out_ds = resample_in_space(
                payload_ds2,
                target_gm=target_gm,
                interp_methods=interp_methods,
            )

        # Unstack back if needed
        if stacked:
            fixed = {}
            for v in [v for v in out_ds.data_vars if v != "spatial_ref"]:
                fixed[v] = _unstack_after_xcube(out_ds[v], extra_dims)
            out_ds = out_ds.assign(fixed)

        # Convert back to original type/shape + fix output dims names
        if input_was_dataarray:
            out_vars = [v for v in out_ds.data_vars if v != "spatial_ref"]
            if not out_vars:
                raise OpenEOException(
                    'method="geocode": no output variables produced (only spatial_ref).'
                )

            out = xr.concat([out_ds[v] for v in out_vars], dim=band_dim).assign_coords(
                {band_dim: out_vars}
            )

            # <<< FIX: rename lat/lon dims to y/x before transpose >>>
            out = _rename_output_spatial_dims_to_cube(out)

            out = out.transpose(*dim_order, missing_dims="ignore")
        else:
            out_ds = _rename_output_spatial_dims_to_cube(out_ds)
            out = out_ds.transpose(*dim_order, missing_dims="ignore")

        # Preserve attrs (except CRS handled elsewhere)
        for k, v in data.attrs.items():
            if k.lower() != "crs":
                out.attrs[k] = v

        return out

    # ORIGINAL ODC-based behavior continues from here
    # Assert resampling method is correct.
    if method == "near":
        method = "nearest"

    elif method not in resample_methods_list:
        raise OpenEOException(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(resample_methods_list)}]"
        )

    if projection is None:
        projection = data_cp.rio.crs

    try:
        projection = CRS.from_user_input(projection)
    except CRSError as e:
        raise CRSError(
            f"Provided projection string: '{projection}' can not be parsed to CRS."
        ) from e

    if resolution == 0:
        resolution = resolution_from_affine(data_cp.odc.geobox.affine).x

    reprojected = data_cp.odc.reproject(
        how=projection, resolution=resolution, resampling=method
    )

    if reprojected.openeo.x_dim != data.openeo.x_dim:
        reprojected = reprojected.rename({reprojected.openeo.x_dim: data.openeo.x_dim})

    if reprojected.openeo.y_dim != data.openeo.y_dim:
        reprojected = reprojected.rename({reprojected.openeo.y_dim: data.openeo.y_dim})

    reprojected = reprojected.transpose(*dim_order)

    reprojected.attrs["crs"] = data_cp.rio.crs

    return reprojected


def resample_cube_spatial(
    data: RasterCube, target: RasterCube, method="near", options=None
) -> RasterCube:
    methods_list = [
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
    ]

    if (
        data.openeo.y_dim is None
        or data.openeo.x_dim is None
        or target.openeo.y_dim is None
        or target.openeo.x_dim is None
    ):
        raise DimensionMissing(
            f"Spatial dimension missing from data or target. Available dimensions for data: {data.dims} for target: {target.dims}"
        )

    # ODC reproject requires y to be before x
    required_dim_order = (..., data.openeo.y_dim, data.openeo.x_dim)

    data_reordered = data.transpose(*required_dim_order, missing_dims="ignore")
    target_reordered = target.transpose(*required_dim_order, missing_dims="ignore")

    if method == "near":
        method = "nearest"

    elif method not in methods_list:
        raise Exception(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(methods_list)}]"
        )

    resampled_data = data_reordered.odc.reproject(
        target_reordered.odc.geobox, resampling=method
    )

    resampled_data.rio.write_crs(target_reordered.rio.crs, inplace=True)

    try:
        # odc.reproject renames the coordinates according to the geobox, this undoes that.
        resampled_data = resampled_data.rename(
            {"longitude": data.openeo.x_dim, "latitude": data.openeo.y_dim}
        )
    except ValueError:
        pass

    # Order axes back to how they were before
    resampled_data = resampled_data.transpose(*data.dims)

    # Ensure that attrs except crs are copied over
    for k, v in data.attrs.items():
        if k.lower() != "crs":
            resampled_data.attrs[k] = v
    return resampled_data


def resample_cube_temporal(data, target, dimension=None, valid_within=None):
    if dimension is None:
        if len(data.openeo.temporal_dims) > 0:
            dimension = data.openeo.temporal_dims[0]
        else:
            raise Exception("DimensionNotAvailable")
    if dimension not in data.dims:
        raise Exception("DimensionNotAvailable")
    if dimension not in target.dims:
        if len(target.openeo.temporal_dims) > 0:
            target_time = target.openeo.temporal_dims[0]
        else:
            raise Exception("DimensionNotAvailable")
        target = target.rename({target_time: dimension})
    index = []
    for d in target[dimension].values:
        difference = np.abs(d - data[dimension].values)
        nearest = np.argwhere(difference == np.min(difference))
        # The rare case of ties is resolved by choosing the earlier timestamps. (index 0)
        if np.shape(nearest) == (2, 1):
            nearest = nearest[0]
        if np.shape(nearest) == (1, 2):
            nearest = nearest[:, 0]
        index.append(int(nearest))
    times_at_target_time = data[dimension].values[index]
    new_data = data.loc[{dimension: times_at_target_time}]
    filter_values = new_data[dimension].values
    new_data[dimension] = target[dimension].values
    # valid_within
    if valid_within is None:
        new_data = new_data
    else:
        minimum = np.timedelta64(valid_within, "D")
        filter_valid = np.abs(filter_values - new_data[dimension].values) <= minimum
        times_valid = new_data[dimension].values[filter_valid]
        valid_data = new_data.loc[{dimension: times_valid}]
        filter_nan = np.abs(filter_values - new_data[dimension].values) > minimum
        times_nan = new_data[dimension].values[filter_nan]
        nan_data = new_data.loc[{dimension: times_nan}] * np.nan
        combined = xr.concat([valid_data, nan_data], dim=dimension)
        new_data = combined.sortby(dimension)
    new_data.attrs = data.attrs
    return new_data
