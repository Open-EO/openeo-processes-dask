import rioxarray  # needs to be imported to set .rio accessor on xarray objects.

from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["resample_cube_spatial"]


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

    # ODC reproject requires y to be before x
    required_dim_order = (
        data.openeo.band_dims
        + data.openeo.temporal_dims
        + tuple(data.openeo.y_dim)
        + tuple(data.openeo.x_dim)
    )

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
