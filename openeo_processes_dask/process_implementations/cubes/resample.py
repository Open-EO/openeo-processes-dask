import odc.algo
import rioxarray  # needs to be imported to set .rio accessor on xarray objects.

from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["resample_cube_spatial"]


def resample_cube_spatial(
    data: RasterCube, target: RasterCube, method="near", options=None
) -> RasterCube:
    # NOTE: Using the odc-algo library is not great, because it is only a random collection of experimental opendatacube features
    # but we've investigated all other available alternatives for resampling that are currently available with dask and none do the job.
    # We've tested pyresample (did not work at all and requires loads of helper code),
    # rasterio.reproject_match (loads all the data into memory to do gdal.warp) and odc-geo (doesn't support dask yet).
    # Github issue tracking this feature in rioxarray: https://github.com/corteva/rioxarray/issues/119
    # Github issue tracking this feature in odc-geo: https://github.com/opendatacube/odc-geo/issues/26

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
    if method == "near":
        method = "nearest"

    elif method not in methods_list:
        raise Exception(
            f'Selected resampling method "{method}" is not available! Please select one of '
            f"[{', '.join(methods_list)}]"
        )

    resampled_data = odc.algo._warp.xr_reproject(data, target.geobox, resampling=method)

    try:
        # xr_reproject renames the coordinates according to the geobox, this undoes that.
        resampled_data = resampled_data.rename({"longitude": "x", "latitude": "y"})
    except ValueError:
        pass

    # Ensure that attrs except crs are copied over
    for k, v in data.attrs.items():
        if k.lower() != "crs":
            resampled_data.attrs[k] = v
    return resampled_data
