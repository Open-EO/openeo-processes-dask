import datetime
import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import numpy as np
import odc.stac
import planetary_computer as pc
import pyproj
import pystac_client
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval
from stac_validator import stac_validator

from openeo_processes_dask.process_implementations.cubes._filter import (
    _reproject_bbox,
    filter_bands,
    filter_bbox,
    filter_temporal,
)
from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    NoDataAvailable,
    OpenEOException,
    TemporalExtentEmpty,
)

# "NoDataAvailable": {
#     "message": "There is no data available for the given extents."
# },
# "TemporalExtentEmpty": {
#     "message": "The temporal extent is empty. The second instant in time must always be greater/later than the first instant in time."
# }
__all__ = ["load_stac"]

logger = logging.getLogger(__name__)


def _validate_stac(url):
    logger.debug(f"Validating the provided STAC url: {url}")
    stac = stac_validator.StacValidate(url)
    is_valid_stac = stac.run()
    if not is_valid_stac:
        raise Exception(
            f"The provided link is not a valid STAC. stac-validator message: {stac.message}"
        )
    if len(stac.message) == 1:
        try:
            asset_type = stac.message[0]["asset_type"]
        except:
            raise Exception(f"stac-validator returned an error: {stac.message}")
    else:
        raise Exception(
            f"stac-validator returned multiple items, not supported yet. {stac.message}"
        )
    return asset_type


def _search_for_parent_catalog(url):
    parsed_url = urlparse(url)
    root_url = parsed_url.scheme + "://" + parsed_url.netloc
    catalog_url = root_url
    url_parts = PurePosixPath(unquote(parsed_url.path)).parts
    collection_id = url_parts[-1]
    for p in url_parts:
        if p != "/":
            catalog_url = catalog_url + "/" + p
        try:
            asset_type = _validate_stac(catalog_url)
        except Exception as e:
            logger.debug(e)
            continue
        if asset_type == "CATALOG":
            break
    if asset_type != "CATALOG":
        raise Exception(
            "It was not possible to find the root STAC Catalog starting from the provided Collection."
        )
    return catalog_url, collection_id


def _load_with_xcube_eopf(
    data_id: str,
    spatial_extent: Optional[BoundingBox] = None,
    temporal_extent: Optional[TemporalInterval] = None,
    bands: Optional[list[str]] = None,
    resolution: Optional[float] = None,
    projection: Optional[Union[int, str]] = None,
) -> RasterCube:
    """Load data using xcube-eopf package for EOPF STAC endpoints."""
    try:
        from xcube.core.store import new_data_store
    except ImportError:
        raise ImportError(
            "xcube-eopf package is required for loading EOPF STAC data. "
            "Please install it with: pip install xcube-eopf"
        )

    # Convert spatial extent to bbox
    bbox = None
    if spatial_extent is not None:
        try:
            spatial_extent_4326 = spatial_extent
            if spatial_extent.crs is not None:
                if not pyproj.crs.CRS(spatial_extent.crs).equals("EPSG:4326"):
                    spatial_extent_4326 = _reproject_bbox(spatial_extent, "EPSG:4326")
            bbox = [
                spatial_extent_4326.west,
                spatial_extent_4326.south,
                spatial_extent_4326.east,
                spatial_extent_4326.north,
            ]
        except Exception as e:
            raise Exception(f"Unable to parse the provided spatial extent: {e}")

    # Convert temporal extent to time_range
    time_range = None
    if temporal_extent is not None:
        start_date = (
            str(temporal_extent[0].to_numpy().astype("datetime64[D]"))
            if temporal_extent[0] is not None
            else None
        )
        end_date = (
            str(temporal_extent[1].to_numpy().astype("datetime64[D]"))
            if temporal_extent[1] is not None
            else None
        )
        #print(start_date, end_date)
        time_range = [start_date, end_date]

    # Set CRS
    crs = projection if projection else "EPSG:4326"

    # Convert resolution from degrees to meters if needed
    spatial_res = resolution if resolution else 10/111320
    #spatial_res = None
    '''
    if resolution is not None:
        if crs == "EPSG:4326":
            # Approximate conversion from degrees to meters at equator
            spatial_res = resolution
        else:
            spatial_res = resolution
    '''
    # Create store and open data
    store = new_data_store("eopf-zarr")
    ds = store.open_data(
        data_id=data_id,
        bbox=bbox,
        time_range=time_range,
        spatial_res=spatial_res,
        crs=crs,
        variables=bands,
    )

    # Convert to dataarray if it's a dataset
    if isinstance(ds, xr.Dataset):
        # Find the appropriate dimension name for bands
        band_dim = None
        for dim in ds.dims:
            if dim.lower() in ["band", "bands", "variable", "variables"]:
                band_dim = dim
                break

        if band_dim is None:
            # If no band dimension found, try to create one
            data_vars = list(ds.data_vars.keys())
            if len(data_vars) > 1:
                ds = ds.to_array(dim="bands")
            else:
                # Single variable, convert to dataarray
                ds = ds[data_vars[0]]
        else:
            ds = ds.to_array(dim=band_dim)

    return ds


def load_stac(
    url: str,
    spatial_extent: Optional[BoundingBox] = None,
    temporal_extent: Optional[TemporalInterval] = None,
    bands: Optional[list[str]] = None,
    properties: Optional[dict] = None,
    resolution: Optional[float] = None,
    projection: Optional[Union[int, str]] = None,
    resampling: Optional[str] = None,
) -> RasterCube:
    # Check if this is an EOPF STAC URL
    if "stac.core.eopf.eodc.eu" in url:
        logger.info(f"Detected EOPF STAC URL: {url}, using xcube-eopf backend")

        # Extract data_id from URL
        parsed_url = urlparse(url)
        path_parts = PurePosixPath(unquote(parsed_url.path)).parts
        data_id = path_parts[-1] if path_parts else "sentinel-2-l2a"  # default fallback

        #print(properties)

        # Use xcube-eopf for loading
        return _load_with_xcube_eopf(
            data_id=data_id,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            bands=bands,
            **(
                {
                    "resolution": properties["resolution"],
                    "projection": properties["projection"],
                }
                if properties and "resolution" in properties and "projection" in properties
                else {}
            ),
        )

    # Original implementation for non-EOPF STAC URLs
    stac_type = _validate_stac(url)

    # TODO: load_stac should have a parameter to enable scale and offset?

    if isinstance(bands, str):
        bands = [bands]

    if stac_type == "COLLECTION":
        # If query parameters are passed, try to get the parent Catalog if possible/exists, to use the /search endpoint
        if spatial_extent or temporal_extent or bands or properties:
            catalog_url, collection_id = _search_for_parent_catalog(url)

            # Check if we are connecting to Microsoft Planetary Computer, where we need to sign the connection

            modifier = pc.sign_inplace if "planetarycomputer" in catalog_url else None
            catalog = pystac_client.Client.open(catalog_url, modifier=modifier)
            query_params = {"collections": [collection_id]}

            if spatial_extent is not None:
                try:
                    spatial_extent_4326 = spatial_extent
                    if spatial_extent.crs is not None:
                        if not pyproj.crs.CRS(spatial_extent.crs).equals("EPSG:4326"):
                            spatial_extent_4326 = _reproject_bbox(
                                spatial_extent, "EPSG:4326"
                            )
                    bbox = [
                        spatial_extent_4326.west,
                        spatial_extent_4326.south,
                        spatial_extent_4326.east,
                        spatial_extent_4326.north,
                    ]
                    query_params["bbox"] = bbox
                except Exception as e:
                    raise Exception(f"Unable to parse the provided spatial extent: {e}")

            if temporal_extent is not None:
                start_date = (
                    str(temporal_extent[0].to_numpy())
                    if temporal_extent[0] is not None
                    else None
                )
                end_date = (
                    str(temporal_extent[1].to_numpy())
                    if temporal_extent[1] is not None
                    else None
                )
                query_params["datetime"] = [start_date, end_date]

            if properties is not None:
                query_params["query"] = properties

            items = catalog.search(**query_params).item_collection()
        else:
            # Load the whole collection wihout filters
            raise Exception(
                "No parameters for filtering provided. Loading the whole STAC Collection is not supported yet."
            )
    elif stac_type == "ITEM":
        stac_api = pystac_client.stac_api_io.StacApiIO()
        stac_dict = json.loads(stac_api.read_text(url))
        items = [stac_api.stac_object_from_dict(stac_dict)]
    else:
        raise Exception(
            f"The provided URL is a STAC {stac_type}, which is not yet supported. Please provide a valid URL to a STAC Collection or Item."
        )

    available_assets = {tuple(i.assets.keys()) for i in items}
    if (len(available_assets)) > 1:
        raise OpenEOException(
            f"The resulting STAC Items contain two separate set of assets: {available_assets}. We can't load them at the same time."
        )

    available_assets = [x for t in available_assets for x in t]

    # Initialize asset metadata tracking
    asset_scale_offset = {}
    zarr_assets = False
    use_xarray_open_kwargs = False
    use_xarray_storage_options = False

    for asset in available_assets:
        asset_dict = items[0].assets[asset].to_dict()
        asset_scale = 1
        asset_offset = 0
        asset_nodata = None
        asset_dtype = None
        asset_type = None

        if "raster:bands" in asset_dict:
            asset_scale = asset_dict["raster:bands"][0].get("scale", 1)
            asset_offset = asset_dict["raster:bands"][0].get("offset", 0)
            asset_nodata = asset_dict["raster:bands"][0].get("nodata", None)
            asset_dtype = asset_dict["raster:bands"][0].get("data_type", None)

        if "type" in asset_dict:
            asset_type = asset_dict["type"]
            if asset_type == "application/vnd+zarr":
                zarr_assets = True

        if "xarray:open_kwargs" in asset_dict:
            use_xarray_open_kwargs = True
        if "xarray:storage_options" in asset_dict:
            use_xarray_storage_options = True

        asset_scale_offset[asset] = {
            "scale": asset_scale,
            "offset": asset_offset,
            "nodata": asset_nodata,
            "data_type": asset_dtype,
            "type": asset_type,
        }

    item_dict = items[0].to_dict() if items else {}
    available_variables = []
    if "properties" in item_dict and "cube:variables" in item_dict["properties"]:
        available_variables = list(item_dict["properties"]["cube:variables"].keys())

    if bands is not None:
        if zarr_assets and available_variables:
            missing_bands = set(bands) - set(available_variables)
            if missing_bands:
                raise OpenEOException(
                    f"The following requested bands were not found: {missing_bands}. "
                    f"Available bands are: {available_variables}"
                )
        else:
            if len(set(available_assets) & set(bands)) == 0:
                raise OpenEOException(
                    f"The provided bands: {bands} can't be found in the STAC assets: {available_assets}"
                )

    reference_system = None
    if "properties" in item_dict and "cube:dimensions" in item_dict["properties"]:
        for d in item_dict["properties"]["cube:dimensions"]:
            if "reference_system" in item_dict["properties"]["cube:dimensions"][d]:
                reference_system = item_dict["properties"]["cube:dimensions"][d][
                    "reference_system"
                ]
                break

    if zarr_assets:
        datasets = []
        for item in items:
            for asset in item.assets.values():
                kwargs = (
                    asset.extra_fields.get("xarray:open_kwargs", {})
                    if use_xarray_open_kwargs
                    else {"engine": "zarr", "consolidated": True, "chunks": {}}
                )

                if use_xarray_storage_options:
                    storage_opts = asset.extra_fields.get("xarray:storage_options", {})
                    s3_endpoint_url = storage_opts.get("client_kwargs", {}).get(
                        "endpoint_url"
                    )
                    if s3_endpoint_url is not None:
                        kwargs["storage_options"] = {
                            "client_kwargs": {"endpoint_url": s3_endpoint_url}
                        }

                ds = xr.open_dataset(asset.href, **kwargs)
                if bands is not None and available_variables:
                    vars_to_load = [b for b in bands if b in ds.data_vars]
                    ds = ds[vars_to_load]
                datasets.append(ds)

        stack = xr.combine_by_coords(
            datasets, join="exact", combine_attrs="drop_conflicts"
        )
        if not stack.rio.crs:
            stack.rio.write_crs(reference_system, inplace=True)
        stack = stack.to_dataarray(dim="bands")
    else:
        # If at least one band has the nodata field set, we have to apply it at loading time
        apply_nodata = True
        nodata_set = {asset_scale_offset[k]["nodata"] for k in asset_scale_offset}
        dtype_set = {asset_scale_offset[k]["data_type"] for k in asset_scale_offset}
        kwargs = {}

        if resolution is not None:
            kwargs["resolution"] = resolution
        if projection is not None:
            kwargs["crs"] = projection
        if resampling is not None:
            kwargs["resampling"] = resampling

        if len(nodata_set) == 1 and list(nodata_set)[0] == None:
            apply_nodata = False
        if apply_nodata:
            # We can pass only a single nodata value for all the assets/variables/bands https://github.com/opendatacube/odc-stac/issues/147#issuecomment-2005315438
            # Therefore, if we load multiple assets having different nodata values, the first one will be used
            kwargs["nodata"] = list(nodata_set)[0]
            dtype = list(dtype_set)[0]
            if dtype is not None:
                kwargs["nodata"] = np.dtype(dtype).type(kwargs["nodata"])
        # TODO: the dimension names (like "bands") should come from the STAC metadata and not hardcoded
        # Note: unfortunately, converting the dataset to a dataarray, casts all the data types to the same

        if bands is not None:
            stack = odc.stac.load(items, bands=bands, chunks={}, **kwargs).to_dataarray(
                dim="bands"
            )
        else:
            stack = odc.stac.load(items, chunks={}, **kwargs).to_dataarray(dim="bands")

    if spatial_extent is not None:
        stack = filter_bbox(stack, spatial_extent)

    if temporal_extent is not None and (stac_type == "ITEM" or zarr_assets):
        stack = filter_temporal(stack, temporal_extent)

    # If at least one band requires to apply scale and/or offset, the datatype of the whole DataArray must be cast to float -> do not apply it automatically yet. see https://github.com/Open-EO/openeo-processes/issues/503
    # b_dim = stack.openeo.band_dims[0]
    # for b in stack[b_dim]:
    #     scale = asset_scale_offset[b.item(0)]["scale"]
    #     offset = asset_scale_offset[b.item(0)]["offset"]
    #     if scale != 1:
    #         stack.loc[{b_dim: b.item(0)}] *= scale
    #     if offset != 0:
    #         stack.loc[{b_dim: b.item(0)}] += offset

    return stack


def load_url(url: str, format: str, options={}):
    import geopandas as gpd
    import requests

    if format not in ["GeoJSON", "JSON", "Parquet"]:
        raise Exception(
            f"FormatUnsuitable: Data can't be loaded with the requested input format {format}."
        )

    response = requests.get(url)
    if not response.status_code < 400:
        raise Exception(f"Provided url {url} unavailable. ")

    if "JSON" in format:
        url_json = response.json()

    if format == "GeoJSON":
        for feature in url_json.get("features", {}):
            if "properties" not in feature:
                feature["properties"] = {}
            elif feature["properties"] is None:
                feature["properties"] = {}
        if isinstance(url_json.get("crs", {}), dict):
            crs = url_json.get("crs", {}).get("properties", {}).get("name", 4326)
        else:
            crs = int(url_json.get("crs", {}))
        logger.info(f"CRS in geometries: {crs}.")

        gdf = gpd.GeoDataFrame.from_features(url_json, crs=crs)

    elif "Parquet" in format:
        import os

        import geoparquet as gpq

        file_name = url.split("/")[-1]

        with open(file_name, "wb") as file:
            file.write(response.content)

        file_size = os.path.getsize(file_name)
        if file_size > 0:
            logger.info(f"File downloaded successfully. File size: {file_size} bytes")

        gdf = gpq.read_geoparquet(file_name)
        os.system(f"rm -rf {file_name}")

    elif format == "JSON":
        return url_json

    import xvec

    if not hasattr(gdf, "crs"):
        gdf = gdf.set_crs("epsg:4326")

    columns = gdf.columns.values
    variables = []
    for geom in columns:
        if geom in [
            "geometry",
            "geometries",
        ]:
            geo_column = geom
        else:
            variables.append(geom)
    cube = xr.Dataset(
        data_vars={
            variable: ([geo_column], gdf[variable].values) for variable in variables
        },
        coords={geo_column: gdf[geo_column].values},
    ).xvec.set_geom_indexes(geo_column, crs=gdf.crs)

    return cube
