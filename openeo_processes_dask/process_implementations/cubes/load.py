import datetime
import json
import logging
from collections.abc import Iterator
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


def load_stac(
    url: str,
    spatial_extent: Optional[BoundingBox] = None,
    temporal_extent: Optional[TemporalInterval] = None,
    bands: Optional[list[str]] = None,
    properties: Optional[dict] = None,
) -> RasterCube:
    asset_type = _validate_stac(url)

    # TODO: load_stac should have a parameter to enable scale and offset
    # apply_offset = False
    # apply_scale = False

    if asset_type == "COLLECTION":
        # If query parameters are passed, try to get the parent Catalog if possible/exists, to use the /search endpoint
        if spatial_extent or temporal_extent or bands or properties:
            # If query parameters are passed, try to get the parent Catalog if possible/exists, to use the /search endpoint
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
                start_date = None
                end_date = None
                if temporal_extent[0] is not None:
                    start_date = str(temporal_extent[0].to_numpy())
                if temporal_extent[1] is not None:
                    end_date = str(temporal_extent[1].to_numpy())
                query_params["datetime"] = [start_date, end_date]

            if properties is not None:
                query_params["query"] = properties

            items = catalog.search(**query_params).item_collection()

        else:
            # Load the whole collection wihout filters
            raise Exception(
                f"No parameters for filtering provided. Loading the whole STAC Collection is not supported yet."
            )

    elif asset_type == "ITEM":
        stac_api = pystac_client.stac_api_io.StacApiIO()
        stac_dict = json.loads(stac_api.read_text(url))
        items = [stac_api.stac_object_from_dict(stac_dict)]

    else:
        raise Exception(
            f"The provided URL is a STAC {asset_type}, which is not yet supported. Please provide a valid URL to a STAC Collection or Item."
        )

    asset_scale_offset = {}
    for asset in items[0].assets:
        asset_scale = 1
        asset_offset = 0
        asset_nodata = None
        asset_dtype = None
        asset_dict = items[0].assets[asset].to_dict()
        if "raster:bands" in asset_dict:
            asset_scale = asset_dict["raster:bands"][0].get("scale", 1)
            asset_offset = asset_dict["raster:bands"][0].get("offset", 0)
            asset_nodata = asset_dict["raster:bands"][0].get("nodata", None)
            asset_dtype = asset_dict["raster:bands"][0].get("data_type", None)
        asset_scale_offset[asset] = {
            "scale": asset_scale,
            "offset": asset_offset,
            "nodata": asset_nodata,
            "data_type": asset_dtype,
        }

    # If at least one band has the nodata field set, we have to apply it at loading time
    apply_nodata = True
    nodata_set = {asset_scale_offset[k]["nodata"] for k in asset_scale_offset}
    dtype_set = {asset_scale_offset[k]["data_type"] for k in asset_scale_offset}
    kwargs = {}
    if len(nodata_set) == 1 and list(nodata_set)[0] == None:
        apply_nodata = False
    if apply_nodata:
        # We can pass only a single nodata value for all the assets/variables/bands https://github.com/opendatacube/odc-stac/issues/147#issuecomment-2005315438
        # Therefore, if we load multiple assets having different nodata values, the first one will be used
        kwargs["nodata"] = list(nodata_set)[0]
        dtype = list(dtype_set)[0]
        if dtype is not None:
            kwargs["nodata"] = np.dtype(dtype).type(kwargs["nodata"])

    print(apply_nodata)
    print("NO DATA KWARGS:", kwargs)
    if bands is not None:
        stack = odc.stac.load(items, bands=bands, chunks={}, **kwargs).to_dataarray(
            dim="band"
        )
    else:
        stack = odc.stac.load(items, chunks={}, **kwargs).to_dataarray(dim="band")

    if spatial_extent is not None:
        stack = filter_bbox(stack, spatial_extent)

    if temporal_extent is not None and asset_type == "ITEM":
        stack = filter_temporal(stack, temporal_extent)

    # If at least one band requires to apply scale and/or offset, the datatype of the whole DataArray must be cast to float
    # apply_scale = True
    # scale_set = set([asset_scale_offset[k]["scale"] for k in asset_scale_offset])
    # if len(scale_set) == 1 and list(scale_set)[0] == 1:
    #     apply_scale = False

    # apply_offset = True
    #     offset_set = set([asset_scale_offset[k]["offset"] for k in asset_scale_offset])
    #     if len(offset_set) == 1 and list(offset_set)[0] == 0:
    #         apply_offset = False

    #     if apply_offset or apply_scale:
    #         stack = stack.astype(float)

    b_dim = stack.openeo.band_dims[0]
    for b in stack[b_dim]:
        scale = asset_scale_offset[b.item(0)]["scale"]
        offset = asset_scale_offset[b.item(0)]["offset"]
        if scale != 1:
            stack.loc[{b_dim: b.item(0)}] *= scale
        if offset != 0:
            stack.loc[{b_dim: b.item(0)}] += offset

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
