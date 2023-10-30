import datetime
import json
import logging
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import planetary_computer as pc
import pyproj
import pystac_client
import stackstac
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
        items = stac_api.stac_object_from_dict(stac_dict)

    else:
        raise Exception(
            f"The provided URL is a STAC {asset_type}, which is not yet supported. Please provide a valid URL to a STAC Collection or Item."
        )

    if bands is not None:
        stack = stackstac.stack(items, assets=bands)
    else:
        stack = stackstac.stack(items)

    if spatial_extent is not None:
        stack = filter_bbox(stack, spatial_extent)

    if temporal_extent is not None and asset_type == "ITEM":
        stack = filter_temporal(stack, temporal_extent)

    return stack
