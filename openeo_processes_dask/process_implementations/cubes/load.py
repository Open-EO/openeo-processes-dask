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


def _validate_stac_and_get_type(url):
    """Validate STAC URL and return asset type and validator object."""
    logger.debug(f"Validating the provided STAC url: {url}")
    stac_validator_obj = stac_validator.StacValidate(url)
    is_valid_stac = stac_validator_obj.run()

    if not is_valid_stac:
        raise Exception(
            f"The provided link is not a valid STAC. stac-validator message: {stac_validator_obj.message}"
        )

    if len(stac_validator_obj.message) == 1:
        try:
            return stac_validator_obj.message[0]["asset_type"], stac_validator_obj
        except:
            raise Exception(
                f"stac-validator returned an error: {stac_validator_obj.message}"
            )
    else:
        raise Exception(
            f"stac-validator returned multiple items, not supported yet. {stac_validator_obj.message}"
        )


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
            asset_type = _validate_stac_and_get_type(catalog_url)[0]
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


def _get_dimension_names_from_stac(stac_validator_obj):
    """Extract dimension names from STAC Collection's cube:dimensions using stac_validator object."""
    # Default canonical openEO dimension names
    dim_names = {"x": "x", "y": "y", "t": "t", "bands": "bands"}

    try:
        # Get the STAC content from the validator object
        stac_content = stac_validator_obj.stac_content

        # Check if cube:dimensions exists in the STAC content
        if "cube:dimensions" in stac_content:
            cube_dims = stac_content["cube:dimensions"]

            # Extract dimension names based on axis/type
            for dim_name, dim_info in cube_dims.items():
                if "axis" in dim_info and dim_info["axis"] == "x":
                    dim_names["x"] = dim_name
                elif "axis" in dim_info and dim_info["axis"] == "y":
                    dim_names["y"] = dim_name
                elif "type" in dim_info and dim_info["type"] == "temporal":
                    dim_names["t"] = dim_name
                elif "type" in dim_info and dim_info["type"] == "bands":
                    dim_names["bands"] = dim_name

            # Store band case mapping if available
            if (
                dim_names["bands"] in cube_dims
                and "values" in cube_dims[dim_names["bands"]]
            ):
                available_bands = cube_dims[dim_names["bands"]]["values"]
                dim_names["_band_case_map"] = {
                    band.lower(): band for band in available_bands
                }
                dim_names["_available_bands"] = available_bands
        else:
            # If no cube:dimensions, try to get from eo:bands in summaries
            if "summaries" in stac_content and "eo:bands" in stac_content["summaries"]:
                available_bands = [
                    band.get("name")
                    for band in stac_content["summaries"]["eo:bands"]
                    if band.get("name")
                ]
                if available_bands:
                    dim_names["_band_case_map"] = {
                        band.lower(): band for band in available_bands
                    }
                    dim_names["_available_bands"] = available_bands

    except Exception as e:
        logger.debug(f"Could not extract dimension names from STAC content: {e}")
        # Fall back to default names

    return dim_names


def _normalize_band_names(bands, band_case_map=None, available_bands_list=None):
    """Normalize band names to match STAC collection casing."""
    if not bands or not band_case_map:
        return bands

    normalized_bands = []
    for band in bands:
        band_lower = band.lower()
        if band_lower in band_case_map:
            normalized_bands.append(band_case_map[band_lower])
        else:
            normalized_bands.append(band)
            logger.warning(f"Band '{band}' not found in available bands, using as-is")
    return normalized_bands


def _spatial_extent_to_bbox(spatial_extent):
    """Convert spatial extent to bbox with proper projection handling."""
    if spatial_extent is None:
        return None, None, None

    try:
        from pyproj import CRS

        crs_obj = CRS(spatial_extent.crs) if spatial_extent.crs else CRS("EPSG:4326")
        epsg = crs_obj.to_epsg()

        if epsg == 4326:
            projection = "EPSG:4326"
            resolution = 10 / 111320
        elif epsg and ((32601 <= epsg <= 32660) or (32701 <= epsg <= 32760)):
            # Any UTM zone (north/south)
            projection = f"EPSG:{epsg}"
            resolution = 10
        else:
            raise ValueError(
                f"Unsupported CRS: {crs_obj.to_string()} "
                "(only EPSG:4326 or UTM EPSG:32601–32660 / 32701–32760 are supported)"
            )

        bbox = [
            spatial_extent.west,
            spatial_extent.south,
            spatial_extent.east,
            spatial_extent.north,
        ]

        return bbox, projection, resolution
    except Exception as e:
        raise Exception(f"Unable to parse the provided spatial extent: {e}")


def _temporal_extent_to_range(temporal_extent):
    """Convert temporal extent to time range strings."""
    if temporal_extent is None:
        return None

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

    return [start_date, end_date]


def _extract_asset_metadata(items):
    """Extract scale, offset, and other metadata from STAC assets."""
    if not items:
        return {}, False, False, False

    available_assets = {tuple(i.assets.keys()) for i in items}
    if len(available_assets) > 1:
        raise OpenEOException(
            f"The resulting STAC Items contain two separate set of assets: {available_assets}. We can't load them at the same time."
        )

    available_assets = [x for t in available_assets for x in t]

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

    return (
        asset_scale_offset,
        zarr_assets,
        use_xarray_open_kwargs,
        use_xarray_storage_options,
    )


def _create_dimension_mapping(stack, target_x, target_y, target_t, target_b):
    """Create dimension renaming mapping based on target names."""
    rename_dict = {}

    # Check each dimension in the loaded data
    for dim in stack.dims:
        dim_lower = dim.lower()

        # Map based on common patterns
        if dim_lower in ["x", "longitude", "lon"] and dim != target_x:
            rename_dict[dim] = target_x
        elif dim_lower in ["y", "latitude", "lat"] and dim != target_y:
            rename_dict[dim] = target_y
        elif dim_lower in ["t", "time", "date", "datetime"] and dim != target_t:
            rename_dict[dim] = target_t
        elif (
            dim_lower in ["band", "bands", "variable", "variables"] and dim != target_b
        ):
            rename_dict[dim] = target_b

    # Also handle case-insensitive exact matches
    for dim in stack.dims:
        if dim.lower() == target_x.lower() and dim != target_x:
            rename_dict[dim] = target_x
        elif dim.lower() == target_y.lower() and dim != target_y:
            rename_dict[dim] = target_y
        elif dim.lower() == target_t.lower() and dim != target_t:
            rename_dict[dim] = target_t
        elif dim.lower() == target_b.lower() and dim != target_b:
            rename_dict[dim] = target_b

    return rename_dict


def _reorder_dimensions(stack, target_t, target_b, target_y, target_x):
    """Reorder dimensions to openEO canonical order (t, bands, y, x)."""
    current_dims = list(stack.dims)
    desired_order = []

    # Try to get t, bands, y, x in that order (openEO canonical)
    for dim_name in [target_t, target_b, target_y, target_x]:
        if dim_name in stack.dims:
            desired_order.append(dim_name)

    # Add any remaining dimensions
    for dim in current_dims:
        if dim not in desired_order:
            desired_order.append(dim)

    # Only transpose if order is different
    if current_dims != desired_order:
        logger.info(f"Reordering dimensions from {current_dims} to {desired_order}")
        try:
            stack = stack.transpose(*desired_order)
        except Exception as e:
            logger.warning(f"Could not reorder dimensions: {e}")

    return stack


def _load_with_xcube_eopf(
    data_id: str,
    spatial_extent: Optional[BoundingBox] = None,
    temporal_extent: Optional[TemporalInterval] = None,
    bands: Optional[list[str]] = None,
    resolution: Optional[float] = None,
    projection: Optional[Union[int, str]] = None,
    dim_names: Optional[dict[str, str]] = None,
) -> RasterCube:
    """Load data using xcube-eopf package for EOPF STAC endpoints."""
    try:
        from xcube.core.store import new_data_store
    except ImportError:
        raise ImportError(
            "xcube-eopf package is required for loading EOPF STAC data. "
            "Please install it with: pip install xcube-eopf"
        )

    from pyproj import CRS

    # Convert spatial extent
    bbox, projection_used, resolution_used = _spatial_extent_to_bbox(spatial_extent)

    # Convert temporal extent
    time_range = _temporal_extent_to_range(temporal_extent)

    # Set CRS
    crs = (
        projection
        if projection
        else (projection_used if projection_used else "EPSG:4326")
    )

    # Set resolution
    spatial_res = (
        resolution
        if resolution
        else (resolution_used if resolution_used else 10 / 111320)
    )

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


def _process_eopf_cube(eopf_cube, target_x, target_y, target_t, target_b):
    """Process and rename dimensions for EOPF cube output."""
    rename_dict = {}

    # Map spatial dimensions - EOPF often returns 'lon' and 'lat'
    spatial_mapping = {
        "lon": target_x,
        "longitude": target_x,
        "x": target_x,
        "lat": target_y,
        "latitude": target_y,
        "y": target_y,
    }

    for dim in eopf_cube.dims:
        dim_lower = dim.lower()
        if dim_lower in spatial_mapping:
            rename_dict[dim] = spatial_mapping[dim_lower]

    # Map time dimension if present
    time_mapping = {"time": target_t, "t": target_t, "date": target_t}

    for dim in eopf_cube.dims:
        dim_lower = dim.lower()
        if dim_lower in time_mapping and target_t != dim:
            rename_dict[dim] = target_t
            break

    # Map band dimension if present
    band_mapping = {
        "band": target_b,
        "bands": target_b,
        "variable": target_b,
        "variables": target_b,
    }

    for dim in eopf_cube.dims:
        dim_lower = dim.lower()
        if dim_lower in band_mapping and target_b != dim:
            rename_dict[dim] = target_b
            break

    if rename_dict:
        eopf_cube = eopf_cube.rename(rename_dict)

    # Reorder dimensions
    eopf_cube = _reorder_dimensions(eopf_cube, target_t, target_b, target_y, target_x)

    return eopf_cube


def _process_stac_collection(
    url, spatial_extent, temporal_extent, bands, properties, catalog_url, collection_id
):
    """Process STAC collection with filters."""
    # Check if we are connecting to Microsoft Planetary Computer
    modifier = pc.sign_inplace if "planetarycomputer" in catalog_url else None
    catalog = pystac_client.Client.open(catalog_url, modifier=modifier)
    query_params = {"collections": [collection_id]}

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
    return items


def _process_stac_item(
    url, target_x, target_y, target_t, target_b, band_case_map, available_bands_list
):
    """Process STAC item and update dimension names from collection if available."""
    stac_api = pystac_client.stac_api_io.StacApiIO()
    stac_dict = json.loads(stac_api.read_text(url))
    items = [stac_api.stac_object_from_dict(stac_dict)]

    # Try to get collection URL from the item
    item = items[0]
    collection_url = None
    if hasattr(item, "collection_id") and item.collection_id:
        # Try to construct collection URL
        parsed = urlparse(url)
        # Remove the item ID from path to get collection
        path = PurePosixPath(parsed.path)
        if len(path.parts) > 1:
            collection_path = str(path.parent)
            collection_url = f"{parsed.scheme}://{parsed.netloc}{collection_path}"
            # Re-validate the collection to get its dimension names
            try:
                coll_validator = stac_validator.StacValidate(collection_url)
                if coll_validator.run():
                    coll_dim_names = _get_dimension_names_from_stac(coll_validator)
                    # Update targets with collection dimension names
                    target_x = coll_dim_names.get("x", target_x)
                    target_y = coll_dim_names.get("y", target_y)
                    target_t = coll_dim_names.get("t", target_t)
                    target_b = coll_dim_names.get("bands", target_b)

                    # Update band case mapping
                    band_case_map = coll_dim_names.get("_band_case_map", band_case_map)
                    available_bands_list = coll_dim_names.get(
                        "_available_bands", available_bands_list
                    )
            except:
                pass  # Keep existing dim_names if collection fetch fails

    return (
        items,
        target_x,
        target_y,
        target_t,
        target_b,
        band_case_map,
        available_bands_list,
    )


def _load_zarr_assets(
    items,
    bands,
    target_b,
    reference_system,
    use_xarray_open_kwargs,
    use_xarray_storage_options,
):
    """Load Zarr assets from STAC items."""
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

            # Get available variables from dataset
            available_variables = list(ds.data_vars.keys())

            if bands is not None and available_variables:
                # Use case-insensitive matching for bands
                vars_to_load = []
                for band in bands:
                    # Try exact match first
                    if band in ds.data_vars:
                        vars_to_load.append(band)
                    else:
                        # Try case-insensitive match
                        for var_name in ds.data_vars:
                            if var_name.lower() == band.lower():
                                vars_to_load.append(var_name)
                                break

                if vars_to_load:
                    ds = ds[vars_to_load]
                else:
                    logger.warning(
                        f"No matching bands found in dataset for requested bands: {bands}"
                    )
            datasets.append(ds)

    if datasets:
        stack = xr.combine_by_coords(
            datasets, join="exact", combine_attrs="drop_conflicts"
        )
        if not stack.rio.crs:
            stack.rio.write_crs(reference_system, inplace=True)
        stack = stack.to_dataarray(dim=target_b)
    else:
        raise NoDataAvailable("No data could be loaded from the STAC items.")

    return stack


def _load_non_zarr_assets(
    items, bands, target_b, resolution, projection, resampling, asset_scale_offset
):
    """Load non-Zarr assets from STAC items."""
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
        # We can pass only a single nodata value for all the assets/variables/bands
        kwargs["nodata"] = list(nodata_set)[0]
        dtype = list(dtype_set)[0]
        if dtype is not None:
            kwargs["nodata"] = np.dtype(dtype).type(kwargs["nodata"])

    if bands is not None:
        stack = odc.stac.load(items, bands=bands, chunks={}, **kwargs).to_dataarray(
            dim=target_b
        )
    else:
        stack = odc.stac.load(items, chunks={}, **kwargs).to_dataarray(dim=target_b)

    return stack


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
    # Validate STAC and get validator object
    asset_type, stac_validator_obj = _validate_stac_and_get_type(url)

    # Get dimension names from the STAC validator object
    dim_names = _get_dimension_names_from_stac(stac_validator_obj)
    target_x = dim_names.get("x", "x")
    target_y = dim_names.get("y", "y")
    target_t = dim_names.get("t", "t")
    target_b = dim_names.get("bands", "bands")

    # Store band case mapping for use later
    band_case_map = dim_names.get("_band_case_map")
    available_bands_list = dim_names.get("_available_bands")

    logger.info(
        f"Extracted dimension names: x={target_x}, y={target_y}, t={target_t}, bands={target_b}"
    )

    # Apply band name case normalization if needed
    if bands and band_case_map:
        bands = _normalize_band_names(bands, band_case_map)

    # Check if this is an EOPF STAC URL
    if "stac.core.eopf.eodc.eu" in url:
        logger.info(f"Detected EOPF STAC URL: {url}, using xcube-eopf backend")

        # Extract data_id from URL
        parsed_url = urlparse(url)
        path_parts = PurePosixPath(unquote(parsed_url.path)).parts
        data_id = path_parts[-1] if path_parts else "sentinel-2-l2a"  # default fallback

        # Use xcube-eopf for loading
        eopf_cube = _load_with_xcube_eopf(
            data_id=data_id,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            bands=bands,
            dim_names=dim_names,
        )

        # Process and rename dimensions
        eopf_cube = _process_eopf_cube(
            eopf_cube, target_x, target_y, target_t, target_b
        )
        return eopf_cube

    # Original implementation for non-EOPF STAC URLs
    # stac_type is already set from validator above

    # TODO: load_stac should have a parameter to enable scale and offset?

    if isinstance(bands, str):
        bands = [bands]

    # Process based on asset type
    if asset_type == "COLLECTION":
        # If query parameters are passed, try to get the parent Catalog if possible/exists, to use the /search endpoint
        if spatial_extent or temporal_extent or bands or properties:
            catalog_url, collection_id = _search_for_parent_catalog(url)
            items = _process_stac_collection(
                url,
                spatial_extent,
                temporal_extent,
                bands,
                properties,
                catalog_url,
                collection_id,
            )
        else:
            # Load the whole collection without filters
            raise Exception(
                "No parameters for filtering provided. Loading the whole STAC Collection is not supported yet."
            )
    elif asset_type == "ITEM":
        (
            items,
            target_x,
            target_y,
            target_t,
            target_b,
            band_case_map,
            available_bands_list,
        ) = _process_stac_item(
            url,
            target_x,
            target_y,
            target_t,
            target_b,
            band_case_map,
            available_bands_list,
        )
    else:
        raise Exception(
            f"The provided URL is a STAC {asset_type}, which is not yet supported. Please provide a valid URL to a STAC Collection or Item."
        )

    # Extract asset metadata
    (
        asset_scale_offset,
        zarr_assets,
        use_xarray_open_kwargs,
        use_xarray_storage_options,
    ) = _extract_asset_metadata(items)

    # Get item dictionary for additional metadata
    item_dict = items[0].to_dict() if items else {}

    # Get available variables
    available_variables = []
    if "properties" in item_dict and "cube:variables" in item_dict["properties"]:
        available_variables = list(item_dict["properties"]["cube:variables"].keys())

    # FIX: Apply band case normalization for available_variables too
    if available_bands_list and available_variables:
        available_variables = available_bands_list

    # Validate bands if provided
    if bands is not None:
        if zarr_assets and available_variables:
            # Use case-insensitive matching for bands
            available_vars_lower = [v.lower() for v in available_variables]
            missing_bands = []
            for band in bands:
                if band.lower() not in available_vars_lower:
                    missing_bands.append(band)

            if missing_bands:
                raise OpenEOException(
                    f"The following requested bands were not found: {missing_bands}. "
                    f"Available bands are: {available_variables}"
                )
        else:
            available_assets = list(asset_scale_offset.keys())
            if len(set(available_assets) & set(bands)) == 0:
                raise OpenEOException(
                    f"The provided bands: {bands} can't be found in the STAC assets: {available_assets}"
                )

    # Get reference system from cube:dimensions if available
    reference_system = None
    if "properties" in item_dict and "cube:dimensions" in item_dict["properties"]:
        for d in item_dict["properties"]["cube:dimensions"]:
            if "reference_system" in item_dict["properties"]["cube:dimensions"][d]:
                reference_system = item_dict["properties"]["cube:dimensions"][d][
                    "reference_system"
                ]
                break

    # Load data based on asset type
    if zarr_assets:
        stack = _load_zarr_assets(
            items,
            bands,
            target_b,
            reference_system,
            use_xarray_open_kwargs,
            use_xarray_storage_options,
        )
    else:
        stack = _load_non_zarr_assets(
            items,
            bands,
            target_b,
            resolution,
            projection,
            resampling,
            asset_scale_offset,
        )

    # Apply spatial filter if needed
    if spatial_extent is not None:
        stack = filter_bbox(stack, spatial_extent)

    # Apply temporal filter if needed
    if temporal_extent is not None and (asset_type == "ITEM" or zarr_assets):
        stack = filter_temporal(stack, temporal_extent)

    # Apply dimension renaming to match STAC collection dimension names
    if hasattr(stack, "dims"):
        rename_dict = _create_dimension_mapping(
            stack, target_x, target_y, target_t, target_b
        )

        # Apply renaming if needed
        if rename_dict:
            logger.info(f"Renaming dimensions: {rename_dict}")
            stack = stack.rename(rename_dict)

        # Reorder dimensions
        stack = _reorder_dimensions(stack, target_t, target_b, target_y, target_x)

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
