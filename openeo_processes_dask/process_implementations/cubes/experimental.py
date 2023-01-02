import json
import logging
import urllib
from typing import Optional

import dask_geopandas
import geopandas as gpd
from openeo_pg_parser_networkx.pg_schema import DEFAULT_CRS

from openeo_processes_dask.process_implementations.data_model import VectorCube

logger = logging.getLogger(__name__)


__all__ = ["load_vector_cube"]


def load_vector_cube(
    URL: Optional[str] = None,
    filename: Optional[str] = None,
    geometries: Optional[dict] = None,
    **kwargs,
) -> VectorCube:
    if not (URL or filename):
        raise Exception(
            "One of these parameters needs to be provided: <job_id>, <URL>, <filename>"
        )

    # This is a workaround for an implementation gap in the openeo-python-client, where geojson is passed inline through the filename argument.
    # TODO: Remove once https://github.com/Open-EO/openeo-python-client/issues/104 is resolved.
    if filename is not None:
        logger.info("Attempting to load vector cube from filename.")
        try:
            geometries = json.loads(filename)
        except Exception as e:
            logger.warning(f"Could not load vector cube from filename {filename}", e)

    # TODO: Loading random files from untrusted URLs is dangerous, this has to be rethought going forward!
    if URL is not None:
        try:
            response = urllib.request.urlopen(URL)
            geometries = json.loads(response.read())
        except json.JSONDecodeError as j:
            logger.warning(f"Could not decode JSON from URL {URL}", j)
        except Exception as e:
            logger.warning(
                f"Unexpected error when trying to load vector data from provided URL {URL}",
                e,
            )

    if geometries is None:
        raise Exception("Could not load the provided geometries!")

    # Each feature must have a properties field, even if there is no property
    # This is necessary due to this bug in geopandas: https://github.com/geopandas/geopandas/pull/2243
    # TODO: Remove once this is solved.
    for feature in geometries["features"]:
        if "properties" not in feature:
            feature["properties"] = {}
        elif feature["properties"] is None:
            feature["properties"] = {}

    geometries_crs = (
        geometries.get("crs", {}).get("properties", {}).get("name", DEFAULT_CRS)
    )

    try:
        gdf = gpd.GeoDataFrame.from_features(geometries, crs=geometries_crs)
        return dask_geopandas.from_geopandas(gdf, npartitions=1)
    except Exception as e:
        logger.error("Could not parse provided vector data into dask_geopandas.", e)
        raise e
