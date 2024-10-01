"""
OpenEO xvec example on a small dataset around Bolzano
"""

import geopandas as gpd
from openeo.local import LocalConnection

local_conn = LocalConnection("./")

url = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"
spatial_extent = {"east": 11.40, "north": 46.52, "south": 46.46, "west": 11.25}
temporal_extent = ["2022-06-01", "2022-06-30"]
bands = ["red"]
properties = {"eo:cloud_cover": dict(lt=80)}

s2_datacube = local_conn.load_stac(
    url=url,
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    bands=bands,
    properties=properties,
)

s2_datacube = s2_datacube.resample_spatial(
    projection="EPSG:4326", resolution=0.0001).drop_dimension("band")

polys_path = "../data/sample_polygons.geojson"
polys = gpd.read_file(polys_path)
polys = polys.__geo_interface__

aggregate = s2_datacube.aggregate_spatial(geometries=polys, reducer="mean")

def run_aggregate():
    aggregate.execute().compute()

if __name__ == "__main__":
    run_aggregate()