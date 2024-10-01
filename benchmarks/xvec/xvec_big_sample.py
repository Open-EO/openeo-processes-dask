"""
OpenEO xvec example using 116 polygons of the
Alto Adige region in Italy.
"""

import geopandas as gpd
from openeo.local import LocalConnection

local_conn = LocalConnection("./")

url = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"
spatial_extent = {"east": 11.8638, "north": 46.7135, "south": 46.3867, "west": 10.7817}
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
    projection="EPSG:4326", resolution=0.0001
).drop_dimension("band")

polys = gpd.read_file("./data/alto_adige.geojson")
polys = polys.__geo_interface__

aggregate = s2_datacube.aggregate_spatial(geometries=polys, reducer="mean")

def run_aggregate():
    aggregate.execute().compute()

run_aggregate()

if __name__ == "__main__":
    run_aggregate()