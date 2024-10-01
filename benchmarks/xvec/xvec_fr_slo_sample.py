"""
OpenEO xvec example using a large area with 2 polygons
on the french side of alps and 2 polygons on the slovenian side.
"""
import geopandas as gpd
from openeo.local import LocalConnection
from memory_profiler import profile

local_conn = LocalConnection("./")

polys_path = "../data/fr_slo.geojson"

polys = gpd.read_file(polys_path)
polys = polys.__geo_interface__

url = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"
temporal_extent = ["2022-06-01", "2022-06-30"]
spatial_extent = {"west": 5.9139132444292954,"south": 45.0965802219263,"east": 13.829701504566117,"north": 46.954031232759576}
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

aggregate = s2_datacube.aggregate_spatial(geometries=polys, reducer="mean")

def execute():
	aggregate.execute().compute()


if __name__ == "__main__":
    execute()

    