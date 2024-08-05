import geopandas as gpd
from openeo.local import LocalConnection
from exactextract import exact_extract

local_conn = LocalConnection("./")

URL = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"
SPATIAL_EXTENT = {"east": 11.8638, "north": 46.7135, "south": 46.3867, "west": 10.7817}
TEMPORAL_EXTENT = ["2022-06-01", "2022-06-30"]
BANDS = ["red"]
PROPERTIES = {"eo:cloud_cover": dict(lt=80)}

s2_datacube = local_conn.load_stac(
    url=URL,
    spatial_extent=SPATIAL_EXTENT,
    temporal_extent=TEMPORAL_EXTENT,
    bands=BANDS,
    properties=PROPERTIES,
)

s2_datacube = s2_datacube.resample_spatial(projection="EPSG:4326",resolution=0.0001).drop_dimension("band")
data = s2_datacube.execute()

polys = gpd.read_file("./data/alto_adige.geojson")

def run_extract():
    exact_extract(data, polys, 'mean')

if __name__ == "__main__":
    run_extract()