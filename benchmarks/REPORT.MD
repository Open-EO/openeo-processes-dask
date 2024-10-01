# Aggregate Spatial Analysis

## Disclaimer

**A lot has changed in the testing setup so the documentation needs to be updated.**

The latest scripts for `xvec` and `exactextract` can be found in:

- `benchmarks/xvec/`
- `benchmarks/exactextract/`

Data still remains in:

- `benchmarks/data/`

Although not maintained anymore `memory_profiler` is still working and was used.

Profiling decorator has been added to the `aggregate_spatial` function to enable line by line profiling.

`openeo-processes-dask/openeo_processes_dask/process_implementations/cubes/aggregate.py`

Example to run the scripts with `memory_profiler`:

```bash
mprof run xvec_small_sample.py
```

Dont forget to install all extras when installing with poetry to get the `benchmark` extras and all other requirements.

```bash
poetry install --all-extras
```

## Table of Contents

1. [Introduction](#10-introduction)
2. [Methodology](#20-methodology)
3. [Datasets](#30-datasets)
4. [Initial Results (OUTDATED)](#40-initial-results-outdated)
    1. [Small Dataset (2 polygons)](#41-small-dataset-2-polygons)
        1. [OpenEO xvec](#411-openeo-xvec)
        2. [OpenEO exactextract](#412-openeo-exactextract)
    2. [Large Dataset (116 polygons)](#42-large-dataset-116-polygons)
        1. [OpenEO xvec](#421-openeo-xvec)
        2. [OpenEO exactextract](#422-openeo-exactextract)
5. [Conclusion and Comments](#50-conclusion-and-comments)

## 1.0 Introduction

The purpose of this benchmark is to quantify the performance of the OpenEO aggregate_spatial process. Current implementation uses the `xvec` library. Issues arise if a large vector dataset is used. The kernel tends to crash, suggesting the current implementation is not efficient enough. The benchmark will test the performance of the current implementation and compare it to a new implementation using the `exactextract` library.

## 2.0 Methodology

TODO

## 3.0 Datasets

For the purpose of testing the performance we have obtained two datasets. The first is a very small dataset containing two polygons in the Bolzano area. The second dataset is a larger dataset containing the Alto Adige region.

The first dataset: `sample_polygons.geojson` has been provided by @clausmichele

- 2 polygons

The second dataset: `alto_adige.geojson` comes from the Openpolis github repository: `https://github.com/openpolis/geojson-italy`

- 116 polygons

## 4.0 Initial Results (Outdated)

The sections below provide some results for the `small` and `large` dataset. We are comparing the current OpenEO `xvec` implementation first, followed by utilizing `exactextract` to perform the same operation.

Quick summary of the results:

| Metric | Small Dataset (2 polygons) | Large Dataset (116 polygons) |
|  ---  |  ---  |  ---  |
|OpenEO xvec| 20-27 seconds | 1057 seconds / 17 minutes |
|OpenEO exactextract| 15-25 seconds | 1085 seconds / 18 minutes |

### 4.1 Small Dataset (2 polygons)

#### 4.1.1 OpenEO xvec

```python
from profiler import Profiler

import json
import fiona
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
    projection="EPSG:4326", resolution=0.0001
).drop_dimension("band")

polys_path = "./data/sample_polygons.geojson"

polys = gpd.read_file(polys_path)

# Don't know why this is needed but it is.
# https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.__geo_interface__.html
polys = polys.__geo_interface__ 

aggregate = s2_datacube.aggregate_spatial(geometries=polys, reducer="mean")

@Profiler(reruns=1, sample_interval=1, log_file="sm_xvec.csv")
def run_aggregate():
    aggregate.execute()

run_aggregate()
```

| Metric | Value |
|  ---  |  ---  |
|Run_ID| X |
|Timestamp| X |
|Sample_Timestamp| X |
|CPU_Usage_%| 24% |
|Time_Taken| 20-27 seconds |
|Median_Memory_Usage_MB| 45 - 65 MB |

#### 4.1.2 OpenEO exactextract

```python
from profiler import Profiler

import json
import geopandas as gpd
from openeo.local import LocalConnection
from exactextract import exact_extract

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

s2_datacube = s2_datacube.resample_spatial(projection="EPSG:4326",resolution=0.0001).drop_dimension("band")
data = s2_datacube.execute()

polys = gpd.read_file("./data/sample_polygons.geojson")

@Profiler(reruns=1, sample_interval=1)
def run_extract():
    exact_extract(data, polys, 'mean')

run_extract()
```

| Metric | Value |
|  ---  |  ---  |
|Run_ID| X |
|Timestamp| X |
|Sample_Timestamp| X |
|CPU_Usage_%| 28% |
|Time_Taken| 15-25 seconds |
|Median_Memory_Usage_MB| 40 - 60 MB |

### 4.2 Large Dataset (116 polygons)

#### 4.2.1 OpenEO xvec

```python
from profiler import Profiler

import json
import geopandas as gpd
from openeo.local import LocalConnection

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

s2_datacube = s2_datacube.resample_spatial(
    projection="EPSG:4326", resolution=0.0001).drop_dimension("band")

polys = gpd.read_file("./data/alto_adige.geojson")
polys = polys.__geo_interface__

aggregate = s2_datacube.aggregate_spatial(geometries=polys, reducer="mean")


@Profiler(reruns=1, sample_interval=1)
def run_aggregate():
    aggregate.execute()

run_aggregate()
```

| Metric | Value |
|  ---  |  ---  |
|Run_ID| X |
|Timestamp| X |
|Sample_Timestamp| X |
|CPU_Usage_%| 59% |
|Time_Taken| 1057 seconds / 17 minutes |
|Median_Memory_Usage_MB| 360 MB |

#### 4.2.2 OpenEO exactextract

```python
from profiler import Profiler

import json
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

@Profiler(reruns=1, sample_interval=1)
def run_extract():
    exact_extract(data, polys, 'mean')

run_extract()
```

| Metric | Value |
|  ---  |  ---  |
|Run_ID| X |
|Timestamp| X |
|Sample_Timestamp| X |
|CPU_Usage_%| 63% |
|Time_Taken| 1085 seconds / 18 minutes |
|Median_Memory_Usage_MB| 360 MB |

### 5.0 Conclusion and Comments

So based on running the two comparisons, the efficiency is actually not that much
different. I am not exactly sure why yet, perhaps the OpenEO input cube causes it
to be slow even on exactextract.
