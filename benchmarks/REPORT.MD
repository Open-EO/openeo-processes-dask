# Aggregate Spatial Analysis

Author: Juraj Zvolensky @jzvolensky

## Table of Contents

1. [Introduction](#10-introduction)
2. [Methodology](#20-methodology)
3. [Datasets](#30-datasets)
4. [Initial Results](#40-initial-results)
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

We have 2 reference datasets. A small sample geojson containing three polygons and a datacube with a small spatial extent. The second dataset contains polygons for the Alto Adige region, as well as a larger datacube spatial extent to cover the area.

To measure the performance of the process, a small `Profiler` class has been developed. This class measures some key metrics of the process, to determine where the potential bottlenecks occur. Table below highlights all of the key metrics collected.

| Metric | Description |
|  ---  |  ---  |
|Run_ID| A unique identifier for each profiling run. |
|Timestamp| The timestamp when the profiling run started. |
|Sample_Timestamp| The timestamp for each memory and CPU sample taken during the profiling run. |
|Memory_sample_MB| The memory usage sample in megabytes taken at each sample interval. |
|CPU_Usage_%| The CPU usage percentage sample taken at each sample interval. |
|Time_Taken| The total time taken for the profiling run in seconds. |
|Median_Memory_Usage_MB| The median memory usage in megabytes over the entire profiling run. |
|Median_CPU_Usage_%| The median CPU usage percentage over the entire profiling run. |
|Disk_IO_Read_MB|The total disk I/O read in megabytes during the profiling run. |
|Disk_IO_Write_MB |The total disk I/O write in megabytes during the profiling run. |
|Network_IO_Sent_MB |The total network I/O sent in megabytes during the profiling run. |
|Network_IO_Received_MB |The total network I/O received in megabytes during the profiling run. |

We use the `Profiler` set for 10 runs in a row for a more accurate average result.

**NOTE**:  These metrics are mostly for show as Python is not very flexible in terms of performance measurements. It is difficult to measure the exact memory usage of a process in Python. The `Profiler` class is a simple class that uses the `psutil` library to measure the memory and CPU usage of the process. The `Profiler` class is not perfect and should be used as a rough estimate of the performance of the process.
  
## 3.0 Datasets

For the purpose of testing the performance we have obtained two datasets. The first is a very small dataset containing two polygons in the Bolzano area. The second dataset is a larger dataset containing the Alto Adige region.

The first dataset: `sample_polygons.geojson` has been provided by @clausmichele

- 2 polygons

The second dataset: `alto_adige.geojson` comes from the Openpolis github repository: `https://github.com/openpolis/geojson-italy`

- 116 polygons

## 4.0 Initial Results

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
