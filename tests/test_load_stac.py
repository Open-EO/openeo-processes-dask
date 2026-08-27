import json
import pathlib
import shutil
from importlib.metadata import version as _pkg_version

import numpy as np
import pystac
import pytest
import xarray as xr
from packaging.version import parse as parse_version

from openeo_processes_dask.process_implementations.cubes.load import load_stac, load_url
from openeo_processes_dask.process_implementations.exceptions import OpenEOException
from tests.mockdata import create_fake_rastercube

_STAC_TEMPLATE_PATH = pathlib.Path("tests/data/stac/s2_l2a_zarr_sample.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_zarr_v3_available: bool = parse_version(_pkg_version("zarr")) >= parse_version("3")


def _make_zarr_stac_item(
    zarr_path: pathlib.Path,
    *,
    open_kwargs: dict | None = None,
    storage_options: dict | None = None,
) -> pathlib.Path:
    """Create a temporary STAC Item JSON pointing to *zarr_path*.

    If *open_kwargs* is ``None`` the ``xarray:open_kwargs`` field is
    removed from the asset.  Same for *storage_options*.
    """
    import json

    with open(str(_STAC_TEMPLATE_PATH)) as f:
        item_dict = json.load(f)

    item_dict["assets"]["data"]["href"] = str(zarr_path)

    if open_kwargs is not None:
        item_dict["assets"]["data"]["xarray:open_kwargs"] = open_kwargs
    else:
        item_dict["assets"]["data"].pop("xarray:open_kwargs", None)

    if storage_options is not None:
        item_dict["assets"]["data"]["xarray:storage_options"] = storage_options
    else:
        item_dict["assets"]["data"].pop("xarray:storage_options", None)

    stac_path = zarr_path.parent / f"{zarr_path.name}_stac.json"
    with open(str(stac_path), "w") as f:
        json.dump(item_dict, f, indent=2)
    return stac_path


def _create_cube_for_zarr(bounding_box, random_raster_data, temporal_interval):
    return create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08", "SCL"],
        backend="numpy",
    )


@pytest.mark.parametrize("size", [(10, 10, 10, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_load_stac(bounding_box, random_raster_data, temporal_interval):
    url = "./tests/data/stac/s2_l2a_test_item.json"
    output_cube = load_stac(
        url=url,
        spatial_extent=bounding_box,
        bands=["red"],
    )

    assert output_cube.openeo is not None
    assert len(output_cube[output_cube.openeo.x_dim]) > 0
    assert len(output_cube[output_cube.openeo.y_dim]) > 0
    assert len(output_cube[output_cube.openeo.band_dims[0]]) > 0
    assert len(output_cube[output_cube.openeo.temporal_dims[0]]) > 0

    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08", "SCL"],
        backend="dask",
    )

    input_cube.to_dataset(dim="bands").to_zarr("./tests/data/s2_l2a_zarr_sample.zarr")

    url = "./tests/data/stac/s2_l2a_zarr_sample.json"
    output_cube = load_stac(
        url=url,
        bands=["B04"],
    )

    assert output_cube.openeo is not None
    assert len(output_cube[output_cube.openeo.x_dim]) > 0
    assert len(output_cube[output_cube.openeo.y_dim]) > 0
    assert len(output_cube[output_cube.openeo.band_dims[0]]) > 0
    assert len(output_cube[output_cube.openeo.temporal_dims[0]]) > 0
    shutil.rmtree("./tests/data/s2_l2a_zarr_sample.zarr")


# ---------------------------------------------------------------------------
# Zarr format compatibility tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(10, 10, 10, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("zarr_version", [2, 3])
def test_load_stac_zarr_with_open_kwargs(
    bounding_box, random_raster_data, temporal_interval, tmp_path, zarr_version
):
    if zarr_version == 3 and not _zarr_v3_available:
        pytest.skip("Zarr v3 is not supported in this configuration")

    cube = _create_cube_for_zarr(bounding_box, random_raster_data, temporal_interval)
    zarr_path = tmp_path / f"v{zarr_version}_with_kwargs.zarr"
    cube.to_dataset(dim="bands").to_zarr(str(zarr_path), zarr_format=zarr_version)

    open_kwargs = {"engine": "zarr", "chunks": {}, "zarr_format": zarr_version}
    stac_path = _make_zarr_stac_item(zarr_path, open_kwargs=open_kwargs)
    output_cube = load_stac(url=str(stac_path), bands=["B04"])

    assert output_cube.openeo is not None
    assert "B04" in output_cube[output_cube.openeo.band_dims[0]].values


@pytest.mark.parametrize("size", [(10, 10, 10, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("consolidated", [True, False])
def test_load_stac_zarr_v2_without_open_kwargs(
    bounding_box, random_raster_data, temporal_interval, tmp_path, consolidated
):
    cube = _create_cube_for_zarr(bounding_box, random_raster_data, temporal_interval)
    consolidated_label = "consolidated" if consolidated else "unconsolidated"
    zarr_path = tmp_path / f"v2_{consolidated_label}.zarr"
    cube.to_dataset(dim="bands").to_zarr(
        str(zarr_path), zarr_format=2, consolidated=consolidated
    )

    stac_path = _make_zarr_stac_item(zarr_path, open_kwargs=None)
    output_cube = load_stac(url=str(stac_path), bands=["B04"])

    assert output_cube.openeo is not None
    assert "B04" in output_cube[output_cube.openeo.band_dims[0]].values


def test_load_stac_zarr_missing_requested_band(bounding_box, tmp_path):
    zarr_path = tmp_path / "missing_band.zarr"
    ds = xr.Dataset(
        {"B04": (("t", "y", "x"), np.ones((1, 10, 10), dtype=np.float32))},
        coords={
            "t": [np.datetime64("2022-06-02")],
            "y": np.arange(10),
            "x": np.arange(10),
        },
    )
    ds.to_zarr(str(zarr_path), zarr_format=2)

    stac_path = _make_zarr_stac_item(zarr_path, open_kwargs=None)
    with pytest.raises(OpenEOException, match="missing_band"):
        load_stac(url=str(stac_path), bands=["missing_band"])


def test_load_url(tmp_path):
    import multiprocessing as _mp

    import geopandas as gpd
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    import shapely as _shapely
    from shapely.geometry import Point, Polygon

    # -- GeoJSON test --
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "test"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [10.0, 46.0],
                            [11.0, 46.0],
                            [11.0, 47.0],
                            [10.0, 47.0],
                            [10.0, 46.0],
                        ]
                    ],
                },
            }
        ],
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }
    geojson_path = tmp_path / "test.geojson"
    geojson_path.write_text(json.dumps(geojson_data))

    load_vector = load_url(url=str(geojson_path), format="GeoJSON")
    assert isinstance(load_vector, xr.Dataset)

    # -- GeoParquet test --
    gdf = gpd.GeoDataFrame(
        {"a": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    field_name = gdf.geometry.name
    df = gdf.copy()
    with _mp.Pool() as pool:
        df[field_name] = pool.map(_shapely.wkb.dumps, df[field_name])
    table = _pa.Table.from_pandas(df)
    geo_metadata = {
        "geometry_fields": [
            {
                "field_name": field_name,
                "geometry_format": "wkb",
                "geometry_types": ["Point"],
                "crs": "EPSG:4326",
                "crs_format": "unknown",
            }
        ]
    }
    tbl_metadata = table.schema.metadata
    tbl_metadata[b"geometry_fields"] = json.dumps(
        geo_metadata["geometry_fields"]
    ).encode("utf-8")
    table = table.replace_schema_metadata(tbl_metadata)
    geoparquet_path = tmp_path / "test.geoparquet"
    _pq.write_table(table, str(geoparquet_path))

    load_vector = load_url(url=str(geoparquet_path), format="Parquet")
    assert isinstance(load_vector, xr.Dataset)
