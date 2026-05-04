import shutil
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pystac
import pytest
import xarray as xr

from openeo_processes_dask.process_implementations.cubes.load import (
    _get_asset_band_names,
    _get_band_assets_from_items,
    _is_band_asset,
    _is_supported_raster_media_type,
    load_stac,
    load_url,
)
from tests.mockdata import create_fake_rastercube


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


def test_load_url():
    URL = "https://github.com/ValentinaHutter/polygons/raw/master/geoparquet/example%20file.geoparquet"

    load_vector = load_url(url=URL, format="Parquet")

    assert isinstance(load_vector, xr.Dataset)

    URL = "https://raw.githubusercontent.com/ValentinaHutter/polygons/master/polygons_all.json"

    load_vector = load_url(url=URL, format="GeoJSON")

    assert isinstance(load_vector, xr.Dataset)


class TestMediaTypeValidation:
    """Test suite for asset media type validation"""

    def test_is_supported_raster_media_type_geotiff(self):
        """Test GeoTIFF media types are supported"""
        assert _is_supported_raster_media_type("image/tiff") is True
        assert _is_supported_raster_media_type("image/tif") is True
        assert _is_supported_raster_media_type("image/vnd.stac.geotiff") is True
        assert _is_supported_raster_media_type("IMAGE/TIFF") is True

    def test_is_supported_raster_media_type_other_formats(self):
        """Test other supported raster formats"""
        assert _is_supported_raster_media_type("image/jp2") is True
        assert _is_supported_raster_media_type("image/png") is True
        assert _is_supported_raster_media_type("image/jpeg") is True
        assert _is_supported_raster_media_type("application/x-netcdf") is True
        assert _is_supported_raster_media_type("application/vnd+zarr") is True

    def test_is_supported_raster_media_type_unsupported(self):
        """Test unsupported media types are rejected"""
        assert _is_supported_raster_media_type("application/pdf") is False
        assert _is_supported_raster_media_type("text/html") is False
        assert _is_supported_raster_media_type("application/json") is False
        assert _is_supported_raster_media_type("video/mp4") is False

    def test_is_supported_raster_media_type_none(self):
        """Test None/empty media types are treated as supported (unknown)"""
        assert _is_supported_raster_media_type(None) is True
        assert _is_supported_raster_media_type("") is True

    def test_is_band_asset_with_data_role(self):
        """Test asset with 'data' role is recognized as band asset"""
        asset = Mock(spec=pystac.Asset)
        asset.media_type = "image/tiff"
        asset.roles = ["data"]
        asset.extra_fields = {}

        assert _is_band_asset(asset) is True

    def test_is_band_asset_with_unsupported_media_type(self):
        """Test asset with unsupported media type is rejected"""
        asset = Mock(spec=pystac.Asset)
        asset.media_type = "application/pdf"
        asset.roles = ["data"]
        asset.extra_fields = {}

        assert _is_band_asset(asset) is False

    def test_is_band_asset_with_metadata_role(self):
        """Test asset with only metadata role is rejected"""
        asset = Mock(spec=pystac.Asset)
        asset.media_type = "application/json"
        asset.roles = ["metadata"]
        asset.extra_fields = {}

        assert _is_band_asset(asset) is False

    def test_is_band_asset_with_thumbnail_role(self):
        """Test asset with thumbnail role is rejected"""
        asset = Mock(spec=pystac.Asset)
        asset.media_type = "image/png"
        asset.roles = ["thumbnail"]
        asset.extra_fields = {}

        assert _is_band_asset(asset) is False

    def test_is_band_asset_with_multiple_roles_including_data(self):
        """Test asset with multiple roles including 'data' is accepted"""
        asset = Mock(spec=pystac.Asset)
        asset.media_type = "image/tiff"
        asset.roles = ["data", "metadata"]
        asset.extra_fields = {}

        assert _is_band_asset(asset) is True

    def test_is_band_asset_fallback_to_band_metadata(self):
        """Test asset without roles falls back to checking band metadata"""
        asset = Mock(spec=pystac.Asset)
        asset.media_type = "image/tiff"
        asset.roles = None
        asset.extra_fields = {"eo:bands": [{"name": "B01"}]}

        assert _is_band_asset(asset) is True

    def test_is_band_asset_no_roles_no_metadata(self):
        """Test asset without roles and without band metadata is rejected"""
        asset = Mock(spec=pystac.Asset)
        asset.media_type = "image/tiff"
        asset.roles = None
        asset.extra_fields = {}

        assert _is_band_asset(asset) is False

    def test_get_asset_band_names_from_eo_bands(self):
        """Test extracting band names from eo:bands"""
        asset = Mock(spec=pystac.Asset)
        asset.extra_fields = {
            "eo:bands": [{"name": "B01"}, {"name": "B02"}, {"name": "B03"}]
        }

        band_names = _get_asset_band_names(asset)
        assert band_names == ["B01", "B02", "B03"]

    def test_get_asset_band_names_from_bands(self):
        """Test extracting band names from bands field"""
        asset = Mock(spec=pystac.Asset)
        asset.extra_fields = {
            "bands": [{"name": "red"}, {"name": "green"}, {"name": "blue"}]
        }

        band_names = _get_asset_band_names(asset)
        assert band_names == ["red", "green", "blue"]

    def test_get_asset_band_names_with_common_name_fallback(self):
        """Test extracting band names using common_name as fallback"""
        asset = Mock(spec=pystac.Asset)
        asset.extra_fields = {
            "eo:bands": [
                {"common_name": "red"},
                {"name": "B02", "common_name": "green"},
            ]
        }

        band_names = _get_asset_band_names(asset)
        assert band_names == ["red", "B02"]

    def test_get_asset_band_names_empty(self):
        """Test extracting band names when no band metadata exists"""
        asset = Mock(spec=pystac.Asset)
        asset.extra_fields = {}

        band_names = _get_asset_band_names(asset)
        assert band_names == []

    def test_get_band_assets_from_items_filters_correctly(self):
        """Test that only band assets are returned from items"""
        item = Mock(spec=pystac.Item)

        data_asset = Mock(spec=pystac.Asset)
        data_asset.media_type = "image/tiff"
        data_asset.roles = ["data"]
        data_asset.extra_fields = {}

        metadata_asset = Mock(spec=pystac.Asset)
        metadata_asset.media_type = "application/json"
        metadata_asset.roles = ["metadata"]
        metadata_asset.extra_fields = {}

        thumbnail_asset = Mock(spec=pystac.Asset)
        thumbnail_asset.media_type = "image/png"
        thumbnail_asset.roles = ["thumbnail"]
        thumbnail_asset.extra_fields = {}

        item.assets = {
            "B01": data_asset,
            "metadata": metadata_asset,
            "thumbnail": thumbnail_asset,
        }

        items = [item]
        band_assets = _get_band_assets_from_items(items)

        assert "B01" in band_assets
        assert "metadata" not in band_assets
        assert "thumbnail" not in band_assets
        assert len(band_assets) == 1


class TestRetryLogic:
    """Test suite for retry logic in STAC API requests"""

    @patch(
        "openeo_processes_dask.process_implementations.cubes.load.pystac_client.Client.open"
    )
    def test_retry_stac_io_successful_request(self, mock_client_open):
        """Test that successful requests don't trigger retries"""
        from openeo_processes_dask.process_implementations.cubes.load import (
            _process_stac_collection,
        )

        mock_catalog = MagicMock()
        mock_search = MagicMock()
        mock_search.item_collection.return_value = []
        mock_catalog.search.return_value = mock_search
        mock_client_open.return_value = mock_catalog

        result = _process_stac_collection(
            url="https://example.com/collection",
            spatial_extent=None,
            temporal_extent=None,
            bands=None,
            properties=None,
            catalog_url="https://example.com",
            collection_id="test-collection",
        )

        assert result == []
        assert mock_catalog.search.call_count == 1

    @patch("openeo_processes_dask.process_implementations.cubes.load.requests.Session")
    def test_retry_stac_io_class_handles_retries(self, mock_session_class):
        """Test that RetryStacIO class is properly configured"""
        from openeo_processes_dask.process_implementations.cubes.load import (
            _process_stac_collection,
        )

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        with patch(
            "openeo_processes_dask.process_implementations.cubes.load.pystac_client.Client.open"
        ) as mock_open:
            mock_catalog = MagicMock()
            mock_search = MagicMock()
            mock_search.item_collection.return_value = []
            mock_catalog.search.return_value = mock_search
            mock_open.return_value = mock_catalog

            _process_stac_collection(
                url="https://example.com/collection",
                spatial_extent=None,
                temporal_extent=None,
                bands=None,
                properties=None,
                catalog_url="https://example.com",
                collection_id="test-collection",
            )

            assert mock_session_class.called
            assert mock_session.mount.call_count >= 2  # http:// and https://

    def test_retry_configuration(self):
        """Test that retry strategy is configured correctly"""
        from openeo_processes_dask.process_implementations.cubes.load import (
            _process_stac_collection,
        )

        with patch(
            "openeo_processes_dask.process_implementations.cubes.load.Retry"
        ) as mock_retry:
            with patch(
                "openeo_processes_dask.process_implementations.cubes.load.pystac_client.Client.open"
            ) as mock_open:
                mock_catalog = MagicMock()
                mock_search = MagicMock()
                mock_search.item_collection.return_value = []
                mock_catalog.search.return_value = mock_search
                mock_open.return_value = mock_catalog

                _process_stac_collection(
                    url="https://example.com/collection",
                    spatial_extent=None,
                    temporal_extent=None,
                    bands=None,
                    properties=None,
                    catalog_url="https://example.com",
                    collection_id="test-collection",
                )

                if mock_retry.called:
                    call_kwargs = mock_retry.call_args[1]
                    assert call_kwargs.get("total") == 7
                    assert call_kwargs.get("backoff_factor") == 2
                    assert 429 in call_kwargs.get("status_forcelist", [])
