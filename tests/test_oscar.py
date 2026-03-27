import json
import tempfile
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
import requests
import yaml

from openeo_processes_dask.process_implementations.exceptions import (
    EGIAuthError,
    OpenEOAuthError,
    OscarNotAvailable,
    OscarServiceCreationError,
    OscarServiceError,
    OscarServiceNotFound,
    OscarUrlError,
)
from openeo_processes_dask.process_implementations.experimental.oscar import (
    OscarRunner,
    run_oscar,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_oscar_client():
    """Mock OSCAR client with common behaviors"""
    client = Mock()
    client.ssl = True
    client.get_cluster_info.return_value = {"status": "ok"}
    client.get_service.return_value = {"name": "test-service"}
    client.create_service.return_value = Mock(status_code=201, text="Service created")
    client.update_service.return_value = Mock(status_code=200)
    client.list_jobs.return_value = Mock(
        text=json.dumps({"job-123": {"creation_time": "2025-01-01T10:00:00Z"}})
    )
    client.get_job_logs.return_value = Mock(
        text="Log output\nSTAC OUTPUT URL: https://example.com/stac/collection.json\n"
    )
    return client


@pytest.fixture
def sample_service_config():
    """Valid OSCAR service YAML configuration"""
    return {
        "functions": {
            "oscar": [
                {
                    "cluster_id": {
                        "name": "test-service",
                        "environment": {
                            "variables": {"EXISTING_VAR": "existing_value"}
                        },
                    }
                }
            ]
        }
    }


@pytest.fixture
def sample_service_config_yaml(tmp_path, sample_service_config):
    """Write sample config to a temporary YAML file"""
    config_file = tmp_path / "service_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(sample_service_config, f)
    return str(config_file)


@pytest.fixture
def sample_logs_with_stac():
    """Sample job logs containing STAC URL"""
    return """
Processing started...
Loading data...
Processing complete.
STAC OUTPUT URL: https://s3.example.com/bucket/output/collection.json
Job finished successfully.
"""


@pytest.fixture
def sample_logs_without_stac():
    """Sample job logs without STAC URL"""
    return """
Processing started...
Loading data...
Job finished successfully.
"""


@pytest.fixture
def mock_egi_token_response():
    """Mock successful EGI token refresh response"""
    return {
        "access_token": "mock-access-token-12345",
        "token_type": "Bearer",
        "expires_in": 3600,
    }


@pytest.fixture
def mock_refresh_token_file(tmp_path):
    """Create a temporary refresh token file"""
    token_data = {
        "https://aai.egi.eu/auth/realms/egi": {
            "openeo-platform-default-client": {
                "refresh_token": "mock-refresh-token-xyz"
            }
        }
    }
    token_file = tmp_path / "refresh_token.json"
    with open(token_file, "w") as f:
        json.dump(token_data, f)
    return str(token_file)


# ============================================================================
# Authentication Tests
# ============================================================================


class TestAuthentication:
    """Test authentication methods"""

    def test_oscar_authenticate_with_env_var(self, monkeypatch):
        """Test authentication using environment variable"""
        monkeypatch.setenv("OPENEO_TOKEN", "test-token-from-env")

        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
            auth_token="OPENEO_TOKEN",
        )

        token = runner.oscar_authenticate(token_env_var="OPENEO_TOKEN")
        assert token == "test-token-from-env"

    def test_oscar_authenticate_with_local_refresh_token(
        self, mock_refresh_token_file, mock_egi_token_response
    ):
        """Test authentication using local refresh token file"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
            local_refresh_token=mock_refresh_token_file,
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_egi_token_response
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            token = runner.oscar_authenticate(
                local_refresh_token=mock_refresh_token_file
            )

            assert token == "mock-access-token-12345"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["data"]["refresh_token"] == "mock-refresh-token-xyz"

    def test_oscar_authenticate_missing_credentials(self):
        """Test authentication fails when no credentials provided"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        with pytest.raises(OpenEOAuthError, match="No OpenEO access token found"):
            runner.oscar_authenticate()

    def test_oscar_authenticate_egi_request_failure(self, mock_refresh_token_file):
        """Test authentication fails when EGI request fails"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
            local_refresh_token=mock_refresh_token_file,
        )

        with patch(
            "requests.post",
            side_effect=requests.exceptions.RequestException("Network error"),
        ):
            with pytest.raises(EGIAuthError, match="Failed to retrieve access token"):
                runner.oscar_authenticate(local_refresh_token=mock_refresh_token_file)


# ============================================================================
# Service Management Tests
# ============================================================================


class TestServiceManagement:
    """Test OSCAR service management methods"""

    def test_oscar_check_connection_success(self, mock_oscar_client):
        """Test successful connection to OSCAR endpoint"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        with patch(
            "openeo_processes_dask.process_implementations.experimental.oscar.Client"
        ) as mock_client_class:
            mock_client_class.return_value = mock_oscar_client

            client = runner.oscar_check_connection(
                "https://oscar.example.com", "test-token"
            )

            assert client == mock_oscar_client
            mock_oscar_client.get_cluster_info.assert_called_once()

    def test_oscar_check_connection_not_available(self):
        """Test connection failure when OSCAR is not available"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        with patch(
            "openeo_processes_dask.process_implementations.experimental.oscar.Client"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.get_cluster_info.side_effect = Exception("Connection refused")
            mock_client_class.return_value = mock_client

            with pytest.raises(OscarNotAvailable, match="OSCAR is not available"):
                runner.oscar_check_connection("https://oscar.example.com", "test-token")

    def test_oscar_check_connection_invalid_url(self):
        """Test connection with invalid OSCAR URL"""
        runner = OscarRunner(
            oscar_endpoint="",
            service="test-service",
            output="output-path",
        )

        with pytest.raises(OscarUrlError, match="OSCAR endpoint is not provided"):
            runner.oscar_check_connection("", "test-token")

    def test_oscar_check_service_exists(self, mock_oscar_client):
        """Test checking for existing service"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )
        runner.oscar_client = mock_oscar_client

        service_info = runner.oscar_check_service(mock_oscar_client, "test-service")

        assert service_info == {"name": "test-service"}
        mock_oscar_client.get_service.assert_called_once_with("test-service")

    def test_oscar_check_service_not_found(self):
        """Test checking for non-existent service"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        mock_client = Mock()
        mock_client.get_service.return_value = None

        service_info = runner.oscar_check_service(mock_client, "non-existent-service")

        assert service_info is None

    def test_oscar_create_service_success(
        self, mock_oscar_client, sample_service_config_yaml
    ):
        """Test successful service creation"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
            service_config=sample_service_config_yaml,
        )

        response = runner.oscar_create_service(
            mock_oscar_client, "test-service", sample_service_config_yaml
        )

        assert response.status_code == 201
        mock_oscar_client.create_service.assert_called_once()

    def test_oscar_create_service_failure(self, sample_service_config_yaml):
        """Test service creation failure"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
            service_config=sample_service_config_yaml,
        )

        mock_client = Mock()
        mock_response = Mock(status_code=500, text="Internal Server Error")
        mock_client.create_service.return_value = mock_response

        with pytest.raises(
            OscarServiceCreationError, match="Failed to create OSCAR service"
        ):
            runner.oscar_create_service(
                mock_client, "test-service", sample_service_config_yaml
            )

    def test_oscar_create_service_no_config(self, mock_oscar_client):
        """Test service creation when no config provided"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        response = runner.oscar_create_service(mock_oscar_client, "test-service", None)

        assert response is None
        mock_oscar_client.create_service.assert_not_called()


# ============================================================================
# Service Configuration Tests
# ============================================================================


class TestServiceConfiguration:
    """Test OSCAR service configuration updates"""

    def test_oscar_update_service_config_with_context(
        self, mock_oscar_client, sample_service_config_yaml
    ):
        """Test updating service config with context variables"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
            context={"VAR1": "value1", "VAR2": "value2"},
        )

        runner.oscar_update_service_config(
            mock_oscar_client,
            "test-service",
            sample_service_config_yaml,
            input_stac_url="https://example.com/stac/input.json",
        )

        mock_oscar_client.update_service.assert_called_once()
        # Verify the service name was passed correctly
        call_args = mock_oscar_client.update_service.call_args
        assert call_args[0][0] == "test-service"

    def test_oscar_update_service_config_from_url(
        self, mock_oscar_client, sample_service_config
    ):
        """Test updating service config from HTTP URL"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
            context={"NEW_VAR": "new_value"},
        )

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.text = yaml.safe_dump(sample_service_config)
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            runner.oscar_update_service_config(
                mock_oscar_client,
                "test-service",
                "https://example.com/config.yaml",
                input_stac_url="https://example.com/stac/input.json",
            )

            mock_get.assert_called_once_with("https://example.com/config.yaml")
            mock_oscar_client.update_service.assert_called_once()

    def test_oscar_update_service_config_no_config(self, mock_oscar_client):
        """Test update when no config provided"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        runner.oscar_update_service_config(
            mock_oscar_client,
            "test-service",
            None,
            input_stac_url="https://example.com/stac/input.json",
        )

        mock_oscar_client.update_service.assert_not_called()


# ============================================================================
# Job Execution Tests
# ============================================================================


class TestJobExecution:
    """Test OSCAR job execution and monitoring"""

    def test_get_job_logs_success(self, mock_oscar_client, sample_logs_with_stac):
        """Test successful retrieval of job logs"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        mock_oscar_client.get_job_logs.return_value = Mock(text=sample_logs_with_stac)

        with patch("time.sleep"):  # Speed up test
            logs = runner.get_job_logs(
                mock_oscar_client, "test-service", poll_interval=1, timeout=10
            )

        assert "STAC OUTPUT URL" in logs
        assert logs == sample_logs_with_stac

    def test_get_job_logs_timeout(self):
        """Test job logs retrieval timeout"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        mock_client = Mock()
        mock_client.list_jobs.return_value = Mock(
            text=json.dumps({"job-123": {"creation_time": "2025-01-01T10:00:00Z"}})
        )
        mock_client.get_job_logs.return_value = Mock(text="")

        with patch("time.sleep"):  # Speed up test
            with pytest.raises(OscarServiceError, match="Timeout"):
                runner.get_job_logs(
                    mock_client, "test-service", poll_interval=1, timeout=2
                )

    def test_get_job_logs_no_jobs(self):
        """Test when no jobs are found for service"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        mock_client = Mock()
        mock_client.list_jobs.return_value = Mock(text=json.dumps({}))

        with pytest.raises(OscarServiceError, match="No jobs found"):
            runner.get_job_logs(mock_client, "test-service")

    def test_parse_logs_stac_success(self, sample_logs_with_stac):
        """Test parsing STAC URL from logs"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        stac_url = runner.parse_logs_stac(sample_logs_with_stac)

        assert stac_url == "https://s3.example.com/bucket/output/collection.json"

    def test_parse_logs_stac_not_found(self, sample_logs_without_stac):
        """Test parsing logs without STAC URL"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        stac_url = runner.parse_logs_stac(sample_logs_without_stac)

        assert stac_url is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestOscarRunnerIntegration:
    """Test complete OscarRunner workflows"""

    def test_run_oscar_service_complete_flow(
        self,
        monkeypatch,
        mock_oscar_client,
        sample_service_config_yaml,
        sample_logs_with_stac,
    ):
        """Test complete workflow from authentication to result"""
        monkeypatch.setenv("OPENEO_TOKEN", "test-token")

        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
            auth_token="OPENEO_TOKEN",
            service_config=sample_service_config_yaml,
            input_stac_url="https://example.com/input.json",
            context={"CONTEXT_VAR": "context_value"},
        )

        with patch(
            "openeo_processes_dask.process_implementations.experimental.oscar.Client"
        ) as mock_client_class:
            mock_client_class.return_value = mock_oscar_client
            mock_oscar_client.get_job_logs.return_value = Mock(
                text=sample_logs_with_stac
            )

            with patch("requests.post") as mock_post:
                mock_response = Mock(status_code=200)
                mock_post.return_value = mock_response

                with patch("time.sleep"):  # Speed up test
                    result = runner.run_oscar_service()

        assert result == "https://s3.example.com/bucket/output/collection.json"

    def test_run_oscar_service_creates_missing_service(
        self, monkeypatch, sample_service_config_yaml, sample_logs_with_stac
    ):
        """Test workflow when service doesn't exist and needs to be created"""
        monkeypatch.setenv("OPENEO_TOKEN", "test-token")

        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="new-service",
            output="output-path",
            auth_token="OPENEO_TOKEN",
            service_config=sample_service_config_yaml,
        )

        mock_client = Mock()
        mock_client.ssl = True
        mock_client.get_cluster_info.return_value = {"status": "ok"}
        mock_client.get_service.side_effect = [
            None,
            {"name": "new-service"},
        ]  # First None, then exists
        mock_client.create_service.return_value = Mock(status_code=201)
        mock_client.list_jobs.return_value = Mock(
            text=json.dumps({"job-456": {"creation_time": "2025-01-01T11:00:00Z"}})
        )
        mock_client.get_job_logs.return_value = Mock(text=sample_logs_with_stac)

        with patch(
            "openeo_processes_dask.process_implementations.experimental.oscar.Client",
            return_value=mock_client,
        ):
            with patch("requests.post", return_value=Mock(status_code=200)):
                with patch("time.sleep"):
                    result = runner.run_oscar_service()

        mock_client.create_service.assert_called_once()
        assert result == "https://s3.example.com/bucket/output/collection.json"


# ============================================================================
# Public API Tests
# ============================================================================


class TestRunOscarPublicAPI:
    """Test the public run_oscar function"""

    def test_run_oscar_function(self, monkeypatch, sample_logs_with_stac):
        """Test the run_oscar public function"""
        monkeypatch.setenv("OPENEO_TOKEN", "test-token")

        mock_client = Mock()
        mock_client.ssl = True
        mock_client.get_cluster_info.return_value = {"status": "ok"}
        mock_client.get_service.return_value = {"name": "test-service"}
        mock_client.list_jobs.return_value = Mock(
            text=json.dumps({"job-789": {"creation_time": "2025-01-01T12:00:00Z"}})
        )
        mock_client.get_job_logs.return_value = Mock(text=sample_logs_with_stac)

        with patch(
            "openeo_processes_dask.process_implementations.experimental.oscar.Client",
            return_value=mock_client,
        ):
            with patch("requests.post", return_value=Mock(status_code=200)):
                with patch("time.sleep"):
                    result = run_oscar(
                        oscar_endpoint="https://oscar.example.com",
                        service="test-service",
                        output="output-path",
                        auth_token="OPENEO_TOKEN",
                    )

        assert result == "https://s3.example.com/bucket/output/collection.json"


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Test input validation methods"""

    def test_validate_input_valid_stac_url(self):
        """Test validation with valid STAC URL"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        mock_item = Mock()
        with patch("pystac.Item.from_file", return_value=mock_item):
            runner.validate_input("https://example.com/item.json")  # Should not raise

    def test_validate_input_invalid_stac_url(self):
        """Test validation with invalid STAC URL"""
        runner = OscarRunner(
            oscar_endpoint="https://oscar.example.com",
            service="test-service",
            output="output-path",
        )

        with patch("pystac.Item.from_file", side_effect=Exception("Invalid STAC")):
            with pytest.raises(ValueError, match="Invalid input STAC URL"):
                runner.validate_input("https://example.com/invalid.json")
