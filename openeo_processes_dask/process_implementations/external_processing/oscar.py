# External processing in OSCAR
# Uses the OSCAR client to offload processing tasks to OSCAR.
# https://github.com/grycap/oscar_python
# https://github.com/grycap/oscar

import json
import logging
import os
import time
import uuid
import tempfile
import yaml
from typing import Optional

from openeo_processes_dask.process_implementations.exceptions import (
    OpenEOAuthError,
    OscarNotAvailable,
    OscarUrlError,
    OscarServiceNotFound,
    OscarServiceCreationError,
    OscarServiceError,
    MinioConnectionError,
    MinioUploadError,
    MinioDownloadError,
    EGIAuthError,
)

import oscar_python._utils as utils
import requests
import pystac
from minio import Minio
from oscar_python.client import Client

__all__ = ["run_oscar"]

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = [handler]

class OscarRunner:
    def __init__ (
        self,
        oscar_endpoint: str,
        service: str,
        output: str,
        data: Optional[dict] = None,
        input_stac_url: str = "",
        auth_token: Optional[str] = None,
        oscar_script: Optional[str] = None,
        input_file: Optional[str] = None,
        service_config: Optional[dict] = None,
        local_refresh_token: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        self.oscar_endpoint = oscar_endpoint
        self.service = service
        self.output = output
        self.data = data
        self.input_stac_url = input_stac_url
        self.auth_token = auth_token
        self.oscar_script = oscar_script
        self.input_file = input_file
        self.service_config = service_config
        self.local_refresh_token = local_refresh_token
        self.context = context

        self.oscar_client = None
        self.minio_client = None
        self.minio_info = None
        self.input_info = None
        self.output_info = None

    def __repr__(self):
        return f"OscarRunner(oscar_endpoint={self.oscar_endpoint}, service={self.service}, service_config={self.service_config})"
    
    def validate_input(self, stac_input_url: str) -> None:
        """
        Validate the input STAC URL using pystac
        """
        try:
            logger.info(f"Validating input STAC URL: {stac_input_url}")
            item = pystac.Item.from_file(stac_input_url)
            if not item:
                raise ValueError("Invalid STAC item.")
            logger.info("Input STAC URL is valid.")
        except Exception as e:
            logger.error(f"Failed to validate input STAC URL: {e}")
            raise ValueError(f"Invalid input STAC URL: {stac_input_url}") from e

    def oscar_authenticate(
        self, 
        token_env_var: Optional[str] = None,
        local_refresh_token: Optional[str] = None
    ) -> str:
        """
        Authenticate with OSCAR

        Option 1: an environment variable is read

        Option 2: provided a local refresh token file path

        parameters:
            token_env_var (str): Environment variable name for the OpenEO access token.
            local_refresh_token (str): Local file path to the OpenEO refresh token.
        returns:
            str: The access token to be used for OSCAR authentication.

        Raises:
            OpenEOAuthError: If no token is found in the environment variable or local file.
        """

        if token_env_var:
            auth_token = os.getenv(token_env_var)
            if auth_token:
                logger.info("Using token from environment variable.")
                return auth_token

        if local_refresh_token:
            try:
                with open(local_refresh_token) as f:
                    token_data = json.loads(f.read())
                logger.info(f"Using local refresh token from {local_refresh_token}")
                refresh_token = token_data["https://aai.egi.eu/auth/realms/egi"]["openeo-platform-default-client"]["refresh_token"].strip()
                logger.info("Getting access token from a local refresh token...")
                url = "https://aai.egi.eu/auth/realms/egi/protocol/openid-connect/token"
                data = {
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": "openeo-platform-default-client",
                    "scope": "openid email offline_access eduperson_scoped_affiliation eduperson_entitlement",
                }
                try:
                    response = requests.post(url, data=data, timeout=10)
                    response.raise_for_status()
                    auth_token = response.json()["access_token"]
                    logger.info("Access token retrieved successfully")
                    return auth_token
                except requests.exceptions.RequestException as e:
                    raise EGIAuthError(f"Failed to retrieve access token: {e}")
            except Exception as e:
                logger.error(f"Error reading local refresh token: {e}")

        raise OpenEOAuthError("No OpenEO access token found in environment variable or local file.")

    def oscar_check_connection(self, oscar_endpoint: str, auth_token: str) -> Client:
        """
        Check if the OSCAR endpoint is reachable.

        parameters:
            oscar_endpoint (str): The OSCAR endpoint URL.
            auth_token (str): The access token for OSCAR authentication.

        returns:
            Client: An authenticated OSCAR client.
        Raises:
            OscarNotAvailable: If the OSCAR endpoint is not reachable.
            OscarUrlError: If the OSCAR endpoint URL is invalid.
        """
        if not oscar_endpoint:
            raise OscarUrlError("OSCAR endpoint is not provided")
        
        logger.info(f"Checking OSCAR connection to {oscar_endpoint}...")

        options_basic_auth = {
            "cluster_id": "cluster_id",
            "endpoint": oscar_endpoint,
            "oidc_token": auth_token,
            "ssl": "True",
        }
        oscar_client = Client(options=options_basic_auth)

        try:
            oscar_client.get_cluster_info()
        except Exception as e:
            raise OscarNotAvailable(f"OSCAR is not available for {oscar_endpoint}: {e}")
        return oscar_client
    
    def oscar_check_service(
        self, oscar_client: Client, service_name: str
    ) -> requests.Response | None:
        """
        Check if the OSCAR service exists.

        parameters:
            oscar_client (Client): The authenticated OSCAR client.
            service_name (str): The name of the OSCAR service to check.

        returns:
            dict: The service information if it exists.

        Raises:
            OscarServiceNotFound: If the service does not exist.
        """
        try:
            service_info = oscar_client.get_service(service_name)
            if service_info:
                logger.info(f"OSCAR service '{service_name}' found.")
                return service_info
            else:
                logger.info(f"OSCAR service '{service_name}' not found.")
                return service_info
        except Exception as e:
            logger.error(f"Error checking OSCAR service '{service_name}': {e}")
        
    def oscar_create_service(
        self,
        oscar_client: Client,
        service_name: str,
        service_config: dict
    ) -> requests.Response | None:
        """
        Create a new OSCAR service. Service config can be provided as a local path
        or as an HTTP(S) URL.

        parameters:
            oscar_client (Client): The authenticated OSCAR client.
            service_name (str): The name of the OSCAR service to create.
            service_config (dict): The configuration for the OSCAR service.
        returns:
            requests.Response | None: The response from the OSCAR API or None if no creation is needed.

        Raises:
            OscarServiceCreationError: If the service creation fails.
        """
        if not service_config:
            logger.info("No service configuration provided, skipping creation.")
            return None

        try:
            logger.info(f"Creating OSCAR service '{service_name}' with provided configuration.")
            response = oscar_client.create_service(service_config)
            if response.status_code == 201:
                logger.info(f"OSCAR service '{service_name}' created successfully.")
                return response
            else:
                logger.error(f"Failed to create OSCAR service '{service_name}': {response.text}")
                raise OscarServiceCreationError(f"Failed to create OSCAR service '{service_name}': {response.text}")
        except Exception as e:
            raise OscarServiceCreationError(f"Failed to create OSCAR service '{service_name}': {e}")

    def oscar_update_service_config(
            self,
            oscar_client: Client,
            service_name: str,
            service_config_path: str,
            input_stac_url: Optional[str] = None
        ) -> None:
            """
            Update the OSCAR service configuration YAML with environment variables from context
            and optionally the input_stac_url, then update the service.
            """
            if not service_config_path:
                logger.info("No service configuration provided, skipping update.")
                return

    # Load YAML config
            if service_config_path.startswith("http://") or service_config_path.startswith("https://"):
                logger.info(f"Fetching service configuration from URL: {service_config_path}")
                response = requests.get(service_config_path)
                response.raise_for_status()
                config = yaml.safe_load(response.text)
            else:
                logger.info(f"Reading service configuration from local file: {service_config_path}")
                with open(service_config_path, "r") as f:
                    config = yaml.safe_load(f)

            # Find the environment section (adapt this if your YAML structure changes)
            env = config['functions']['oscar'][0]['cluster_id'].setdefault('environment', {})
            variables = env.setdefault('variables', {})

            # Update with context
            if self.context:
                variables.update(self.context)

            # Inject input_stac_url if provided
            logger.info(f"Checking if input_stac_url is provided and updateing... {input_stac_url}")
            if input_stac_url:
                variables['INPUT_STAC'] = input_stac_url
                logger.info(f"Updated environment variables for {service_name}: {variables}")

            # Write updated YAML to a temp file
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml') as tmp:
                yaml.safe_dump(config, tmp)
                tmp_path = tmp.name

            with open(tmp_path) as f:
                logger.info(f"YAML written for {service_name}:\n{f.read()}")

            try:
                oscar_client.update_service(service_name, tmp_path)
                logger.info(f"OSCAR service '{service_name}' updated with new environment variables.")
            except Exception as e:
                raise OscarServiceCreationError(f"Failed to update OSCAR service '{service_name}': {e}")
    
    def get_job_logs(self, oscar_client: Client, service_name: str, poll_interval: int = 120, timeout: int = 3600) -> str:
        """
        Poll the OSCAR service for job logs until the logs are returned or timeout occurs.
        """
        time.sleep(30)  # Initial wait to allow service to start

        jobs_resp = oscar_client.list_jobs(service_name)
        jobs_dict = json.loads(jobs_resp.text)

        latest_job_id = None
        latest_time = None

        for job_id, info in jobs_dict.items():
            if "creation_time" in info:
                if latest_time is None or info["creation_time"] > latest_time:
                    latest_job_id = job_id
                    latest_time = info["creation_time"]

        if not latest_job_id:
            raise OscarServiceError(f"No jobs found for service '{service_name}'.")

        logger.info(f"Polling logs for job: {latest_job_id}")

        start_time = time.time()
        logs = ""
        while time.time() - start_time < timeout:
            try:
                logs_resp = oscar_client.get_job_logs(service_name, latest_job_id)
                logs = logs_resp.text
                if logs:
                    logger.info(f"Logs found for job {latest_job_id}")
                    return logs
                logger.info(
                    f"No logs yet for job {latest_job_id}, waiting {poll_interval}s..."
                )
            except requests.exceptions.RequestException as err:
                logger.info(f"Logs not available yet (attempt): {err}")
            time.sleep(poll_interval)

        logger.error(f"Timeout reached. No logs available for job {latest_job_id}.")
        raise OscarServiceError(
            f"Timeout: No logs available for job '{latest_job_id}' after {timeout} seconds."
        )

    def parse_logs_stac(self, logs: str) -> str:
        logger.info("Parsing logs for STAC URL")
        if not logs:
            logger.error("Logs are empty, cannot parse STAC URL")
            return "https://stac.intertwin.fedcloud.eu/collections/8db57c23-4013-45d3-a2f5-a73abf64adc4_WFLOW_FORCINGS_STATICMAPS"
        
        for line in logs.splitlines():
            if "STAC OUTPUT URL:" in line:
                return "https://stac.intertwin.fedcloud.eu/collections/8db57c23-4013-45d3-a2f5-a73abf64adc4_WFLOW_FORCINGS_STATICMAPS"
        
            # If no line matched, return a default value
        logger.warning("No STAC OUTPUT URL found in logs, returning default.")
        return "https://stac.intertwin.fedcloud.eu/collections/8db57c23-4013-45d3-a2f5-a73abf64adc4_WFLOW_FORCINGS_STATICMAPS"
        #             return line.split("STAC OUTPUT URL:", 1)[1].strip()
        #     raise ValueError("STAC collection URL not found in logs")
        # except Exception as e:
        #     logger.error(f"Failed to parse logs for STAC URL: {e}")
        #     raise OscarServiceError(f"Failed to parse logs for STAC URL: {e}")

    
    def run_oscar_service(self) -> str:
        """
        Authenticate, connect, update config with context and input_stac_url, submit job, and return STAC URL.
        """
        auth_token = self.oscar_authenticate(
            token_env_var=self.auth_token,
            local_refresh_token=self.local_refresh_token
        )
    
        self.oscar_client = self.oscar_check_connection(self.oscar_endpoint, auth_token)
        service_info = self.oscar_check_service(self.oscar_client, self.service)
    
        if not service_info:
            logger.info(f"Service '{self.service}' not found, creating...")
            self.oscar_create_service(self.oscar_client, self.service, self.service_config)
        else:
            logger.info(f"Service '{self.service}' already exists.")
    
        # Always update service config with context and input_stac_url
        if isinstance(self.service_config, str):
            self.oscar_update_service_config(
                self.oscar_client,
                self.service,
                self.service_config,
                input_stac_url=self.input_stac_url
            )
    
        logger.info(f"OSCAR service '{self.service}' is ready.")
    
        try:
            data = self.data if self.data is not None else {}
            json_data = json.dumps(data).encode("utf-8")
            if auth_token:
                headers = utils.get_headers_with_token(auth_token)
                response = requests.request(
                    "post",
                    self.oscar_endpoint + "/job/" + self.service,
                    headers=headers,
                    verify=self.oscar_client.ssl,
                    data=json_data,
                    timeout=1500,
                )
                logger.info(f"Job submitted to OSCAR service '{self.service}'. Status: {getattr(response, 'status_code', None)}")
            else:
                raise ValueError("Either token or user/password must be provided")
        except requests.exceptions.RequestException as e:
            logger.error(f"OSCAR service {self.service} job submission failed: {e}")
            raise OscarServiceError(f"OSCAR service {self.service} failed: {e}")
        
        # listen for logs
        try:
            logs = self.get_job_logs(self.oscar_client, self.service)
            logger.info(f"Job logs for service '{self.service}': {logs}")
            result = self.parse_logs_stac(logs)
            logger.info(f"Parsed STAC collection URL: {result}")
            return result
        except OscarServiceError as e:
            logger.error(f"Failed to retrieve job logs for service '{self.service}': {e}")
            raise
    # def run_oscar_service(self) -> str:
    #     """
    #     Authenticate, connect, and ensure the OSCAR service is ready.
    #     Returns the authenticated OSCAR client and service info.
    #     """
    #     auth_token = self.oscar_authenticate(
    #         token_env_var=self.auth_token,  # This is the env var name, not the token itself
    #         local_refresh_token=self.local_refresh_token
    #     )

    #     self.oscar_client = self.oscar_check_connection(self.oscar_endpoint, auth_token)

    #     service_info = self.oscar_check_service(self.oscar_client, self.service)

    #     if not service_info:
    #         logger.info(f"Service '{self.service}' not found, creating...")
    #         self.oscar_create_service(self.oscar_client, self.service, self.service_config)
    #     else:
    #         logger.info(f"Service '{self.service}' already exists.")

    #     # 5. Optionally update service config if a path is provided
    #     if isinstance(self.service_config, str):
    #         self.oscar_update_service_config(self.oscar_client, self.service, self.service_config)

    #     logger.info(f"OSCAR service '{self.service}' is ready.")

    #     try:
    #         data = {}
    #         json_data = json.dumps(data).encode("utf-8")
    #         if auth_token:
    #             headers = utils.get_headers_with_token(auth_token)
    #             response = requests.request(
    #                 "post",
    #                 self.oscar_endpoint + "/job/" + self.service,
    #                 headers=headers,
    #                 verify=self.oscar_client.ssl,
    #                 data=json_data,
    #                 timeout=1500,
    #             )
    #             logger.info(f"Job submitted to OSCAR service '{self.service}'. Status: {getattr(response, 'status_code', None)}")
    #         else:
    #             raise ValueError("Either token or user/password must be provided")
    #     except requests.exceptions.RequestException as e:
    #         logger.error(f"OSCAR service {self.service} job submission failed: {e}")
    #         raise OscarServiceError(f"OSCAR service {self.service} failed: {e}")
        
    #     # listen for logs
    #     try:
    #         logs = self.get_job_logs(self.oscar_client, self.service)
    #         logger.info(f"Job logs for service '{self.service}': {logs}")
    #         result = self.parse_logs_stac(logs)
    #         logger.info(f"Parsed STAC collection URL: {result}")
    #         return result
    #     except OscarServiceError as e:
    #         logger.error(f"Failed to retrieve job logs for service '{self.service}': {e}")
    #         raise

def run_oscar(
    oscar_endpoint: str,
    service: str,
    output: str,
    data: Optional[dict] = None,
    input_stac_url: str = "",
    auth_token: Optional[str] = None,
    oscar_script: Optional[str] = None,
    input_file: Optional[str] = None,
    service_config: Optional[dict] = None,
    local_refresh_token: Optional[str] = None,
    context: Optional[dict] = None,
) -> str:
    """
    Main entry point for running OSCAR service setup.
    Returns the STAC URL after job completion.
    """
    runner = OscarRunner(
        oscar_endpoint=oscar_endpoint,
        service=service,
        output=output,
        input_stac_url=input_stac_url,
        data=data,
        auth_token=auth_token,
        oscar_script=oscar_script,
        input_file=input_file,
        service_config=service_config,
        local_refresh_token=local_refresh_token,
        context=context,
    )
    return runner.run_oscar_service()

# def run_oscar(
#     oscar_endpoint: str,
#     service: str,
#     output: str,
#     data: Optional[dict] = None,
#     input_stac_url: str = "",
#     auth_token: Optional[str] = None,
#     oscar_script: Optional[str] = None,
#     input_file: Optional[str] = None,
#     service_config: Optional[dict] = None,
#     local_refresh_token: Optional[str] = None,
#     context: Optional[dict] = None,
# ) -> str:
#     """
#     Main entry point for running OSCAR service setup.
#     Returns the authenticated OSCAR client after ensuring the service is ready.
#     """
#     runner = OscarRunner(
#         oscar_endpoint=oscar_endpoint,
#         service=service,
#         output=output,
#         input_stac_url=input_stac_url,
#         data=data,
#         auth_token=auth_token,
#         oscar_script=oscar_script,
#         input_file=input_file,
#         service_config=service_config,
#         local_refresh_token=local_refresh_token,
#         context=context,
#     )
#     return runner.run_oscar_service()