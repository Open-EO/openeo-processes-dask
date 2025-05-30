# TODO: GET RID OF ALL THE LOGS WHICH LOG SENSITIVE DATA
# THIS IS ONLY MEANT FOR DEVELOPMENT
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import json
import logging
import os
import uuid
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
    EGIAuthError
)

import oscar_python._utils as utils
import requests
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


def _get_refresh_token(
    token_env_var: Optional[str]= None,
    local_refresh_token: Optional[str] = None
) -> str:
    """
    Get the access token from OpenEO to be used to auth OSCAR

    :param token_env_var: OpenEO auth token environment variable

    :return: refresh token

    :raises OpenEOAuthError: If the token environment variable is not found
    """
    logger.info(f"Getting auth token...")
    if local_refresh_token:
        with open(local_refresh_token) as f:
            token_data = json.loads(f.read())
            logger.info(f"Using local refresh token from {local_refresh_token}")
        refresh_token = token_data["https://aai.egi.eu/auth/realms/egi"]["openeo-platform-default-client"]["refresh_token"].strip()
        return refresh_token
    else:
        logger.info(f"Using access token from environment variable {token_env_var}")
        refresh_token = os.getenv(token_env_var)
        if refresh_token is None:
            raise OpenEOAuthError(
                f"OpenEO auth token environment variable {token_env_var} not found"
            )
        return refresh_token


def _get_access_token(refresh_token: str) -> str:
    """
    Get the access token using the refresh token.

    :param refresh_token: The refresh token to exchange for an access token.
    :return: The access token.
    :raises ValueError: If the token retrieval fails.
    """
    logger.info(f"Getting access token from a local refresh token...")
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
        access_token = response.json()
        access_token = access_token["access_token"]
        if access_token is not None:
            logger.info(f"Access token retrieved successfully")
        return access_token
    except requests.exceptions.RequestException as e:
        raise EGIAuthError(f"Failed to retrieve access token: {e}")


def _check_oscar_connection(
    oscar_endpoint: str,
    auth_token: str,
) -> Client:
    """
    Check if the OSCAR connection is available

    :param oscar_endpoint: OSCAR endpoint
    :param auth_token: OpenEO auth token

    :return: OSCAR client
    :raises OscarNotAvailable: If the OSCAR connection is not available
    :raises OscarUrlError: If the OSCAR endpoint is not provided
    """
    logger.info(f"Checking OSCAR connection {oscar_endpoint}")
    #if oscar_endpoint or auth_token is None:
        #raise OscarUrlError("OSCAR endpoint or auth token is not provided")

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
        raise OscarNotAvailable(f"OSCAR is not available: {e}")
    return oscar_client


def _check_oscar_service(
    oscar_client: Client, service: str, service_config: Optional[dict] = None
) -> tuple:
    logger.info(f"Checking OSCAR service {service}")

    try:
        service_info = oscar_client.get_service(service)
        logger.info(f"Service info: {service_info}")
        service_data = json.loads(service_info.text)

        minio_info = service_data["storage_providers"]["minio"]["default"]
        input_info = service_data["input"][0]
        output_info = service_data["output"][0]

        logger.info(f"OSCAR service {service} is available")
        return minio_info, input_info, output_info

    except Exception as e:
        logger.info(f"OSCAR service {service} not found, attempting creation.")
        if not service_config:
            raise OscarServiceCreationError(f"Service config not provided for creation.")
        
        try:
            creation = oscar_client.create_service(service_config)
            logger.info(f"Service creation response: {creation}")

            service_info = oscar_client.get_service(service)
            service_data = json.loads(service_info.text)
            minio_info = service_data["storage_providers"]["minio"]["default"]
            input_info = service_data["input"][0]
            output_info = service_data["output"][0]

            logger.info(f"OSCAR service {service} successfully created")
            return minio_info, input_info, output_info

        except Exception as e2:
            raise OscarServiceCreationError(f"OSCAR service {service} creation failed: {e2}")

def _connect_minio(minio_info) -> Minio:
    """
    Create a connection to MinIO

    :param minio_info: MinIO info

    :return: MinIO client

    :raises MinioConnectionError: If the MinIO connection fails
    """
    try:
        minio_client = Minio(
            minio_info["endpoint"].split("//")[1],
            minio_info["access_key"],
            minio_info["secret_key"],
        )
    except Exception as e:
        raise MinioConnectionError(f"MinIO connection failed: {e}")
    return minio_client


def _upload_file_minio(client: Minio, input_info, input_file):
    """
    Upload a file to MinIO

    :param client: MinIO client
    :param input_info: Input info
    :param input_file: Input file

    :return: Random file name

    :raises MinioUploadError: If the MinIO upload fails
    """
    random = uuid.uuid4().hex + "_" + input_file.split("/")[-1]
    try:
        client.fput_object(
            input_info["path"].split("/")[0],
            "/".join(input_info["path"].split("/")[1:]) + "/" + random,
            input_file,
        )
    except Exception as e:
        raise MinioUploadError(f"MinIO upload failed: {e}")

    return random.split("_")[0]


def _wait_and_download_output(client: Client, output_info, output) -> str:
    """
    Wait for the output file to be available in MinIO and download it

    :param client: OSCAR client
    :param output_info: Output info
    :param output: Output path

    :return: Output file path

    :raises MinioDownloadError: If the MinIO download fails
    """

    with client.listen_bucket_notification(
        output_info["path"].split("/")[0],
        prefix="/".join(output_info["path"].split("/")[1:]),
        events=["s3:ObjectCreated:*", "s3:ObjectRemoved:*"],
    ) as events:
        for event in events:
            outputfile = event["Records"][0]["s3"]["object"]["key"]
            print(event["Records"][0]["s3"]["object"]["key"])
            break

    try:
        client.fget_object(
            output_info["path"].split("/")[0],
            outputfile,
            output + "/" + outputfile.split("/")[-1],
        )
    except Exception as e:
        raise MinioDownloadError(f"MinIO download failed: {e}")

    return output + "/" + outputfile.split("/")[-1]


def _run_oscar_service(
    client: Client, oscar_endpoint: str, service: str, auth_token: str, output: str
) -> str:
    response = None

    try:
        data = {
            "Records": [
                {
                    "requestParameters": {
                        "principalId": "uid",
                        "sourceIPAddress": "ip",
                    },
                }
            ]
        }
        json_data = json.dumps(data).encode("utf-8")
        if auth_token:
            headers = utils.get_headers_with_token(auth_token)
            try:
                response = requests.request(
                    "post",
                    oscar_endpoint + "/job/" + service,
                    headers=headers,
                    verify=client.ssl,
                    data=json_data,
                    timeout=1500,
                )
            except requests.exceptions.RequestException as e:
                raise OscarServiceError(f"OSCAR service {service} failed: {e}")
        else:
            raise ValueError("Either token or user/password must be provided")
    except Exception as err:
        print("Failed with: ", err)
    return response


def run_oscar(
    token_env_var: str,
    oscar_endpoint: str,
    service: str,
    output: str,
    input_file: Optional[str] = None,
    service_config: Optional[dict] = None,
    local_refresh_token: Optional[str] = None,
) -> str:
    """
    Run the OSCAR service

    :param token_env_var: OpenEO auth token environment variable
    :param oscar_endpoint: OSCAR endpoint
    :param service: OSCAR service
    :param service_config: OSCAR service config
    :param input_file: Input file
    :param output: Output path

    :return: Output file path

    :raises OscarNotAvailable: If the OSCAR connection is not available
    :raises OscarUrlError: If the OSCAR endpoint is not provided
    :raises OscarServiceNotFound: If the OSCAR service is not found
    :raises OscarServiceCreationError: If the OSCAR service creation fails
    :raises MinioConnectionError: If the MinIO connection fails
    :raises MinioUploadError: If the MinIO upload fails
    :raises MinioDownloadError: If the MinIO download fails
    """

    #TODO: Make minio/s3 optional

    logger.info("OSCAR process started")

    refresh_token = _get_refresh_token(token_env_var, local_refresh_token)
    access_token = _get_access_token(refresh_token)
    
    oscar_client = _check_oscar_connection(oscar_endpoint, access_token)

    minio_info, input_info, output_info = _check_oscar_service(
        oscar_client, service, service_config
    )
    
    minio_client = _connect_minio(minio_info)

    if input_file is not None:
        random_file_name = _upload_file_minio(minio_client, input_info, input_file)
    else:
        pass

    response = _run_oscar_service(
        oscar_client, oscar_endpoint, service, access_token, output
    )

    output_file = _wait_and_download_output(
        _connect_minio(minio_info), output_info, output
    )

    return output_file
