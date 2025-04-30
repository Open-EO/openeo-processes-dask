#TODO:
# Exceptions to be moved to the appropriate module in /processes/exceptions.py
# Adjust the process JSON definition to fit the new function signature
# Test the implementation
# !!!Input data handling!!!
# STAC API output registration

import os
import uuid
import logging
import json
from typing import Optional

from oscar_python.client import Client
import oscar_python._utils as utils
from minio import Minio
import requests


__all__ = ["run_oscar"]

logger = logging.getLogger(__name__)


class OpenEOException(Exception):
    pass

class OpenEOAuthError(OpenEOException):
    pass

class EGIAuthError(OpenEOException):
    pass

class OscarNotAvailable(OpenEOException):
    pass

class OscarUrlError(OpenEOException):
    pass

class OscarServiceNotFound(OpenEOException):
    pass

class OscarServiceCreationError(OpenEOException):
    pass

class OscarServiceError(OpenEOException):
    pass

class MinioConnectionError(OpenEOException):
    pass

class MinioUploadError(OpenEOException):
    pass

class MinioDownloadError(OpenEOException):
    pass


def _get_refresh_token(
    token_env_var: str,
) -> str:
    """
    Get the access token from OpenEO to be used to auth OSCAR

    :param token_env_var: OpenEO auth token environment variable

    :return: refresh token

    :raises OpenEOAuthError: If the token environment variable is not found
    """

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
        return access_token["access_token"]
    except requests.exceptions.RequestException as e:
        raise EGIAuthError(
            f"Failed to retrieve access token: {e}"
        )
    
    
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
    if oscar_endpoint or auth_token is None:
        raise OscarUrlError("OSCAR endpoint or auth token is not provided")

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
) -> None:
    """
    Check if the OSCAR service is available, if not, create it

    :param client: OSCAR client
    :param service: OSCAR service
    :param service_config: OSCAR service config

    :raises OscarServiceNotFound: If the OSCAR service is not found
    :raises OscarServiceCreationError: If the OSCAR service creation fails
    """
    try:
        service_info = oscar_client.get_service(service)
        service_data = json.loads(service_info.text)
        minio_info = service_data["storage_providers"]["minio"]["default"]
        input_info = service_data["input"][0]
        output_info = service_data["output"][0]

        if service_info.status_code == 200:
            logger.info(f"OSCAR service {service} is available")
            return minio_info, input_info, output_info
    except Exception as e:
        logger.info(f"OSCAR service {service} is not available, creating...")
        try:
            creation = oscar_client.create_service(service_config)
            logger.info(f"OSCAR service {creation} created")
        except Exception as e:
            raise OscarServiceCreationError(
                f"OSCAR service {service} creation failed: {e}"
            )


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
        prefix='/'.join(output_info["path"].split("/")[1:]),
        events=["s3:ObjectCreated:*", "s3:ObjectRemoved:*"],
    ) as events:
        for event in events:
            outputfile = event["Records"][0]["s3"]["object"]["key"]
            print(event["Records"][0]["s3"]["object"]["key"])
            break
    
    try:
        client.fget_object(output_info["path"].split("/")[0], 
                       outputfile,
                       output + "/" + outputfile.split("/")[-1])
    except Exception as e:
        raise MinioDownloadError(f"MinIO download failed: {e}")
    
    return output + "/" + outputfile.split("/")[-1]

def _run_oscar_service(client: Client, oscar_endpoint: str, service: str, auth_token: str, output: str) -> str:

    response = None

    try:
        data = {
            "Records": [
                {
                    "requestParameters": {
                        "principalId": "uid",
                        "sourceIPAddress": "ip"
                    },
                }
            ]
        }
        json_data = json.dumps(data).encode('utf-8')
        if auth_token:
            headers = utils.get_headers_with_token(auth_token)
            try:
                response = requests.request("post", oscar_endpoint + "/job/" + service, headers=headers, verify=client.ssl, data=json_data, timeout=1500)
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

    # OpenEO EGI token stuff

    refresh_token = _get_refresh_token(token_env_var)
    auth_token = _get_access_token(refresh_token)

    oscar_client = _check_oscar_connection(oscar_endpoint, auth_token)
    
    # Checks if the service is available, if not, creates it
    # and gets the minio info, input info and output info
    minio_info, input_info, output_info = _check_oscar_service(oscar_client, service, service_config)
    
    minio_client = _connect_minio(minio_info)
    
    # Optional if we want to upload an input file
    random_file_name = _upload_file_minio(minio_client, input_info, input_file)
    
    # Executes the OSCAR service
    response = _run_oscar_service(oscar_client, oscar_endpoint, service, auth_token, output)
    
    # Waits and returns the result
    # Need to check the actual formatting though
    # We will only need the S3 link
    output_file = _wait_and_download_output(_connect_minio(minio_info), output_info, output)

    return output_file
