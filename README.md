# OpenEO Processes Dask

## Mini-backend development environment
In addition to process implementations, this repo contains a minimal dockerized environment for developing and testing scalable openEO process implementations using the parallelisation and lazy-loading capabilities of dask and xarray. These implementations can then be scaled to real-world amounts of data with little extra effort simply by configuring a connection to a scaled-up remote dask cluster.  

The mini-backend is intended to be operated through the `Visual Studio Code Remote - Containers` extension, which "lets you use a Docker container as a full-featured development environment".
In this project, the "full-featured development environment" is specified via a docker-compose file that spins up the resources required to simulate the processing aspect of an OpenEO backend:
 
- Dummy dask cluster: 
    - Simple setup with 1 scheduler and 1 worker node that run's locally on the developer's machine.
    - Worker: has additional dependencies to run the process implementations installed.
    - Scheduler: Proxies and intelligently distributes compute tasks from client to attached worker nodes. The dask status dashboard is made available on the host-machine via port-forward to `localhost:8787`
- Devcontainer: 
    - Has all the system-level and Python dependencies necessary for executing the OpenEO process implementations (e.g. GDAL, dask, xarray).
    - Mounts the source code via Docker volumes to provide a uniform reproducible environment for process development.
    - Connects to the dummy dask cluster via the `dask.distributed` API to perform work as if connecting to a larger, externally hosted cluster. Note how this is different to using a `LocalCluster` - these containers run in isolation from the devcontainer and communicate over a network using the `dask.distributed` API.

### Usage
#### VSCode (recommended)
Download the `Visual Studio Code Remote - Containers` and run the "Reopen in Container" command.

#### Pure docker
To spin up the environment run `docker compose -f ./.devcontainer/docker-compose.yml up -d`
To rebuild the entire setup (e.g. for when you've changed something in the Dockerfiles) use the following command with the ` --no-cache` flag: `docker compose -f ./.devcontainer/docker-compose.yml build --no-cache`

To get an interactive shell into the running devcontainer (named `openeo-devcontainer`) run `docker exec -it openeo-devcontainer /bin/bash`.
Test that your setup works with `poetry run python -m pytest`. 
To run the example notebooks in `examples` you'll need to setup the devcontainer to host a jupyter notebook. 

To tear the environment down again run `docker compose -f ./.devcontainer/docker-compose.yml down`.