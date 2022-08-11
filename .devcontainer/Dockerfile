ARG WORKDIR=/opt/openeo-processes-dask
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=ubuntu

FROM osgeo/gdal:ubuntu-small-3.5.0 AS devcontainer-base

ARG POETRY_VERSION=1.1.14
ARG PIP_NO_CACHE_DIR 1
ARG PYTHONDONTWRITEBYTECODE 1
ARG PYTHONBUFFERED 1
ARG POETRY_VIRTUALENVS_IN_PROJECT=0
ARG POETRY_HOME="/opt/poetry/"
ARG VIRTUAL_ENV=/opt/openeo-processes-dask/.venv/
ARG WORKDIR
ARG USER_ID
ARG GROUP_ID
ARG USERNAME

# Do this to make these env variables available to all stages of the build (see https://github.com/moby/moby/issues/37345#issuecomment-400250849)
ENV POETRY_VERSION=$POETRY_VERSION
ENV PIP_NO_CACHE_DIR=$PIP_NO_CACHE_DIR
ENV PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE
ENV PYTHONBUFFERED=$PYTHONBUFFERED
ENV POETRY_HOME=$POETRY_HOME
ENV WORKDIR=$WORKDIR
ENV POETRY_VIRTUALENVS_IN_PROJECT=$POETRY_VIRTUALENVS_IN_PROJECT
ENV VIRTUAL_ENV=$VIRTUAL_ENV

ENV PATH="${POETRY_HOME}/bin:$VIRTUAL_ENV/bin:${PATH}"

RUN apt-get update -qy && \
    apt-get install -qy --no-install-recommends \
        #  libpq-dev and build-essential are necessary for psycopg2, which is required for datacube
        build-essential \
        libpq-dev \
        graphviz \
        git 

RUN apt-get update -qy && \
    apt-get install -qy --no-install-recommends python3 python3-distutils python3-venv

# Install Poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -

# Create the user so the program doesn't run as root. This increases security of the container.
RUN addgroup --gid $GROUP_ID $USERNAME
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USERNAME

RUN mkdir -p -m=00775 ${WORKDIR}
RUN chown -R $USER_ID:$GROUP_ID ${WORKDIR}

USER $USERNAME

WORKDIR ${WORKDIR}

FROM devcontainer-base AS devcontainer-builder
# Note that this step needs to happen in the directory that the venv will live after, because this location is written to the shebang in the `.venv/bin/pip` file.

ENV POETRY_VIRTUALENVS_IN_PROJECT 1
ARG USER_ID
ARG GROUP_ID
ARG USERNAME

ARG CACHEBUST=5
USER $USERNAME

# Copy in project dependency specification.
COPY --chown=${USER_ID}:${GROUP_ID} pyproject.toml ${WORKDIR}/

RUN poetry install --no-root

RUN rm pyproject.toml

FROM devcontainer-base as devcontainer
ARG USER_ID
ARG GROUP_ID
ARG USERNAME
ARG WORKDIR
ARG VIRTUAL_ENV

ENV POETRY_VIRTUALENVS_IN_PROJECT 0

COPY --from=devcontainer-builder --chown=${USER_ID}:${GROUP_ID} ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# RUN poetry env use ${VIRTUAL_ENV}

# Build the wheel in an ephemeral stage to make it available to dask workers later!
FROM devcontainer-builder as wheel-builder
ARG USER_ID
ARG GROUP_ID
ARG USERNAME
ARG WORKDIR
ARG VIRTUAL_ENV

ENV POETRY_VIRTUALENVS_IN_PROJECT 0

COPY --from=devcontainer-builder --chown=${USER_ID}:${GROUP_ID} ${WORKDIR}/.venv/ ${VIRTUAL_ENV}

COPY . ${WORKDIR}
RUN poetry build --format wheel

# BUILD DASK WORKER
FROM ghcr.io/dask/dask:2022.8.0 AS dask-worker

ARG WORKDIR
# TODO: variable
WORKDIR /opt/openeo-processes-dask/dist/

COPY --from=wheel-builder ${WORKDIR}/dist/ .

RUN apt-get update -qy && \
    apt-get install -qy --no-install-recommends \
        #  libpq-dev and build-essential are necessary for psycopg2, which is required for datacube
        build-essential \
        libpq-dev

# Find the prebuilt wheel and install its dependencies
RUN cd /opt/openeo-processes-dask/dist && \
    pattern="*.whl" && \
    files=( $pattern ) && \
    export PIP_INSTALL_WHEEL="${files[0]}" && \
    pip install $PIP_INSTALL_WHEEL

# We need only the dependencies, so get rid of 
RUN pip uninstall openeo-processes -y
