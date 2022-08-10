ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=ubuntu
ARG GROUPNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000

FROM osgeo/gdal:ubuntu-small-3.5.0 AS devcontainer
ARG DEBIAN_FRONTEND
ARG USERNAME
ARG GROUPNAME
ARG USER_UID
ARG USER_GID

ENV TZ=Etc/GMT \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry/" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    VENV_PATH=".venv/" \
    APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

ENV PATH="$POETRY_HOME/bin:/home/$USERNAME/.local/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update -qy && \
    apt-get install -qy --no-install-recommends \
        #  libpq-dev and build-essential are necessary for psycopg2, which is required for datacube
        build-essential \
        libpq-dev \
        graphviz

# Adapted from https://github.com/TheKevJames/tools/blob/master/docker-nox/Dockerfile
RUN apt-get update -qy && \
    apt-get install -qy --no-install-recommends \
        ca-certificates \
        curl \
        gnupg2 && \
    . /etc/os-release && \
    echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/deadsnakes.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 && \
    apt-get update -qy && \
    apt-get install -qy --fix-missing --no-install-recommends \
        git \
        openssh-client \
        python3.6 \
        python3.6-dev \
        python3.6-distutils \
        python3.6-venv \
        python3.7 \
        python3.7-dev \
        python3.7-distutils \
        python3.7-venv \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3.8-venv \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
        python3.9-venv \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3.10-venv \
        python3-pip \
        python3-distutils
    
# Install Poetry - respects $POETRY_VERSION & $POETRY_HOME
ENV POETRY_VERSION=1.1.13
RUN curl -sSL https://install.python-poetry.org | python3 -

RUN useradd -ms /bin/bash --uid $USER_UID --user-group $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME
USER $USER_UID

ENV PATH="$POETRY_HOME/bin:/home/$USERNAME/.local/bin:$VENV_PATH/bin:$PATH"

RUN curl -fsS https://bootstrap.pypa.io/get-pip.py --output /tmp/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

RUN python3.6 -m pip install --user --no-cache-dir --upgrade pip && \
    python3.7 -m pip install --user --no-cache-dir --upgrade pip && \
    python3.8 -m pip install --user --no-cache-dir --upgrade pip && \
    python3.9 -m pip install --user --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --user --no-cache-dir --upgrade pip && \
    rm -rf /var/cache/apt/lists

RUN python3.10 -m pip install --user --no-cache-dir 'nox-poetry==1.0.0'

COPY pyproject.toml poetry.lock /home/$USERNAME/
RUN cd /home/$USERNAME && \ 
    poetry install --no-root --extras 'load-save vector ml'

# Build the wheel in an ephemeral stage to make it available to dask workers later!
FROM devcontainer as wheel-builder
ARG USERNAME

COPY . /home/$USERNAME/
WORKDIR /home/$USERNAME/

RUN poetry build --format wheel

# Dask worker stage
FROM ghcr.io/dask/dask:2022.8.0 AS dask-worker
ARG USERNAME

# store previous pwd
ENV DASK_WORKER_CWD="$PWD"

COPY --from=wheel-builder /home/$USERNAME/dist /home/$USERNAME/dist

RUN apt-get update -qy && \
    apt-get install -qy --no-install-recommends \
        #  libpq-dev and build-essential are necessary for psycopg2, which is required for datacube
        build-essential \
        libpq-dev

# Find the prebuilt wheel and install its dependencies
RUN cd /home/$USERNAME/dist && \
    pattern="*.whl" && \
    files=( $pattern ) && \
    export PIP_INSTALL_WHEEL="${files[0]}[dask-worker]" && \
    pip install $PIP_INSTALL_WHEEL

# We need only the dependencies, so get rid of 
RUN pip uninstall openeo-processes -y
