# OpenEO Processes Dask

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
![PyPI - Status](https://img.shields.io/pypi/status/openeo-processes-dask)
![PyPI](https://img.shields.io/pypi/v/openeo-processes-dask)
![Python Version](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)
[![codecov](https://codecov.io/github/Eurac-Research-Institute-for-EO/openeo-processes-dask/branch/fix/dependencies_update/graph/badge.svg?token=RA82MUN9RZ)](https://codecov.io/github/Eurac-Research-Institute-for-EO/openeo-processes-dask)

`openeo-processes-dask` is a collection of Python implementations of [OpenEO processes](https://processes.openeo.org/) based on the [xarray](https://github.com/pydata/xarray)/[dask](https://github.com/dask/dask) ecosystem. It is intended to be used alongside with [openeo-pg-parser-networkx](https://github.com/Open-EO/openeo-pg-parser-networkx), which handles the parsing and execution of [OpenEO process graphs](https://openeo.org/documentation/1.0/developers/api/reference.html#section/Processes/Process-Graphs). There you'll also find a tutorial on how to register process implementations from an arbitrary source (e.g. this repo) to the registry of available processes.

## Installation

### Conda (recommended — mirrors CI)

Conda-forge provides GDAL system libraries and Python bindings as a single coherent package. Once installed, pip does not need to touch GDAL at all.

```bash
conda create -n openeo_processes_dask -c conda-forge python=3.12 gdal
conda activate openeo_processes_dask
pip install openeo-processes-dask[implementations]
```

**Micromamba (lightweight alternative):**

```bash
micromamba create -n openeo_processes_dask -c conda-forge python=3.12 gdal
micromamba activate openeo_processes_dask
pip install openeo-processes-dask[implementations]
```

---

### System packages (Ubuntu/Debian)

If you already have GDAL system libraries, pin the pip `gdal` wheel to the matching version.
Ubuntu 24.04 ships GDAL **3.8.4** — the minimum version required for system-level installation:

```bash
sudo apt-get install gdal-bin libgdal-dev python3-gdal
pip install "gdal==$(gdal-config --version)" openeo-processes-dask[implementations]
```

> This pin avoids the common mismatch between a pip `gdal` wheel and your system `libgdal`. When using conda (above) this is unnecessary because conda-forge provides both the library and the Python binding together.

Note that by default `pip install openeo-processes-dask` only installs the JSON process specs.
In order to install the actual implementations, add the `implementations` extra as shown in the examples above.

---

### Extra build variants

A subset of process implementations with heavy or unstable dependencies are hidden behind these extras:

* **ML processes:**

  ```bash
  pip install openeo-processes-dask[ml]
  ```


⚠️ **Note on GDAL:**
The `implementations` extra depends on GDAL transitively via `rasterio`, `rioxarray`, `odc-stac`, and `geopandas`.
Always install GDAL **first** (via conda-forge or system packages) before pip-installing extras.
The `ml` (`xgboost`) and `deforestation` (`rqadeforestation`) extras do not directly depend on GDAL.
This project requires **GDAL >=3.8.4** (the version shipped by Ubuntu 24.04) and is CI-tested against conda-forge GDAL on Python 3.10–3.13.

**Version-ceiling policy:** Library dependencies in `pyproject.toml` declare only minimum versions (`>=X`). Any upper bounds (`<Y`) needed for CI go into `[tool.poetry.group.ci.dependencies]` (or the conda `ci-environment.yml` pin) so downstream consumers are never blocked. Install the `civersions` extra if your environment also needs the ceiling.

---

## Development environment

This package requires at least Poetry 2.2, see their [docs](https://python-poetry.org/docs/#installation) for installation instructions.

Clone the repository with `--recurse-submodules` to also fetch the process specs:

```bash
git clone --recurse-submodules git@github.com:Open-EO/openeo-processes-dask.git
```

**Development setup (CI pattern — conda-forge GDAL + Poetry):**

```bash
# 1. Create conda env with GDAL (mirrors `.github/ci-environment.yml`)
conda create -n openeo_processes_dask_dev -c conda-forge python=3.12 gdal
conda activate openeo_processes_dask_dev

# 2. Install Poetry deps into the conda env
poetry config virtualenvs.create false
poetry install --with dev,ci --all-extras

# 3. Verify GDAL
gdalinfo --version
python -c "from osgeo import gdal; print('GDAL Python:', gdal.__version__)"
```

> `poetry config virtualenvs.create false` ensures GDAL from conda-forge is visible to pip-installed geospatial packages (`rasterio`, `rioxarray`, etc.). Without this, Poetry's isolated venv may not find the conda-forge GDAL libraries.

---

To add a new core dependency run:

```bash
poetry add some_new_dependency
```

To add a new development dependency run:

```bash
poetry add some_new_dependency --group dev
```

To run the test suite run:

```bash
poetry run python -m pytest
```

Note that you can also use the virtual environment that's generated by poetry as the kernel for the ipynb notebooks.

### Pre-commit hooks

This repo makes use of [pre-commit](https://pre-commit.com/) hooks to enforce linting & a few sanity checks. In a fresh development setup, install the hooks using `poetry run pre-commit install`. These will then automatically be checked against your changes before making the commit.

### Specs

The json specs for the individual processes are tracked as a git submodule in `openeo_processes_dask/specs/openeo-processes`.
The raw json for a specific process can be imported using `from openeo_processes_dask.specs import reduce_dimension`.

To bump these specs to a later version use:
`git -C openeo_processes_dask/specs/openeo-processes checkout <tag>`
`git add openeo_processes_dask/specs/openeo-processes`
