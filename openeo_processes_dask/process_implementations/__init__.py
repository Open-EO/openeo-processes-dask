import logging

logger = logging.getLogger(__name__)


from .arrays import *
from .cubes import *
from .math import *

try:
    from .ml import *
except ImportError as e:
    logger.warning(
        "Did not load machine learning processes due to missing dependencies: Install them like this: `pip install openeo-processes-dask[implementations, ml]`",
        e,
    )

try:
    from .experimental import *
except ImportError as e:
    logger.warning(
        "Did not load experimental processes due to missing dependencies. Install them like this: `pip install openeo-processes-dask[implementations, experimental]`",
        e,
    )

import rioxarray as rio  # Required for the .rio accessor on xarrays.

import openeo_processes_dask.process_implementations.cubes._xr_interop
