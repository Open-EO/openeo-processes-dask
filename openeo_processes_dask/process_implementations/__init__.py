import logging

logger = logging.getLogger(__name__)

from .arrays import *
from .comparison import *
from .cubes import *
from .inspect import *
from .logic import *
from .math import *
from .text import *

try:
    from .ml import *
except ImportError as e:
    logger.warning(
        "Did not load machine learning processes due to missing dependencies: Install them like this: `pip install openeo-processes-dask[implementations, ml]`"
    )

try:
    from .experimental import *
except ImportError as e:
    logger.warning("Did not load experimental processes.")

import rioxarray as rio  # Required for the .rio accessor on xarrays.

import openeo_processes_dask.process_implementations.cubes._xr_interop
