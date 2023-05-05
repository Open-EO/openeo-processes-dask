from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def is_valid(x: Any) -> ArrayLike:
    null_mask = pd.isnull(np.asarray(x))
    return ~null_mask
