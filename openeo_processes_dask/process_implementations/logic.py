from typing import Any

import numpy as np
import pandas as pd


def is_valid(x: Any) -> bool:
    null_mask = pd.isnull(np.asarray(x))
    return ~null_mask
