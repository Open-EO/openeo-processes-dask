from typing import Any

import numpy as np
import pandas as pd

__all__ = ["is_valid"]


def is_valid(x: Any) -> bool:
    null_mask = pd.isnull(np.asarray(x))
    return ~null_mask
