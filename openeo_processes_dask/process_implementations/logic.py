import dask.array as da
import numpy as np

__all__ = ["and_", "or_", "xor", "not_", "if_", "any_", "all_"]


def and_(x, y):
    x_nan = x.where(x == True, False)  # Set NaN to False
    y_nan = y.where(y == True, False)
    logical_and = da.logical_and(x, y)
    logical_and = logical_and.where(x == x_nan, np.nan)
    logical_and = logical_and.where(y == y_nan, np.nan)
    return logical_and


def or_(x, y):
    x_nan = x.where(x == True, False)  # Set NaN to False
    y_nan = y.where(y == True, False)
    logical_or = da.logical_or(x, y)
    logical_or = logical_or.where(x == x_nan, np.nan)
    logical_or = logical_or.where(y == y_nan, np.nan)
    return logical_or


def xor(x, y):
    x_nan = x.where(x == True, False)  # Set NaN to False
    y_nan = y.where(y == True, False)
    logical_xor = da.logical_xor(x, y)
    logical_xor = logical_xor.where(x == x_nan, np.nan)
    logical_xor = logical_xor.where(y == y_nan, np.nan)
    return logical_xor


def not_(x):
    return da.logical_not(x)


def if_(value, accept, reject=np.nan):
    p = value.where(value == 0, accept)
    p = p.where(value == 1, reject)
    return p


def any_(data, ignore_nodata=True, dimension=None):
    data_nan = data.where(data == True, False)  # Set NaN to False
    if ignore_nodata:
        return data_nan.any(dim=dimension)
    else:
        data = data.any(dim=dimension)
        data_nan = data_nan.any(dim=dimension)
        if (data == data_nan).all():  # See if there are NaNs, that were set to False
            return data
        else:
            return data.where(data == data_nan, np.nan)


def all_(data, ignore_nodata=True, dimension=None):
    data_nan = data.where(data == True, False)
    if ignore_nodata:
        return data.all(dim=dimension)  # all ignores NaNs
    else:
        data = data.all(dim=dimension)
        data_nan = data_nan.all(dim=dimension)
        if (data == data_nan).all():  # See if there are NaNs, that were set to False
            return data
        else:
            return data.where(data == data_nan, np.nan)
