# How to handle nodata values in Rastercubes

## Context
The logic processes as defined by OpenEO can return True/False/null. In numpy there is no sentinel value for missing data in boolean/integer/char arrays (like there is with NaN for float).

See [this blog](https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html) for a good overview to missing data handling in Python. Also see [NEP-26](https://numpy.org/neps/nep-0026-missing-data-summary.html) for the state of this issue in the python community.

To implement `and` as specified, instead of this:

**Example A: What you'd expect from an implementation for a logical `and` process**

logical_and = np.logical_and(x, y)

we have to do this:

**Example B: Workaround we have to use to account for null values are returned by logic processes**

```python
nan_x = np.isnan(x)
nan_y = np.isnan(y)
xy = np.logical_and(x, y)
nan_mask = np.logical_and(nan_x, xy)
xy = np.where(~nan_mask, xy, np.nan)
nan_mask = np.logical_and(nan_y, xy)
xy = np.where(~nan_mask, xy, np.nan)
return xy
Note that the output array of Example A has the dtype `bool_`, whereas the output array of Example B has dtype `float64`. This is because the missing data value `np.nan` is only defined for dtype `float64` and thus the entire array needs to be upcast to `float64`.

See these numpy snippets to confirm this:

>>> np.array([True, False])
array([ True, False], dtype='bool')  # boolean array
>>> np.array([True, False]).itemsize
1  # one byte per item
>>> np.array([True, False, np.nan])
array([ 1., 0., nan], dtype='float64')  # casts to float64
>>> np.array([True, False, np.nan]).itemsize
8
Apart from looking really awkward, this also has direct performance implications, as the memory footprint of this array is multiplied by 8 and operations with float64 arrays will also be slower than on pure boolean arrays.

We've had a handful of discussions with the maintainers of openeo-processes here:
- Missing data handling in boolean processes [#410](https://github.com/Open-EO/openeo-processes/issues/410)
- first/last: getting first/last element of empty array should error [#408](https://github.com/Open-EO/openeo-processes/issues/408)
- initial PR where this came up: https://github.com/Open-EO/openeo-processes-dask/pull/40#issuecomment-1411839632

The outcomes of these discussions were as follows (in my recollection):
- numpy was initially designed as a linear algebra library, which is the historic reason why the options for handling nodata are so limited
- EO data has missing values all the time, therefore the OpenEO datacube abstraction needs to handle `null` even at the level of boolean arrays
-

Given this, we are left with no other choice than addressing this in our process implementations.

We've evaluated the following approaches:
1) Using the np.ma.MaskedArray library
    - We could return np.ma.masked whenever a null value is produced
    - This would require us to cast every datacube to use masked arrays as the backend, because child processes (e.g. `mean` when called in `reduce_dimension`) cannot affect the container type of their parent datacube (i.e. turn an np.array into a np.ma.MaskedArray).
    - xarray functions like `isnull` wouldn't necessarily work correctly with this
2) Do it like xarray does it
    - see [xarray null handling](https://github.com/pydata/xarray/blob/da8746b46265a61a5a5020924d27aeccd1f43f98/xarray/core/duck_array_ops.py#L116). Basically use `np.nan` for null values and use `pd.isnull()` to check for nullness
    - This means that in favor of null-handling, we'll incur performance penalties when working with arrays of a dtype different than `float` (int/bool/char)
    - But xarray nullness related functionality should still work.

## Decision
We think that the best solution at this stage is to imitate xarray, use `pd.isnull()` to check for null-ness and return `np.nan` whenever the spec wants a process to return `null`.
Both approaches 1) and 2) would mean changing the default array container type to either masked array or sparse arrays

## Consequences
This has performance implications, as this will cause arrays of a dtype different than `float` (int/bool/char) to always be upcast to `float64`.
On the plus-side, this aligns closest with the process definitions in the specification, so it should be easier to follow them faithfully in the future.
