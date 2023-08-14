from typing import List, Optional, Union

import dask
import dask.distributed
import dask_geopandas
import geopandas as gpd
import numpy as np
import xarray as xr
from xgboost.core import Booster

from openeo_processes_dask.process_implementations.cubes.experimental import (
    load_vector_cube,
)
from openeo_processes_dask.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)

__all__ = ["fit_regr_random_forest", "predict_random_forest"]


def fit_regr_random_forest(
    predictors: VectorCube,
    target: VectorCube,
    num_trees: int = 100,
    max_variables: Optional[Union[int, str]] = None,
    predictors_vars: Optional[list[str]] = None,
    target_var: str = None,
    **kwargs,
) -> Booster:
    import xgboost as xgb

    params = {
        "learning_rate": 1,
        "max_depth": 5,
        "num_parallel_tree": int(num_trees),
        "objective": "reg:squarederror",
        "subsample": 0.8,
        "tree_method": "hist",
        "colsample_bynode": 1,
    }

    if isinstance(predictors, gpd.GeoDataFrame):
        predictors = dask_geopandas.from_geopandas(predictors, npartitions=1)

    if isinstance(predictors, dask_geopandas.core.GeoDataFrame):
        data_ddf = (
            predictors.to_dask_dataframe().reset_index().repartition(npartitions=1)
        )

    if isinstance(predictors, dask.dataframe.DataFrame):
        data_ddf = predictors

    if not isinstance(predictors, dask.dataframe.DataFrame):
        raise Exception("[!] No compatible vector input data has been provided.")

    if predictors_vars is not None:
        X = data_ddf.drop(data_ddf.columns.difference(predictors_vars), axis=1)
    else:
        X = data_ddf

    # This is a workaround for the openeo-python-client current returning inline geojson for this process
    if isinstance(target, str):
        target = load_vector_cube(filename=target)

    y = target.drop(target.columns.difference([target_var]), axis=1)

    client = dask.distributed.default_client()
    dtrain = xgb.dask.DaskDMatrix(client, X, y)
    output = xgb.dask.train(client, params, dtrain, num_boost_round=1)

    return output["booster"]


def predict_random_forest(
    data: RasterCube,
    model: Booster,
    axis: int = -1,
) -> RasterCube:
    import xgboost as xgb

    n_features = len(model.feature_names)
    if n_features != data.shape[axis]:
        raise Exception(
            f"Number of predictors does not match number of features that were trained with."
        )
    # move feature axis to first position and flatten other dimensions
    X = np.moveaxis(data, axis, 0).reshape((n_features, -1)).transpose()

    # Run prediction
    client = dask.distributed.default_client()
    preds_flat = xgb.dask.inplace_predict(client, model, X)

    output_shape = data.shape[0:axis] + data.shape[axis + 1 :]
    preds = preds_flat.reshape(output_shape)
    return preds
