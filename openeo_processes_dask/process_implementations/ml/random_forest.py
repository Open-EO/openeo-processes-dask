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

    def load_geometries(geometries):
        if isinstance(geometries, str):
            # we allow loading as filename, URL or geometries dictionary
            try:
                geometries = load_vector_cube(filename=geometries)
            except Exception:
                pass
        if isinstance(geometries, str):
            geometries = load_vector_cube(URL=geometries)
        if isinstance(geometries, dict):
            geometries = load_vector_cube(geometries=geometries)
        return geometries

    def drop_col(df, keep_var):
        if keep_var is None:
            keep_var = []
            for column in df.columns:
                if column not in ["geometry", "id"]:
                    keep_var.append(column)
        if isinstance(keep_var, str):
            keep_var = [keep_var]
        if isinstance(keep_var, list):
            df = df[keep_var]
        return df

    params = {
        "learning_rate": 1,
        "max_depth": 5,
        "num_parallel_tree": int(num_trees),
        "objective": "reg:squarederror",
        "subsample": 0.8,
        "tree_method": "hist",
        "colsample_bynode": 1,
    }

    if isinstance(predictors, xr.DataArray):
        dimensions = predictors.dims
        array_dim = []
        geom_dim = []
        extra_dim = []
        for d in dimensions:
            if d in ["bands", "t", "time"]:
                array_dim.append(d)
            elif d in ["geometry", "geometries"]:
                geom_dim.append(d)
            else:
                extra_dim.append(d)
        if len(geom_dim) != 1:
            raise Exception(f"{predictors} is not a valid vector data cube.")
        if len(extra_dim) > 0:
            for d in extra_dim:
                predictors = predictors.isel({d: 0}).drop_vars(d)
        if len(array_dim) == 1:
            d = array_dim[0]
            predictors = predictors.to_dataset(dim=d)
        elif len(array_dim) == 0:
            predictors = predictors.to_dataset(name="bands")
        else:
            raise Exception(f"{predictors} is not a valid vector data cube.")

        predictors = predictors.xvec.to_geodataframe()

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

    X = drop_col(data_ddf, predictors_vars)

    # This is a workaround for the openeo-python-client current returning inline geojson for this process
    target = load_geometries(target)

    y = drop_col(target, target_var)

    client = dask.distributed.default_client()
    dtrain = xgb.dask.DaskDMatrix(client, X, y)
    output = xgb.dask.train(client, params, dtrain, num_boost_round=1)

    return output["booster"]


def predict_random_forest(
    data: RasterCube,
    model: Booster,
    axis: int = -1,
    context: dict = None,
) -> RasterCube:
    import xgboost as xgb

    if not model:
        if isinstance(context, dict) and "model" in context:
            model = context["model"]

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

    output_shape = list(data.shape)
    output_shape[axis] = 1

    preds = preds_flat.reshape(tuple(output_shape))
    return preds
