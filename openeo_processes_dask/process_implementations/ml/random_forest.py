from typing import List, Optional, Union
from openeo_processes_dask.process_implementations.data_model import RasterCube, VectorCube
import numpy as np
import geopandas as gpd
import dask_geopandas
import xgboost as xgb
import dask
import xarray as xr
import dask.distributed

__all__ = ["fit_regr_random_forest", "predict_random_forest"]


def fit_regr_random_forest(
    predictors: VectorCube, 
    target: VectorCube,
    num_trees: int = 100, 
    max_variables: Optional[Union[int, str]] = None,  
    predictors_vars: Optional[List[str]] = None, 
    target_var: str = None, 
    **kwargs
    ): # -> RegressionModel
    params = {
        'learning_rate': 1,
        'max_depth': 5,
        'num_parallel_tree': int(num_trees),
        'objective': 'reg:squarederror',
        'subsample': 0.8,
        'tree_method': 'hist',
        'colsample_bynode': 1}

    if isinstance(predictors, gpd.GeoDataFrame):
        predictors = dask_geopandas.from_geopandas(predictors, npartitions=1)

    if isinstance(predictors, dask_geopandas.core.GeoDataFrame):
        data_ddf = predictors.to_dask_dataframe().reset_index().repartition(npartitions=1)
        
    if not isinstance(predictors, dask.dataframe.DataFrame):
        raise Exception('[!] No compatible vector input data has been provided.')

    if predictors_vars is not None:
        X = data_ddf.drop(data_ddf.columns.difference(predictors_vars), axis=1)
    else:
        X = data_ddf

    y = target.drop(target.columns.difference([target_var]), axis=1)
    
    client = dask.distributed.default_client()
    dtrain = xgb.dask.DaskDMatrix(client, X, y)
    output = xgb.dask.train(client, params, dtrain, num_boost_round=1)

    return output

def predict_random_forest(
    data: RasterCube, 
    dimension: str, 
    model: xgb.Booster, 
) -> RasterCube:

    # Detect whether all features that the model was trained with are present in this data.
    model_features = np.array(model.feature_names, dtype=object)
    data_features = data.get_index(dimension).values
    missing_features = np.setdiff1d(model_features, data_features)
    if missing_features.size > 0:
        raise Exception(f"Provided predictors are missing the following features: {missing_features.tolist()}")

    # Drop any feature columns that the original model wasn't trained on
    idx_to_drop = np.setdiff1d(data_features, model_features)
    data = data.drop_sel({dimension: idx_to_drop})

    # Stack xarray to produce correct shape for xgb.dask.inplace_predict
    non_feature_dims = set(data.dims)
    non_feature_dims.remove(dimension)

    X = dask.array.reshape(data.data, (len(model_features), -1)).transpose()

    # Run prediction
    client = dask.distributed.default_client()
    preds_flat = xgb.dask.inplace_predict(client, model, X)

    # Construct the output rastercube
    output_coords = {coord_name: data.coords[coord_name] for coord_name in non_feature_dims}

    # Output shape needs to be (1, data.x, data.y)
    output_shape = tuple([1] + [len(data[coord]) for coord in output_coords])
    output_array = dask.array.reshape(preds_flat, output_shape)

    # We call this coord "bands" because the save_result logic expects it to be "bands"
    output_coords["bands"] = np.array(["result"])

    preds_xr = xr.DataArray(
        data=output_array,
        coords=output_coords,
        dims=["bands"] + [dim for dim in output_coords.keys() if not dim == "bands"],
        attrs=data.attrs
    )
    
    preds_xr = preds_xr.rio.write_crs(data.rio.crs)

    return preds_xr