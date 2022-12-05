import geopandas as gpd
import xgboost as xgb

from openeo_processes_dask.process_implementations.ml import fit_regr_random_forest


def test_fit_regr_random_forest(vector_data_cube, dask_client):
    predictors_vars = ["value2"]
    target_var = "value1"

    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube,
        target_var=target_var,
        predictors_vars=predictors_vars,
    )

    assert isinstance(model, xgb.core.Booster)


def test_fit_regr_random_forest_inline_geojson(
    vector_data_cube: gpd.GeoDataFrame, dask_client
):
    predictors_vars = ["value2"]
    target_var = "value1"

    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube.compute().to_json(),
        target_var=target_var,
        predictors_vars=predictors_vars,
    )

    assert isinstance(model, xgb.core.Booster)
