from functools import partial

import geopandas as gpd
import numpy as np
import pytest
import xgboost as xgb
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.cubes.apply import apply_dimension
from openeo_processes_dask.process_implementations.ml import (
    fit_curve,
    fit_regr_random_forest,
)
from tests.mockdata import create_fake_rastercube


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


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_fit_curve(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04"],
        backend="dask",
    )

    def fitFunction(t, a, b, c):
        t0 = 2 * np.pi / 31557600 * t
        return a + b * np.cos(t0) + c * np.sin(t0)

    result = fit_curve(origin_cube, parameters={}, function=fitFunction, dimension="t")
