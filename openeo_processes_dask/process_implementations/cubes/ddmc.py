from openeo_processes_dask.process_implementations.arrays import array_element
from openeo_processes_dask.process_implementations.cubes.general import add_dimension
from openeo_processes_dask.process_implementations.cubes.merge import merge_cubes
from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["ddmc"]


def ddmc(
    data: RasterCube,
    nir08="nir08",
    nir09="nir09",
    cirrus="cirrus",
    swir16="swir16",
    swir22="swir22",
    gain=2.5,
    target_band=None,
):
    dimension = data.openeo.band_dims[0]
    if target_band is None:
        target_band = dimension

    # Mid-Level Clouds
    def MIDCL(data):
        # B08 = array_element(data, label=nir08, axis = axis)

        B08 = data.sel(**{dimension: nir08})

        # B09 = array_element(data, label=nir09, axis = axis)

        B09 = data.sel(**{dimension: nir09})

        MIDCL = B08 - B09

        MIDCL_result = MIDCL * gain

        return MIDCL_result

    # Deep moist convection
    def DC(data):
        # B10 = array_element(data, label=cirrus, axis = axis)
        # B12 = array_element(data, label=swir22, axis = axis)

        B10 = data.sel(**{dimension: cirrus})
        B12 = data.sel(**{dimension: swir22})

        DC = B10 - B12

        DC_result = DC * gain

        return DC_result

    # low-level cloudiness
    def LOWCL(data):
        # B10 = array_element(data, label=cirrus, axis = axis)
        # B11 = array_element(data, label=swir16, axis = axis)
        B10 = data.sel(**{dimension: cirrus})
        B11 = data.sel(**{dimension: swir16})

        LOWCL = B11 - B10

        LOWCL_result = LOWCL * gain

        return LOWCL_result

    # midcl = reduce_dimension(data, reducer=MIDCL, dimension=dimension)
    midcl = MIDCL(data)
    midcl = add_dimension(midcl, name=target_band, label="midcl", type=dimension)

    # dc = reduce_dimension(data, reducer=DC, dimension=dimension)
    dc = DC(data)
    # dc = add_dimension(dc, target_band, "dc")
    dc = add_dimension(dc, target_band, label="dc", type=dimension)

    # lowcl = reduce_dimension(data, reducer=LOWCL, dimension=dimension)
    lowcl = LOWCL(data)
    lowcl = add_dimension(lowcl, target_band, label="lowcl", type=dimension)

    # ddmc = merge_cubes(merge_cubes(midcl, dc), lowcl)
    ddmc1 = merge_cubes(midcl, lowcl)
    ddmc1.openeo.add_dim_type(name=target_band, type=dimension)
    ddmc = merge_cubes(dc, ddmc1, overlap_resolver=target_band)

    # return a datacube
    return ddmc
