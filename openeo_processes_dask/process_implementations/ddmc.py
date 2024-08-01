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
    dimension = data.openeo.band_dims
    if not target_band:
        target_band = dimension

    # Mid-Level Clouds
    def MIDCL(data):
        B08 = array_element(data, label=nir08)
        B09 = array_element(data, label=nir09)

        MIDCL = B08 - B09

        MIDCL_result = MIDCL * gain

        return MIDCL_result

    # Deep moist convection
    def DC(data):
        B10 = array_element(data, label=cirrus)
        B12 = array_element(data, label=swir22)

        DC = B10 - B12

        DC_result = DC * gain

        return DC_result

    # low-level cloudiness
    def LOWCL(data):
        B10 = array_element(data, label=cirrus)
        B11 = array_element(data, label=swir16)

        LOWCL = B11 - B10

        LOWCL_result = LOWCL * gain

        return LOWCL_result

    midcl = reduce_dimension(data, reducer=MIDCL, dimension=dimension)
    midcl = add_dimension(midcl, target_band, "midcl")

    dc = reduce_dimension(data, reducer=DC, dimension=dimension)
    dc = add_dimension(dc, target_band, "dc")

    lowcl = reduce_dimension(data, reducer=LOWCL, dimension=dimension)
    lowcl = add_dimension(lowcl, target_band, "lowcl")

    ddmc = merge_cubes(merge_cubes(midcl, dc), lowcl)

    # return a datacube
    return ddmc
