#%%
#load modules
import xarray as xr
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
import geopandas as gpd
from pathlib import Path
import utils
import dask
from rasterio import features, transform


def calculate_nap_top_depth(bottom_selection_depth, thickness):
    nap_top_depth = bottom_selection_depth + thickness.cumsum(dim='layer')
    nap_top_depth = nap_top_depth.where(thickness.notnull())
    return nap_top_depth

def calculate_points_above_gw(adapted_tops, adapted_thickness, gw_stand, organisch_mask, points_per_cm):
    cm_above_gw = adapted_tops - gw_stand
    cm_above_gw = cm_above_gw.where(cm_above_gw > 0)
    cell_thick_above_gw= np.minimum(cm_above_gw, adapted_thickness)
    cm_organisch_above_gw = cell_thick_above_gw.where(organisch_mask).sum('layer')
    return cm_organisch_above_gw * points_per_cm

dask.config.set({"array.slicing.split_large_chunks": True})

if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="kartering_oxidatie")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output

#%%
sm = xr.open_dataset(input_fns.base_data_shifted_fn).chunk({'x': 1000, 'y': 1000, 'layer': -1})
sm['bots_onder_mv'] = sm.tops_onder_mv - sm.thickness

# mask = xr.DataArray(
#     data=features.rasterize(
#         nl_shape.geometry,
#         out_shape=(len(sm.y), len(sm.x)),
#         transform=transform.from_bounds(
#             sm.x.min(), sm.y.min(), sm.x.max(), sm.y.max(),
#             len(sm.x), len(sm.y)
#         ),
#         all_touched = True
#     ),
#     dims=['y', 'x'],
#     coords={'y': sm.y, 'x': sm.x}
# ).chunk(x=1000,y=1000) == 1

#gw data
#######################
#OOK SOMERS GRONDWATERSTANDEN TOEPASSEN? 
#ANDERE DATASET DUS HOE MERGEN?
#TODO: MASK GROUNDWATER WITH LHM SURFACE WATER MASK
#EDIT: DIE BESTAAT NIET HAHA. HOE DAN? 
############################

gw_stand_nap = xr.open_dataarray(input_fns.lhm_heads_fn).chunk(x=200, y = 200, time = 50)
top_l1_da = xr.open_dataarray(input_fns.lhm_top_fn).isel(band=0).drop_vars('band')
gw_stand_mv = gw_stand_nap - top_l1_da
gw_stand_mv =  gw_stand_mv.interp(x=sm.x, y=sm.y, method='nearest')
# min_gw = gw_stand_mv.min('time').compute()
# min_gw.rio.to_raster(results_folder / f"min_gw_2022.tif", compress='lzw')

#################
#for testing
# testcol = [sm.x[1730], sm.y[817]]
# sm = sm.sel(x=testcol[0], y=testcol[1], method = 'nearest').compute()
# gw_stand_mv = gw_stand_mv.isel(time=[10]).sel(x=testcol[0], y=testcol[1], method = 'nearest').compute()
# gw_stand_fine_nap = gw_stand_nap.sel(x=testcol[0], y=testcol[1], method = 'nearest').compute()
################

sm['organisch_mask'] = (sm.mass_fraction_organic>params.organisch_threshold).where(sm.mass_fraction_organic.notnull())
sm['gw_stand'] = gw_stand_mv.chunk(x=-1, y = -1, time = 1)

save_coarse = False
#sm = sm.drop_vars(['lithology', 'mass_fraction_organic', 'thickness'])
##%%
#calculate oxidation risk
#shallow
top_selection_depth = -0.1
bottom_selection_depth = -0.5
shallow_points_per_cm = 1

shallow_tops, shallow_thickness, _ = utils.select_layers(
    lith_col=sm.lithology,
    layer_bottoms=sm.bots_onder_mv,
    layer_tops=sm.tops_onder_mv,
    startdepth=top_selection_depth,
    enddepth=bottom_selection_depth
)

points_shallow = calculate_points_above_gw(
    adapted_tops=shallow_tops,
    adapted_thickness=shallow_thickness,
    gw_stand=sm.gw_stand,
    organisch_mask=sm.organisch_mask,
    points_per_cm=shallow_points_per_cm
)

#diep
top_selection_depth = -0.5
bottom_selection_depth = -0.7
deep_points_per_cm = 0.5

deep_tops, deep_thickness, _ = utils.select_layers(
    lith_col=sm.lithology,
    layer_bottoms=sm.bots_onder_mv,
    layer_tops=sm.tops_onder_mv,
    startdepth=top_selection_depth,
    enddepth=bottom_selection_depth
)

points_deep= calculate_points_above_gw(
    adapted_tops=deep_tops,
    adapted_thickness=deep_thickness,
    gw_stand=sm.gw_stand,
    organisch_mask=sm.organisch_mask,
    points_per_cm=deep_points_per_cm
)

#add points
total_points = points_shallow + points_deep

sm['points_shallow'] = points_shallow
sm['points_deep'] = points_deep
sm['total_points'] = total_points

#total_points = total_points
yearly_points = total_points.sum('time')
sm['yearly_points'] = yearly_points
sm['yearly_points'] = sm['yearly_points'].where(sm['yearly_points'] > 0)

##%%
with ProgressBar():
    if params.save:
        if params.coarsen > 1:
            coarse_factor = int(params.coarsen)
            save_sm = sm.isel(x = slice(None, None, coarse_factor),
                                y = slice(None, None, coarse_factor))  
        else: 
            save_sm = sm
        
        #save to netcdf
        save_sm.to_netcdf(output_fns.oxidatie_data_fn,  encoding={v: {"zlib": True, "complevel": 4} for v in save_sm.data_vars})

        # if save_coarse:
    #     coarse_factor = 20
    #     sm_coarse = sm.sel(x=slice(None, None, coarse_factor), 
    #                         y=slice(None, None, coarse_factor))
    #     sm_coarse.to_netcdf(results_folder / f"oxidatie_data_coarse_{year}.nc")
    # else:
    #     sm.to_netcdf(results_folder / f"oxidatie_data_full_{year}.nc")
    sm.yearly_points.rio.to_raster(output_fns.oxidatie_output_fn, compress='lzw')


# %%
