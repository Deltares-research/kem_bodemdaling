#%% load modules
import xarray as xr
from pathlib import Path
from dask.diagnostics import ProgressBar
import numpy as np

import utils

base_folder = Path(r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse")

def calculate_points_above_gw(adapted_tops, adapted_thickness, gw_stand, lutum_frac, points_per_cm):
    cm_above_gw = adapted_tops - gw_stand
    cm_above_gw = cm_above_gw.where(cm_above_gw > 0)
    cell_thick_above_gw= np.minimum(cm_above_gw, adapted_thickness)
    lutum_frac_above_gw = cell_thick_above_gw*lutum_frac
    cm_lutum_above_gw = lutum_frac_above_gw.sum('layer')
    return cm_lutum_above_gw * points_per_cm

#%%
#load data
#ondergrond data
subsurface_model_fn = base_folder / "AtlantisRuns/atlans_subsurface_model_Knaake.nc"
sm = xr.open_dataset(subsurface_model_fn).chunk(chunks={"x": 200, "y": 200, 'layer': -1})

###################
#ALS IK DE GLG GEBRUIK VAN DE LHM IS DIE HOGER DAN DE L1 TOP? HOE DAN?
year = 2022
lhm_folder = Path(r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse\kartering\lhm_data")
gw_data_fn = lhm_folder / f"lhm_433_heads_l1_{year}.nc"
top_fn = lhm_folder / 'TOP_L1_LHM433.tif'
gw_stand_nap = xr.open_dataarray(gw_data_fn).chunk(x=200, y = 200, time = 50)
glg_nap = utils.calc_avg_lowest_three(gw_stand_nap)
top_l1_da = xr.open_dataarray(top_fn).isel(band=0).drop_vars('band')
glg_stand_mv = glg_nap - top_l1_da
with ProgressBar():
    glg_stand_mv =  glg_stand_mv.interp(x=sm.x, y=sm.y, method='nearest').compute()

tot_thickness = sm.thickness.cumsum('layer').where(sm.thickness.notnull())
sm['tops_onder_mv'] = tot_thickness - tot_thickness.max('layer')
sm['bots_onder_mv'] = sm.tops_onder_mv - sm.thickness
sm['bots_onder_mv'] = xr.where(sm['bots_onder_mv'] > glg_stand_mv, sm['bots_onder_mv'], glg_stand_mv)
#sm['thickness'] = sm['tops_onder_mv'] - sm['bots_onder_mv']
#sm['thickness'] = sm['thickness'].where(sm['thickness'] > 0)
#sm['center_onder_mv'] = sm['bots_onder_mv'] + sm['thickness'] / 2
#pleistoceen_onder_mv = (sm.surface_level - sm.domainbase).fillna(0).compute()
#sm['pleistoceen_mask'] = sm.center_onder_mv < pleistoceen_onder_mv
sm['glg_mv'] = glg_stand_mv
sm = sm.sel(layer=sm.layer[::-1])

sm['klei_fractie'] = xr.where(sm.lithology == 1, 0.1,   # organic
     xr.where(sm.lithology == 2, 1.0,    # clay
     xr.where(sm.lithology == 3, 0.35,    # loam
              0))) 

# testcol = [sm.x[1730], sm.y[817]]
# testcol = [100000, 400000]
# sm = sm.sel(x=testcol[0], y=testcol[1], method = 'nearest').compute()
#%%

#shallow points
top_selection_depth = -0.3
bottom_selection_depth = -0.6
shallow_points_per_m = 0.2*100

shallow_tops, shallow_thickness, _ = utils.select_layers(
    lith_col=sm.lithology,
    layer_bottoms=sm.bots_onder_mv,
    layer_tops=sm.tops_onder_mv,
    startdepth=top_selection_depth,
    enddepth=bottom_selection_depth
)

points_shallow= calculate_points_above_gw(
    adapted_tops=shallow_tops,
    adapted_thickness=shallow_thickness,
    gw_stand=sm['glg_mv'],
    lutum_frac=sm.klei_fractie,
    points_per_cm=shallow_points_per_m
)

#intermediate points
top_selection_depth = -0.6
bottom_selection_depth = -0.9
intermediate_points_per_m = 0.8*100

intermediate_tops, intermediate_thickness, _ = utils.select_layers(
    lith_col=sm.lithology,
    layer_bottoms=sm.bots_onder_mv,
    layer_tops=sm.tops_onder_mv,
    startdepth=top_selection_depth,
    enddepth=bottom_selection_depth
)

points_intermediate= calculate_points_above_gw(
    adapted_tops=intermediate_tops,
    adapted_thickness=intermediate_thickness,
    gw_stand=sm['glg_mv'],
    lutum_frac=sm.klei_fractie,
    points_per_cm=intermediate_points_per_m
)

#deep points
top_selection_depth = -0.9
bottom_selection_depth = -100
deep_points_per_m = 1.0*100

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
    gw_stand=sm['glg_mv'],
    lutum_frac=sm.klei_fractie,
    points_per_cm=deep_points_per_m
)

#total points
total_points = points_deep + points_intermediate + points_shallow

total_points = total_points.where(total_points > 0)

#%%
with ProgressBar():
    total_points.rio.to_raster(base_folder / "kartering" / "resultaten"/ "krimp_points.tif")

# %%
