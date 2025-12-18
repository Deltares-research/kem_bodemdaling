#%% load modules
import xarray as xr
from pathlib import Path
from dask.diagnostics import ProgressBar
import numpy as np

import utils

if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="kartering_krimp")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output

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
sm = xr.open_dataset(input_fns.atlans_fn).chunk(chunks={"x": 200, "y": 200, 'layer': -1})

#glg uit LHM (is in meter onder maaiveld, dus positief = onder maaiveld)
glg_da = xr.open_dataarray(input_fns.lhm_glg_fn).chunk(x=200, y = 200)
sm['glg_mv'] = glg_da.interp(x=sm.x, y=sm.y, method='nearest').compute()
sm['glg_mv'] = -sm['glg_mv']  #omzetten naar onder maaiveld (negatief onder maaiveld)
limited_glg = np.maximum(sm['glg_mv'], -2.0)  #limit groundwater level to max 6m depth

tot_thickness = sm.thickness.cumsum('layer').where(sm.thickness.notnull())
sm['tops_onder_mv'] = tot_thickness - tot_thickness.max('layer')
sm['bots_onder_mv'] = sm.tops_onder_mv - sm.thickness
sm['bots_onder_mv'] = xr.where(sm['bots_onder_mv'] > sm.glg_mv, sm['bots_onder_mv'], sm.glg_mv)
sm = sm.sel(layer=sm.layer[::-1])

sm['klei_fractie'] = xr.where(sm.lithology == 1, 0.2,   # organic
     xr.where(sm.lithology == 2, 1.0,    # clay
     xr.where(sm.lithology == 3, 0.35,    # loam
              0))) 

# testcol = [sm.x[1730], sm.y[817]]
# testcol = [100000, 400000]
# sm = sm.sel(x=testcol[0], y=testcol[1], method = 'nearest').compute()
#%%

#shallow points
top_selection_depth = sm['glg_mv']+0.9
bottom_selection_depth = sm['glg_mv']+0.6
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
    gw_stand=limited_glg,
    lutum_frac=sm.klei_fractie,
    points_per_cm=shallow_points_per_m
)

#intermediate points
top_selection_depth = sm['glg_mv']+0.6
bottom_selection_depth = sm['glg_mv']+0.3
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
    gw_stand=limited_glg,
    lutum_frac=sm.klei_fractie,
    points_per_cm=intermediate_points_per_m
)

#deep points
top_selection_depth = sm['glg_mv']
bottom_selection_depth = sm['glg_mv'] + 0.3
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
    gw_stand=limited_glg,
    lutum_frac=sm.klei_fractie,
    points_per_cm=deep_points_per_m
)

sm['points_shallow'] = points_shallow
sm['points_intermediate'] = points_intermediate
sm['points_deep'] = points_deep

#total points
total_points = points_deep + points_intermediate + points_shallow

total_points = total_points.where(total_points > 0)
sm['total_points'] = total_points

savecols = ['points_shallow', 'points_intermediate', 'points_deep', 'total_points', 
            'lithology', 'tops_onder_mv', 'bots_onder_mv', 'klei_fractie',
            'glg_mv']

#%%
with ProgressBar():
    if params.coarsen > 1:
        save_sm = sm[savecols]
        coarse_factor = int(params.coarsen)
        save_sm = sm.isel(x = slice(None, None, coarse_factor),
                                y = slice(None, None, coarse_factor))  
        save_sm.to_netcdf(output_fns.krimp_data_fn,  encoding={v: {"zlib": True, "complevel": 4} for v in save_sm.data_vars})
    
    total_points.rio.write_crs("EPSG:28992", inplace=True)
    total_points.rio.to_raster(output_fns.krimp_output_fn)

# %%
