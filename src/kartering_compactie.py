#%%
# load modules

import xarray as xr
from pathlib import Path
import numpy as np
import geopandas as gpd
from dask.diagnostics import ProgressBar
import utils

if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="kartering_compactie")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output
#%%
max_depth = -8
save_coarse = False

#load data
#ondergrond data
# Optimize chunking for better performance
sm = xr.open_dataset(input_fns.atlans_fn).chunk(chunks={"x": 500, "y": 500, 'layer': -1})
tot_thickness = sm.thickness.cumsum('layer').where(sm.thickness.notnull())
sm['tops_onder_mv'] = tot_thickness - tot_thickness.max('layer')
sm['bots_onder_mv'] = sm.tops_onder_mv - sm.thickness
sm['bots_onder_mv'] = xr.where(sm['bots_onder_mv'] > max_depth, sm['bots_onder_mv'], max_depth)
sm['thickness'] = sm['tops_onder_mv'] - sm['bots_onder_mv']
sm['thickness'] = sm['thickness'].where(sm['thickness'] > 0)
sm['center_onder_mv'] = sm['bots_onder_mv'] + sm['thickness'] / 2
pleistoceen_onder_mv = (sm.surface_level - sm.domainbase).fillna(0).compute()
sm['pleistoceen_mask'] = sm.center_onder_mv < pleistoceen_onder_mv
sm = sm.sel(layer=sm.layer[::-1])


#glg for now
##################
#Yearly LG3 or real GLG? Or even just lowest groundwater ever?
#TODO: MASK GROUNDWATER WITH LHM SURFACE WATER MASK
#EDIT: DIE BESTAAT NIET HAHA. HOE DAN? 

###################
# #ALS IK DE GLG GEBRUIK VAN DE LHM IS DIE HOGER DAN DE L1 TOP? HOE DAN?
# year = 2022
# lhm_folder = Path(r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse\kartering\lhm_data")
# gw_data_fn = lhm_folder / f"lhm_433_heads_l1_{year}.nc"
# top_fn = lhm_folder / 'TOP_L1_LHM433.tif'
# gw_stand_nap = xr.open_dataarray(gw_data_fn).chunk(x=200, y = 200, time = 50)
# glg_nap = utils.calc_avg_lowest_three(gw_stand_nap)
# top_l1_da = xr.open_dataarray(top_fn).isel(band=0).drop_vars('band')
# glg_stand_mv = glg_nap - top_l1_da
# with ProgressBar():
#     glg_stand_mv =  glg_stand_mv.interp(x=sm.x, y=sm.y, method='nearest').compute()

#DIT IS NIET GOED WAARSCHIJNLIJK!
##########################
sm['phreatic_level'] = sm.phreatic_level.where(sm.phreatic_level > -100)
sm['glg_mv'] = sm.phreatic_level - sm.surface_level
########################
# %%
#calculate compactie score

# #################
# #for testing
# testcol = [sm.x[1730], sm.y[817]]
# #sm_compact = sm_compact.sel(x=testcol[0], y=testcol[1], method = 'nearest').compute()
# sm = sm.sel(x=testcol[0], y=testcol[1], method = 'nearest').compute()
# glg = -sm.phreatic_level
###################


# Method 1: Using xarray's where() function for vectorized conditional mapping
# This is much more efficient and dask-friendly than apply_ufunc

print("Mapping lithology to specific weights using vectorized operations...")
# Create specific weight mapping using xarray.where() - much faster and dask-friendly
sw = xr.where(sm.lithology == 1, 0.05,   # organic
     xr.where(sm.lithology == 2, 0.5,    # clay
     xr.where(sm.lithology == 3, 0.7,    # loam
     xr.where(sm.lithology.isin([0, 5, 6, 7, 8, 9, 11]), 1, # antropogenic, sand, gravel, shells
              np.nan))))  # default for unknown values

print("Mapping lithology to compressibility using vectorized operations...")
# Create compressibility mapping using xarray.where()
cr = xr.where(sm.lithology == 1, 1.0,    # peat - highly compressible
    xr.where((sm.lithology == 2)&(~sm.pleistoceen_mask), 0.6,    # clay Holocene, 0.6
    xr.where((sm.lithology == 2)&(sm.pleistoceen_mask), 0.2,   # clay Pleistocene, 0.2
    xr.where(sm.lithology == 3, 0.2,    # loam
    xr.where(sm.lithology.isin([0, 5, 6, 7, 8, 9, 11]), 0.0,  # antropogenic, sand, gravel, shells
            np.nan)))))  # default for unknown values

print("Calculating effective stress...")
#effective stress calculation - optimized
thickness_sw = sm.thickness * sw
eff_stress = thickness_sw.cumsum('layer')
eff_stress = eff_stress - (thickness_sw * 0.5)  # More efficient than division

# limit to glg
glg_mask = (sm.center_onder_mv <= sm.glg_mv)

print("Calculating compaction score...")
#bereken score
score_per_layer = (cr / eff_stress)
score = score_per_layer.where(glg_mask).sum('layer')

sm['sw'] = sw
sm['cr'] = cr
sm['eff_stress'] = eff_stress
sm['layer_compactie_score'] = score_per_layer
sm['compactie_score'] = score.where(sm.lithology.notnull().any('layer'))

sm['compactie_score'] = sm['compactie_score'].where(sm['compactie_score'] > 0)
sm.rio.write_crs("EPSG:28992", inplace=True)

savecols = ['lithology', 'tops_onder_mv', 'bots_onder_mv', 'pleistoceen_mask',
             'glg_mv', 'eff_stress', 'cr', 'sw', 'layer_compactie_score', 'compactie_score']

#%%
print("Computing final result with progress bar...")
with ProgressBar():
    save_sm = sm[savecols]
    if params.coarsen > 1:
        coarse_factor = int(params.coarsen)
        save_sm = sm.isel(x = slice(None, None, coarse_factor),
                                y = slice(None, None, coarse_factor))  
        save_sm.to_netcdf(output_fns.compactie_data_fn,  encoding={v: {"zlib": True, "complevel": 4} for v in save_sm.data_vars})

    sm['compactie_score'].rio.to_raster(output_fns.compactie_output_fn)

# %%