#%%
# load modules

import xarray as xr
from pathlib import Path
import numpy as np
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
sm = xr.open_dataset(input_fns.atlans_fn).chunk(chunks={"x": 100, "y": 100, 'layer': -1})
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

#glg uit LHM (is in meter onder maaiveld, dus positief = onder maaiveld)
glg_da = xr.open_dataarray(input_fns.lhm_glg_fn).chunk(x=500, y = 500)
sm['glg_mv'] = glg_da.interp(x=sm.x, y=sm.y, method='nearest').compute()
sm['glg_mv'] = -sm['glg_mv']  #omzetten naar onder maaiveld (negatief onder maaiveld)
# %%
#calculate compactie score

# #################
# #for testing
# testcol = [sm.x[1730], sm.y[817]]
# #sm_compact = sm_compact.sel(x=testcol[0], y=testcol[1], method = 'nearest').compute()
# sm = sm.sel(x=testcol[0], y=testcol[1], method = 'nearest').compute()
# glg = -sm.phreatic_level
###################

###########################
#for testing layer number effect
###################################
# test_da = xr.DataArray([[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,np.nan,np.nan,np.nan,np.nan,np.nan]],
#                       coords={'x': [1000,1001], 'layer': np.arange(10)})
# test_sm = xr.Dataset({'lithology': test_da})
# test_thickness_data = np.array([np.ones(10)*0.1, np.r_[np.ones(5)*0.2, np.zeros(5)]])
# test_sm['thickness'] = xr.DataArray(test_thickness_data, coords={'x': [1000,1001], 'layer': np.arange(10)})
# test_sm['tops_onder_mv'] = -test_sm.thickness.cumsum('layer')
# test_sm['center_onder_mv'] = -test_sm.thickness.cumsum('layer') + (test_sm.thickness / 2)
# test_sm['pleistoceen_mask'] = True
# test_sm['glg_mv'] = -0.1

####################

dz = 0.1
n_layers = int(np.ceil((0 - max_depth) / dz))
new_thickness = xr.DataArray(np.repeat(dz, n_layers),
                             coords={'layer': np.arange(n_layers)+1})
new_sm = xr.Dataset({'thickness': new_thickness})
new_sm['tops_onder_mv'] = -new_sm.thickness.cumsum('layer')+ new_sm.thickness
new_sm['bots_onder_mv'] = new_sm.tops_onder_mv 
new_sm['center_onder_mv'] = new_sm.bots_onder_mv + (new_sm.thickness / 2)
new_sm = new_sm.broadcast_like(sm.isel(layer=0))
new_sm['lithology'] = xr.full_like(new_sm.thickness, np.nan)
new_sm['pleistoceen_mask'] = xr.full_like(new_sm.thickness, False, dtype=bool)
new_sm = new_sm.chunk(chunks={"x": 100, "y": 100, 'layer': -1})

for layer in new_sm.layer:
    l_top = new_sm.tops_onder_mv.sel(layer=layer)
    l_bot = new_sm.bots_onder_mv.sel(layer=layer)
    z = (l_top+l_bot).drop_vars('layer')/2
    
    l_sm = sm.where((z<sm.tops_onder_mv) & (z>=sm.bots_onder_mv)).max('layer')

    new_sm[['lithology', 'pleistoceen_mask']] = xr.where(new_sm.layer==layer, 
                                                        l_sm[['lithology', 'pleistoceen_mask']], 
                                                        new_sm[['lithology', 'pleistoceen_mask']])
new_sm[['lithology', 'pleistoceen_mask']] = new_sm[['lithology', 'pleistoceen_mask']].transpose('y', 'x', 'layer')
new_sm['glg_mv'] = sm['glg_mv']
new_sm['pleistoceen_mask'] = new_sm['pleistoceen_mask'].astype(bool)
#%%
sm = new_sm.copy()
print('starting compactie calculation...')
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
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! heb ik hier nu wat aan gedaan
#DAT DE SCORE GENORMALISEERD IS PER CM EN NIET PER LAAG?
##############################################
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