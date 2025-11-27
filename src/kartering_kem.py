#%%
#load modules
import xarray as xr
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
import rioxarray 
import geopandas as gpd
from pathlib import Path
import utils

if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="kartering_kem")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output

#%%
#load data
sm = xr.open_dataset(input_fns.base_data_shifted_fn).chunk({'x': 1000, 'y': 1000, 'layer': -1})

impact_factor = xr.zeros_like(sm.lithology.isel(layer=0))+0.8

sm['bots_onder_mv'] = sm.tops_onder_mv - sm.thickness

# x = 185350
# y = 316450
# sm = sm.sel(x=x, y=y).compute()
#%%
#ondiep
top_selection_depth = -0.5
bottom_selection_depth = -1.5

shallow_tops, shallow_thickness, shallow_lith = utils.select_layers(
    lith_col=sm.lithology,
    layer_bottoms=sm.bots_onder_mv,
    layer_tops=sm.tops_onder_mv,
    startdepth=top_selection_depth,
    enddepth=bottom_selection_depth
)

sm['shallow_doorlatend_mask'] = shallow_lith.isin(params.doorlatend_enums)
sm['perc_shallow_doorlatend']  = (shallow_thickness.where(sm['shallow_doorlatend_mask']).sum(dim='layer') / shallow_thickness.sum(dim='layer'))

shallow_doorlatend_threshold = 0.8
shallow_doorlatend_verdict = sm['perc_shallow_doorlatend']  >= shallow_doorlatend_threshold

#diep
top_selection_depth = -1.5
bottom_selection_depth = -2.5

deep_tops, deep_thickness, deep_lith = utils.select_layers(
    lith_col=sm.lithology,
    layer_bottoms=sm.bots_onder_mv,
    layer_tops=sm.tops_onder_mv,
    startdepth=top_selection_depth,
    enddepth=bottom_selection_depth
)

sm['deep_doorlatend_mask'] = deep_lith.isin(params.doorlatend_enums)
sm['perc_deep_doorlatend'] = (deep_thickness.where(sm['deep_doorlatend_mask']).sum(dim='layer') / deep_thickness.sum(dim='layer'))

deep_doorlatend_threshold = 0.5
deep_doorlatend_verdict = sm['perc_deep_doorlatend'] >= deep_doorlatend_threshold

#combine
#als ondiep wel en diep niet doorlatend, categorie C
impact_factor = xr.where(shallow_doorlatend_verdict & ~deep_doorlatend_verdict, 9999.0, impact_factor)

#als ondiep en diep doorlatend, categorie A
impact_factor = xr.where(~shallow_doorlatend_verdict & ~deep_doorlatend_verdict, 0.2, impact_factor)
# %%
savecols= ['tops_onder_mv', 'bots_onder_mv', 'lithology', 
           'shallow_doorlatend_mask','perc_shallow_doorlatend',
              'deep_doorlatend_mask', 'perc_deep_doorlatend']
save_sm = sm[savecols]

#do calculation
with ProgressBar():
    if params.save:
        if params.coarsen > 1:
            coarse_factor = int(params.coarsen)
            save_sm = save_sm.isel(x = slice(None, None, coarse_factor),
                                   y = slice(None, None, coarse_factor))  
        save_sm.to_netcdf(output_fns.kem_data_fn,  encoding={v: {"zlib": True, "complevel": 4} for v in save_sm.data_vars})

    #impact_factor = impact_factor.compute()

#mask, only on land
impact_factor.rio.write_crs("EPSG:28992", inplace=True)
impact_factor.rio.to_raster(output_fns.kem_output_fn)



# %%
