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
#%%
#load data
base_folder = Path(r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse\kartering")
input_folder = base_folder / 'input'
subsurface_model_fn = input_folder / 'base_data_shifted.nc'
sm = xr.open_dataset(subsurface_model_fn).chunk({'x': 1000, 'y': 1000, 'layer': -1})

impact_factor = xr.zeros_like(sm.lithology.isel(layer=0))+0.8

lithology_series = pd.Series({
    "anthropogenic": 0,
    "organic": 1,
    "clay": 2,
    "loam": 3,
    "fine_sand": 5,
    "medium_sand": 6,
    "coarse_sand": 7,
    "gravel": 8,
    "shells": 9,
    "other_sand": 11
})

doorlatend_enums = [0, 1, 5, 6, 7, 8, 9, 11]

nl_shape_fn = r"P:\gis-data\provincie\2021_provincies_zonder_water.shp"
nl_shape = gpd.read_file(nl_shape_fn)

result_folder = base_folder / "resultaten"

# sm['layer_tops'] = sm.thickness.cumsum(dim='layer') + sm.zbase
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

sm['shallow_doorlatend_mask'] = shallow_lith.isin(doorlatend_enums)
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

sm['deep_doorlatend_mask'] = deep_lith.isin(doorlatend_enums)
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
    # sm = sm.where((sm.layer_tops- sm.surface_level) > -5, drop=True)
    save_sm.to_netcdf(result_folder / "kartering_ondergrondklassen_criteria.nc",  encoding={v: {"zlib": True, "complevel": 4} for v in save_sm.data_vars})
    # keep_sm = sm[['layer_tops', 'layer_bottoms', 'lithology', 
    #               'shallow_doorlatend_mask','perc_shallow_doorlatend',
    #               'deep_doorlatend_mask', 'perc_deep_doorlatend']].compute()
    impact_factor = impact_factor.compute()

#mask, only on land
impact_factor.rio.write_crs("EPSG:28992", inplace=True)
impact_factor = impact_factor.rio.clip(nl_shape.geometry, nl_shape.crs, drop=True, invert=False, all_touched=True)
impact_factor.rio.to_raster(result_folder / "impact_factor_kem_teun.tif")



# %%
