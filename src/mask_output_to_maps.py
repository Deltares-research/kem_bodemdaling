#%%
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from scipy import ndimage
import utils
import xugrid as xu

#%%
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="mask_outputs")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output
#%%
#load data and masks
data_da = xr.open_dataarray(input_fns.data_fn).isel(band=0)
mask_da = xr.open_dataset(input_fns.comb_masks_fn)

#get relevant masks
mask_vars_as_dims = xr.concat([mask_da[m] for m in params.mask_categories], dim='mask_category')
comb_mask = ~mask_vars_as_dims.any(dim='mask_category')
comb_mask = comb_mask.sortby('y',ascending=False)

#mask data
masked_data_da = (data_da
                  .where(comb_mask)
                  .where(data_da > params.score_threshold))

masked_data_da.rio.write_crs("EPSG:28992", inplace=True)
masked_data_da.rio.to_raster(output_fns.masked_fn)



# %%
