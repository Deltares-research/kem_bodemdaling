#%%
import xarray as xr
from pathlib import Path
import numpy as np
import utils

#%%
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="combine_kartering")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output

weights = params.factor_weights
#%%
oxidatie = xr.open_dataarray(input_fns.oxidatie_fn)
compactie = xr.open_dataarray(input_fns.compactie_fn)
krimp = xr.open_dataarray(input_fns.krimp_fn)

legends = {Path(fn).stem.split('_')[0]: utils.read_qgis_raster_legend(fn)[0] for fn in input_fns.legend_fns}

# Group data using legend values via searchsorted
oxidatie_grpd = xr.apply_ufunc(
    lambda x: np.searchsorted(legends['oxidatie'], x, side='right'),
    oxidatie,
    dask='allowed'
)
oxidatie_grpd = oxidatie_grpd.where(oxidatie.notnull()).fillna(0)

compactie_grpd = xr.apply_ufunc(
    lambda x: np.searchsorted(legends['compactie'], x, side='right'),
    compactie,
    dask='allowed'
)
compactie_grpd = compactie_grpd.where(compactie.notnull()).fillna(0)

krimp_grpd = xr.apply_ufunc(
    lambda x: np.searchsorted(legends['krimp'], x, side='right'),
    krimp,
    dask='allowed'
)
krimp_grpd = krimp_grpd.where(krimp.notnull()).fillna(0)


total_weight = np.prod(list(weights.values()))


combined = (oxidatie_grpd*weights['oxidatie'] + 
            compactie_grpd * weights['compactie'] +
            krimp_grpd * weights['krimp']) / total_weight

combined = combined.isel(band=0)
combined = combined.where(combined>0)
combined = combined.rio.write_crs(28992)
combined.rio.to_raster(output_fns.comb_kartering_output)
# %%
