#%%
import xarray as xr
from pathlib import Path
import numpy as np
import utils

#%%
if "snakemake" not in globals():
    replace_dict = {'{year}': '2015'}
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="combine_kartering")
    snakemake = utils.replace_wildcards_in_snakemake(snakemake, replace_dict)

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

#new combination, 
# 1 if oxidatie dominant
# 2 if compactie dominant
# 3 if krimp dominant
# 4 if oxidatie and compactie equal dominant
# 5 if oxidatie and krimp equal dominant
# 6 if compactie and krimp equal dominant
# 7 if all equal dominant

# Create masks for each condition (>= 3)
oxidatie_dominant = oxidatie_grpd >= 3
compactie_dominant = compactie_grpd >= 3
krimp_dominant = krimp_grpd >= 3

# Assign scores based on dominance patterns
combined = xr.where(
    oxidatie_dominant & compactie_dominant & krimp_dominant, 7,  # all equal dominant
    xr.where(
        oxidatie_dominant & compactie_dominant & ~krimp_dominant, 4,  # oxidatie and compactie equal dominant
        xr.where(
            oxidatie_dominant & krimp_dominant & ~compactie_dominant, 5,  # oxidatie and krimp equal dominant
            xr.where(
                compactie_dominant & krimp_dominant & ~oxidatie_dominant, 6,  # compactie and krimp equal dominant
                xr.where(
                    oxidatie_dominant & ~krimp_dominant & ~compactie_dominant, 1,  # oxidatie dominant
                    xr.where(
                        compactie_dominant & ~krimp_dominant & ~oxidatie_dominant, 2,  # compactie dominant
                        xr.where(krimp_dominant & ~oxidatie_dominant & ~compactie_dominant, 3, 0)  # krimp dominant, else 0
                    )
                )
            )
        )
    )
)

combined = combined.isel(band=0)
combined = combined.where(combined>0)
combined = combined.rio.write_crs(28992)
combined.rio.to_raster(output_fns.comb_kartering_output)
#%% 
## old combination, sum each group
# total_weight = np.prod(list(weights.values()))


# combined = (oxidatie_grpd*weights['oxidatie'] + 
#             compactie_grpd * weights['compactie'] +
#             krimp_grpd * weights['krimp']) / total_weight

# combined = combined.isel(band=0)
# combined = combined.where(combined>0)
# combined = combined.rio.write_crs(28992)
# combined.rio.to_raster(output_fns.comb_kartering_output)
# %%
