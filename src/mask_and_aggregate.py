#%%
import xarray as xr
import geopandas as gpd
import numpy as np
import utils
import xugrid as xu
from pathlib import Path
import dask_geopandas as dgpd
from dask.diagnostics import ProgressBar

def weighted_mean(x, data_col, weight_col):
    v = x[data_col]
    w = x[weight_col]
    mask = ~v.isna()  # exclude NaNs
    return (v[mask] * w[mask]).sum() / w[mask].sum()

def most_frequent_by_weight(x, data_col, weight_col):
    # Group by data values and sum weights
    if x[data_col].isnull().all():
        return np.nan
    weighted_sums = x.groupby(data_col)[weight_col].sum()
    return weighted_sums.idxmax()
#%%
if "snakemake" not in globals():
    replace_dict = {'{type}': 'kem'}
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="mask_aggregate_data")
    snakemake = utils.replace_wildcards_in_snakemake(snakemake, replace_dict)

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output
#%%
# Load data and masks
data_da = xr.open_dataarray(input_fns.data_fn).isel(band=0)
drainage_gdf = gpd.read_file(input_fns.drainage_fn)

# Load peilgebieden (water level areas) and triangulate for ugrid
peilgebieden_gdf = gpd.read_file(input_fns.peilgebieden_fn).explode()[['geometry']]
#%%

#peilgeieden to du
peilgebieden_du = xu.earcut_triangulate_polygons(peilgebieden_gdf)

masked_data_du = xu.UgridDataArray.from_structured2d(data_da)
# Remove cells with no valid data
masked_data_du = masked_data_du.dropna('mesh2d_nFaces')

# Create overlap regridder to map data to peilgebieden
overlap_regr = xu.OverlapRegridder(masked_data_du, peilgebieden_du, method = 'max_overlap')
weights = overlap_regr.weights_as_dataframe()
weights.rename(columns={'source_index':'data_idx', 'target_index':'peil_idx'}, inplace=True)
weights['ori_peil_idx'] = peilgebieden_du.values[weights['peil_idx'].values]
weights['value'] = masked_data_du.values[weights['data_idx'].values]

#get fraction of data per peilgebied
data_in_peil_area = weights.groupby('ori_peil_idx')
rel_data_in_peil_area = data_in_peil_area['weight'].sum() / peilgebieden_gdf.area
peilgebieden_gdf['frac_data_in_peil'] = rel_data_in_peil_area

#get aggregated value per peilgebied
if params.agg_method == 'weighted_average':
    peilgebieden_gdf['agg_value'] = data_in_peil_area.apply(lambda grp: weighted_mean(grp, 'value', 'weight'))
elif params.agg_method == 'most_frequent_by_weight':
    peilgebieden_gdf['agg_value'] = data_in_peil_area.apply(lambda grp: most_frequent_by_weight(grp, 'value', 'weight'))
else:
    raise ValueError(f"Unknown aggregation method: {params.agg_method}")

#filter peilgebieden with sufficient data
sufficient_data_peilgebieden = peilgebieden_gdf[peilgebieden_gdf['frac_data_in_peil'] >= params.min_data_fraction]
sufficient_data_peilgebieden = sufficient_data_peilgebieden[['geometry', 'agg_value']]
sufficient_data_peilgebieden.to_file(str(output_fns.agg_output))

if 'kem' in snakemake.wildcards.type:
    drainage_gdf = dgpd.from_geopandas(drainage_gdf, npartitions=50)

    drainage_in_peil = drainage_gdf.clip(sufficient_data_peilgebieden)
    with ProgressBar():
        drainage_in_peil = drainage_in_peil.compute()
    drainage_in_peil.to_file(Path(output_fns.agg_output).parent / 'drainage_in_peil.gpkg')




# %%
