#%%
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
import utils
import xugrid as xu
import dask_geopandas
from dask.diagnostics import ProgressBar
import imod

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
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="mask_aggregate_kem")

input_fns = snakemake.input
output_fns = snakemake.output
#%%
# Load data and masks
data_da = xr.open_dataarray(input_fns.data_fn).isel(band=0)
mask_da = xr.open_dataset(input_fns.comb_masks_fn)
drainage_gdf = gpd.read_file(input_fns.drainage_fn)

# Load peilgebieden (water level areas) and triangulate for ugrid
peilgebieden_gdf = gpd.read_file(input_fns.peilgebieden_fn).explode()[['geometry']]

#%%
# Get relevant mask categories and create combined mask
mask_categories = ['opnemend', 'bebouwing', 'water', 'nl']
mask_vars_as_dims = xr.concat([mask_da[m] for m in mask_categories], dim='mask_category')
# Combined mask excludes areas where ANY mask category is True
comb_mask = ~mask_vars_as_dims.any(dim='mask_category')
comb_mask = comb_mask.sortby('y', ascending=False)

# Apply mask to data and convert to unstructured grid
masked_data_da = data_da.where(comb_mask)

with ProgressBar():
    test_overlap = imod.prepare.zonal_aggregate_raster(input_fns.peilgebieden_fn,
                                                    'code',
                                                    masked_data_da,
                                                    resolution=1.0,
                                                    method = lambda x: x.mode().iloc[0],
                                                    chunksize = 500)


peilgebieden_du = xu.earcut_triangulate_polygons(peilgebieden_gdf)


# masked_data_gdf = utils.data_to_geodataframe(masked_data_da)

# dask_drainage_gdf = dask_geopandas.from_geopandas(drainage_gdf, npartitions=10)
# dask_peilgebieden_gdf = dask_geopandas.from_geopandas(peilgebieden_gdf, npartitions=10)
# dask_masked_data_gdf = dask_geopandas.from_geopandas(masked_data_gdf, npartitions=10)

# masked_data_gdf.to_file(r"P:/nl2120veen/kartering/resultaten/masked_kem_temp.gpkg")

# kem_peilgebieden = []
# for v, kem_data in dask_masked_data_gdf.groupby('value'):
#     kem_per_peilgebied = dask_peilgebieden_gdf.clip(kem_data.geometry)
#     kem_per_peilgebied['value'] = v
#     with ProgressBar():
#         kem_per_peilgebied = kem_per_peilgebied.compute()
#     kem_peilgebieden.append(kem_per_peilgebied)


























#%%
peilgebieden_du = peilgebieden_du.assign_coords({'mesh2d_nFaces': ('mesh2d_nFaces', peilgebieden_du.values)})

masked_data_du = xu.UgridDataArray.from_structured2d(masked_data_da)
# Remove cells with no valid data
masked_data_du = masked_data_du.dropna('mesh2d_nFaces')

# Create overlap regridder to map data to peilgebieden
overlap_regr = xu.OverlapRegridder(masked_data_du, peilgebieden_du, method = 'max_overlap')
weights = overlap_regr.weights_as_dataframe()
weights_source = weights.groupby('target_index').sum()

# Merge weights with peilgebieden geometry information
dummy_df = peilgebieden_du.to_dataframe(name='dummy').reset_index()
overlap = pd.merge(weights_source, dummy_df, left_on='target_index', right_index=True, how='right')

# Create area per cell as UgridDataArray
area_per_cell = (type(peilgebieden_du)).from_series(overlap.set_index(list(peilgebieden_du.dims))['weight'])

# Create proper UgridDataArray with grid and data
temp_du = xu.UgridDataArray(area_per_cell, 
                         grid=peilgebieden_du.grid)

# Convert to GeoDataFrame and add regridded data
temp_gdf = temp_du.ugrid.to_geodataframe()
temp_gdf['nr'] = peilgebieden_du.values
# temp_gdf['data'] = overlap_regr.regrid(masked_data_du).to_dataframe()['band_data']

# Calculate statistics per peilgebied
peilgebieden_gdf['data_area'] = temp_gdf.groupby('nr')['weight'].sum().values
peilgebieden_gdf['data_fraction'] = peilgebieden_gdf['data_area'] / peilgebieden_gdf.geometry.area



#peilgebieden_gdf['data'] = temp_gdf.groupby('nr').apply(lambda grp: most_frequent_by_weight(grp, 'data', 'weight')).values

# Only keep data where more than 50% of peilgebied has valid data
peilgebieden_gdf['data'] = peilgebieden_gdf['data'].where(peilgebieden_gdf['data_fraction'] > 0.5)

##%%
dask_peilgebieden_gdf = dask_geopandas.from_geopandas(peilgebieden_gdf, npartitions=10)
dask_drainage_gdf = dask_geopandas.from_geopandas(drainage_gdf, npartitions=10)

result = dask_geopandas.sjoin(dask_peilgebieden_gdf, dask_drainage_gdf, how="inner", predicate="intersects")

with ProgressBar():
    result = result.compute()

peil_geos = result['geometry'].reset_index(drop=True)
drainage_geos = drainage_gdf.loc[result['index_right'], 'geometry'].reset_index(drop=True)

new_geos = peil_geos.intersection(drainage_geos)
new_data = gpd.GeoDataFrame(data = [333]*len(new_geos), 
                            geometry = new_geos, 
                            crs=peilgebieden_gdf.crs,
                            columns = ['data'])

old_geos = peil_geos.difference(drainage_geos)
old_data = gpd.GeoDataFrame(data = result['data'], 
                            geometry = old_geos, 
                            crs=peilgebieden_gdf.crs)

non_overlapping = peilgebieden_gdf[~peilgebieden_gdf.index.isin(result.index.unique())]

##%%
final_kem_gdf = pd.concat([new_data, old_data, non_overlapping], ignore_index=True)
final_kem_gdf = final_kem_gdf[['geometry', 'data']]
final_kem_gdf.to_file(str(output_fns.kem_agg_output))


#%%
#Add drainage mask to data
# Use spatial index for faster intersection
drainage_sindex = drainage_gdf.sindex
peilgebieden_gdf['has_drainage'] = False
peilgebieden_gdf['data_with_drainage'] = peilgebieden_gdf['data'].copy()

# Create new GeoDataFrame for drainage areas
drainage_areas = []

for idx, peil_geom in tqdm(peilgebieden_gdf['geometry'].items(), total = len(peilgebieden_gdf)):
    possible_matches_index = list(drainage_sindex.intersection(peil_geom.bounds))
    possible_matches = drainage_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(peil_geom)]
    if not precise_matches.empty:
        peilgebieden_gdf.loc[idx, 'has_drainage'] = True
        peilgebieden_gdf.loc[idx, 'data_with_drainage'] = 'drainage'
        # Store the intersecting drainage geometries
        intersecting_geoms = precise_matches.geometry.union_all()
        drainage_areas.append({
            'peilgebied_idx': idx,
            'geometry': intersecting_geoms
        })

# Create new GeoDataFrame for drainage geometries
drainage_gdf_result = gpd.GeoDataFrame(drainage_areas, crs=peilgebieden_gdf.crs)


# Spatial index
sindex = drainage_gdf.sindex
possible_matches_index = peilgebieden_gdf.geometry.apply(lambda x: list(sindex.intersection(x.bounds)))
gdf2_filtered = drainage_gdf.iloc[[i for sublist in possible_matches_index for i in sublist]]

# Then overlay
result = gpd.overlay(peilgebieden_gdf, gdf2_filtered, how="intersection")

# %%
