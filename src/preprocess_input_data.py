#%%
import xarray as xr
import geopandas as gpd
import xugrid as xu
import pandas as pd

import utils
#%%
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="preprocess_input")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output

#%%
#load basedata
dummy_da = xr.open_dataset(input_fns.atlans_fn).surface_level

##%%
#co2 opnemende natuur; die er uit filteren

da = xr.open_dataarray(input_fns.uitstotend_fn).chunk(x=1000,y=1000)
opnemend_vars = [d.item() for d in da.layer if "opnemend" in d.item()]
opnemend = da.sel(layer=opnemend_vars).max(dim="layer")
regridder = xu.OverlapRegridder(opnemend, dummy_da)
opnemend_regridded = regridder.regrid(opnemend) 
opnemend_regrid_mask = (opnemend_regridded!=0).astype(int).compute()
opnemend_regrid_mask.rio.write_crs(28992, inplace=True)
#opnemend_regrid_mask.rio.to_raster(uitsluiting_folder / "opnemend_mask.tif")

##%% 
#bebouwd gebied waar bebouwdeko notnull
gdf = gpd.read_file(input_fns.novex_fn)
huidige_bebouwing = gdf[gdf.bebouwdeko.notnull()]
bebouwing_pols = huidige_bebouwing.geometry.explode()

bebouwing_fraction = utils.calculate_fraction_of_gdf(bebouwing_pols, dummy_da)

#huidige_bebouwing.geometry.to_file(uitsluiting_folder  / "huidige_bebouwing.shp")
bebouwing_fraction.rio.write_crs(28992, inplace=True)
#bebouwing_fraction.rio.to_raster(uitsluiting_folder / "bebouwing_fraction.tif")
##%% 
#bgt waterdeel, alleen grotere wateren
watervlakte = gpd.read_file(input_fns.watervlakte_fn)
watervlakte = watervlakte.simplify(tolerance=10)

fraction_water_per_cell = utils.calculate_fraction_of_gdf(watervlakte, dummy_da)
#fraction_water_per_cell.rio.to_raster(uitsluiting_folder / "water_fraction.tif")

##%%
#peilgestuurde gebieden
peilgestuurd_gdf = gpd.read_file(input_fns.peilgestuurd_fn)

peilgestuurd_mask = utils.rasterize_gdf_to_match(peilgestuurd_gdf, dummy_da)
peilgestuurd_mask.rio.write_crs(28992, inplace=True)
#peilgestuurd_mask.rio.to_raster(uitsluiting_folder / "peilgestuurd_mask.tif")

##%%
#nl shape
nl_gdf = gpd.read_file(input_fns.nl_shape_fn)
nl_mask = utils.rasterize_gdf_to_match(nl_gdf, dummy_da)

##%% combine masks
bebouwing_fraction_threshold = 0.5
water_fraction_threshold = 0.5
bebouwing_mask = (bebouwing_fraction < bebouwing_fraction_threshold) | (
                  bebouwing_fraction.notnull())
mask_dict = xr.Dataset({
    'opnemend': opnemend_regrid_mask.astype(bool),
    'bebouwing': bebouwing_mask,
    'water': fraction_water_per_cell < water_fraction_threshold,
    'peilgestuurd': ~peilgestuurd_mask.astype(bool),
    'nl': ~nl_mask.astype(bool)
})
mask_dict.to_netcdf(output_fns.comb_masks_fn)

# ## peilgebieden
# peilgebied_fn = r"P:\nl2120veen\kartering\input\peilgebied.gpkg"
# peilgebied_gdf = gpd.read_file(peilgebied_fn)

# t = peilgebied_gdf.statuspeilgebied.value_counts()

# keep = ['Vigerend definitief', 'Praktijk']

# cleaned_peilgebied = peilgebied_gdf[~peilgebied_gdf.code.astype(str).str.endswith('-P')]
# cleaned_peilgebied = cleaned_peilgebied[cleaned_peilgebied.statuspeilgebied.isin(keep)|
#                                         cleaned_peilgebied.code.isnull()]
# cleaned_peilgebied = cleaned_peilgebied.loc[cleaned_peilgebied.code.notnull()].groupby('code').first().reset_index()
# # Keep original rows where code is null and append them
# null_code_rows = peilgebied_gdf[peilgebied_gdf.code.isnull()]
# cleaned_peilgebied = pd.concat([cleaned_peilgebied, null_code_rows], ignore_index=True)
# cleaned_peilgebied.to_file(r"P:\nl2120veen\kartering\input\peilgebied_cleaned.gpkg")
#%%
# n_partitions = 150

# #nl shape - start with total shape, partitioned for parallel processing
# nl_gdf = gpd.read_file(input_fns.nl_shape_fn)
# dask_nl_shape = dask_geopandas.from_geopandas(nl_gdf, npartitions=n_partitions)
# # Parallel union operation
# dask_nl_union = dask_nl_shape.dissolve().geometry

# ##%%
# #co2 opnemende natuur - parallel processing with dask
# da = xr.open_dataarray(input_fns.uitstotend_fn).chunk(x=1000,y=1000)
# opnemend_vars = [d.item() for d in da.layer if "opnemend" in d.item()]
# opnemend = da.sel(layer=opnemend_vars).max(dim="layer")
# opnemend_gdf = utils.mask_to_geodataframe(opnemend!=0)
# # Partition and dissolve in parallel
# dask_opnemend = dask_geopandas.from_geopandas(opnemend_gdf, npartitions=n_partitions)
# dask_opnemend_union = dask_opnemend.dissolve().geometry

# #bebouwd gebied - parallel dissolve
# gdf = gpd.read_file(input_fns.novex_fn)
# huidige_bebouwing = gdf[gdf.bebouwdeko.notnull()]
# # Partition by spatial proximity for efficient parallel processing
# dask_bebouwing = dask_geopandas.from_geopandas(huidige_bebouwing, npartitions=n_partitions)
# dask_bebouwing_union = dask_bebouwing.dissolve().geometry

# #water - parallel processing with simplification
# watervlakte = gpd.read_file(input_fns.watervlakte_fn)
# # Simplify first, then partition and dissolve in parallel
# watervlakte_simple = watervlakte.assign(geometry=watervlakte.geometry.simplify(tolerance=10))
# dask_watervlakte = dask_geopandas.from_geopandas(watervlakte_simple, npartitions=n_partitions)
# dask_water_union = dask_watervlakte.dissolve().geometry

# # #peilgestuurde gebieden - parallel dissolve
# # peilgestuurd_gdf = gpd.read_file(input_fns.peilgestuurd_fn)
# # dask_peilgestuurd = dask_geopandas.from_geopandas(peilgestuurd_gdf, npartitions=n_partitions)
# # dask_peil_union = dask_peilgestuurd.dissolve().geometry

# # Chain difference operations using dask - all computed in parallel
# final_mask = (dask_nl_union
#               .difference(dask_opnemend_union)
#               .difference(dask_bebouwing_union)
#               .difference(dask_water_union))

# # Compute the final result with all parallel operations
# with ProgressBar():
#     final_mask = final_mask.compute()

#     # Save final combined mask
#     final_mask.to_file(output_fns.comb_masks_fn)

#%%

# %%
