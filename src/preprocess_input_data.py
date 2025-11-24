#%%
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
import xugrid as xu
from rasterio.features import rasterize

input_folder = Path(r"P:\nl2120veen\kartering\input")
uitsluiting_folder = input_folder.parent / 'uitsluitingsgebieden'

#%%
def rasterize_gdf_to_match(gdf, template_da, value=1, fill=0, dtype=np.uint8):
    """Rasterize a GeoDataFrame to match the spatial grid of a template DataArray."""
    transform = template_da.rio.transform()
    shape = (template_da.sizes['y'], template_da.sizes['x'])
    
    raster_ar = rasterize(
        ((geom, value) for geom in gdf.geometry),
        out_shape=shape,
        transform=transform,
        fill=fill,
        dtype=dtype
    )
    
    return xr.DataArray(
        raster_ar,
        dims=('y', 'x'),
        coords={'y': template_da.y, 'x': template_da.x}
    )

def calculate_fraction_of_gdf(input_gdf, template_da):
    """Calculate fraction of coverage per cell in template grid."""
    watervlakte_du = xu.earcut_triangulate_polygons(input_gdf.to_frame())
    rel_overlap_regr = xu.RelativeOverlapRegridder(template_da, watervlakte_du)
    weights = rel_overlap_regr.weights_as_dataframe()
    weights_source = weights.groupby('source_index').sum()
    weights_source['weight'] = np.minimum(weights_source['weight'], 1.0)

    dummy_df = template_da.to_dataframe(name='dummy').reset_index()
    overlap = pd.merge(weights_source, dummy_df, left_on='source_index', right_index=True, how='right')

    fraction_water_per_cell = xr.DataArray.from_series(overlap.set_index(['y','x'])['weight'])
    fraction_water_per_cell.rio.set_crs(28992, inplace=True)
    
    return fraction_water_per_cell
#%%
#load basedata
atlans_fn = input_folder / "atlans_subsurface_model_Knaake.nc"
dummy_da = xr.open_dataset(atlans_fn).surface_level

#%%
#co2 opnemende natuur; die er uit filteren
uitstotend_fn = input_folder / "nbp_coverage.nc"

da = xr.open_dataarray(uitstotend_fn).chunk(x=1000,y=1000)
opnemend_vars = [d.item() for d in da.layer if "opnemend" in d.item()]
opnemend = da.sel(layer=opnemend_vars).max(dim="layer")
regridder = xu.OverlapRegridder(opnemend, dummy_da)
opnemend_regridded = regridder.regrid(opnemend) 
opnemend_regrid_mask = (opnemend_regridded!=0).astype(int)
opnemend_regrid_mask.rio.set_crs(28992, inplace=True)
opnemend_regrid_mask.rio.to_raster(uitsluiting_folder / "opnemend_mask.tif")

#%% 
#bebouwd gebied waar inwoners
novex_fn = uitsluiting_folder / 'woningbouw_NOVEX' / "TOP10NL_NOVEX_combined.shp"
gdf = gpd.read_file(novex_fn)
huidige_bebouwing = gdf[gdf.bebouwdeko.notnull()]
bebouwing_pols = huidige_bebouwing.geometry.explode()

bebouwing_fraction = calculate_fraction_of_gdf(bebouwing_pols, dummy_da)

huidige_bebouwing.geometry.to_file(uitsluiting_folder  / "huidige_bebouwing.shp")
bebouwing_fraction.rio.set_crs(28992, inplace=True)
bebouwing_fraction.rio.to_raster(uitsluiting_folder / "bebouwing_fraction.tif")
#%% 
#bgt waterdeel, alleen grotere wateren
watervlakte_fn = input_folder / 'watervlakte.gpkg'
watervlakte = gpd.read_file(watervlakte_fn)
watervlakte = watervlakte.simplify(tolerance=10)

fraction_water_per_cell = calculate_fraction_of_gdf(watervlakte, dummy_da)
fraction_water_per_cell.rio.to_raster(uitsluiting_folder / "water_fraction.tif")

# %%
#peilgestuurde gebieden
peilgestuurd_fn = uitsluiting_folder / 'peilgestuurd.shp'
peilgestuurd_gdf = gpd.read_file(peilgestuurd_fn)

peilgestuurd_mask = rasterize_gdf_to_match(peilgestuurd_gdf, dummy_da)
peilgestuurd_mask.rio.set_crs(28992, inplace=True)
peilgestuurd_mask.rio.to_raster(uitsluiting_folder / "peilgestuurd_mask.tif")
