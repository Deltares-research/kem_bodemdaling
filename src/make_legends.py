#%%
from pathlib import Path
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from snakemake.io import Namedlist

import utils

if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(utils.SNAKEFILE_PATH, name="make_legends")

input_fns = snakemake.input
output_fns = snakemake.output

#%%
print(type(input_fns.input_map))
if type(input_fns.input_map) is str:
    #input_map = r"P:\nl2120veen\kartering\resultaten\masked_maps\oxidatie_2012_score_masked.tif"
    input_data = xr.open_dataarray(input_fns.input_map)
elif type(input_fns.input_map) is Namedlist:
    data_arrays = [xr.open_dataarray(f) for f in input_fns.input_map]
    input_data = xr.concat(data_arrays, dim='year')

quantiles = [0.2, 0.4, 0.6, 0.8, 1.0]
quantile_values= input_data.quantile(quantiles).to_dataframe()

cmap = plt.get_cmap("viridis", len(quantiles))
cmap_colors = [tuple(int(255*c) for c in cmap(i)[:4]) for i in range(cmap.N)]

labels = ["Zeer laag", "Laag", "Gemiddeld", "Hoog", "Zeer hoog"]

utils.write_qgis_legend(
    filename=output_fns.legend_fn,
    x_values=quantile_values[input_data.name].tolist(),
    colors=cmap_colors,
    labels=labels,
    interpolation="DISCRETE"
)   

# %%
