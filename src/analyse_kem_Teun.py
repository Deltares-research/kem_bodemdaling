#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import utils
from dask.diagnostics import ProgressBar

input_fn = r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse\kartering\resultaten\kartering_ondergrondklassen_criteria.nc"
input = xr.open_dataset(input_fn).chunk({'x': 1000, 'y': 1000, 'layer': -1})

output_fn = r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse\kartering\resultaten\impact_factor_kem_teun.tif"
output = xr.open_dataarray(output_fn).squeeze('band', drop=True).compute()

lithoklasse_colors = {
    0: "#C0C0C0",  # Grijs (a)
    1: "#8B4513",  # SaddleBrown (v)
    2: "#008000",# Groenachtig (k)
    3: "#6CA36C",  # Minder fel groen (kz)
    5: "#FFFF00",  # Geel (zf)
    6: "#FFD700",  # Goud (zm)
    7: "#FFA500",  # Oranje (zg)
    8: "#DAA520",  # GoldenRod (g)
    9: "#0000FF",   # Blauw (she)
    11:"#FFD700",  # Goud (zm)
}

doorlatend_enums = [0, 1, 5, 6, 7, 8, 9, 11]

hatch_dict = {True: '///', False: None}

def make_title(input_ps, i):
    return f'Point {i+1}\nx={int(input_ps.x[i])}\ny={int(input_ps.y[i])}\nOndiep: {input_ps.perc_shallow_doorlatend.isel(points=i).item():.2f}\nDiep: {input_ps.perc_deep_doorlatend.isel(points=i).item():.2f}'


# %%
num_samples_per_value = 100 # Set the number of sample points per unique value
rows, cols = utils.best_grid(num_samples_per_value)
size_factor = 3
figsize= (rows*size_factor, cols*size_factor)

coords_dict = utils.sample_coordinates_by_value(output, num_samples_per_value)

for val, coords in coords_dict.items():
    if not coords:
        continue
    # if not val == 0.2:
    #     continue
    #coords = [[151450, 410950]]
    x = xr.DataArray([c[0] for c in coords], dims="points")
    y = xr.DataArray([c[1] for c in coords], dims="points")
    with ProgressBar():
        input_ps = input.sel(x=x, y=y, method = 'nearest').compute()

    doorlatend = input_ps.shallow_doorlatend_mask | input_ps.deep_doorlatend_mask
    ######################################

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True)
    flatax = axes.flatten()
    for i in range(len(input_ps.points)):
        ax = flatax[i]
        lith = input_ps.lithology.isel(points=i).values
        tops = input_ps.tops_onder_mv.isel(points=i).values 
        bottoms = input_ps.bots_onder_mv.isel(points=i).values
        doorlat = doorlatend.isel(points=i).values

        mask = ~np.isnan(lith)
        for j, (top, bottom, litho, doorla) in enumerate(zip(tops[mask], bottoms[mask], lith[mask], doorlat[mask])):
            if np.isnan(top) or np.isnan(bottom):
                continue
            color = lithoklasse_colors.get(int(litho), "#FFFFFF")
            ax.bar(0, top - bottom, bottom=bottom, 
                     color=color, edgecolor='k', 
                     width=0.8, alpha=1,
                     hatch = hatch_dict[doorla])
        title = make_title(input_ps, i)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, -3)
        ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=-1.5, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=-2.5, color='red', linestyle='--', alpha=0.7)
        ax.invert_yaxis()
    fig.suptitle(f'Value={val}')
    #axes[0].set_ylabel('Depth')
    #plt.tight_layout()
    fig.savefig(Path(output_fn).parent / f'lithology_columns_val_{val}.png')

#%%
# test_c = (326950.0, 198450.0)
# input_ps = input.sel(x=test_c[0], y=test_c[1], method='nearest')
# %%
