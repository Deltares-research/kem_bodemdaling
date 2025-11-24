#%% 
#load modules
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import utils
from dask.diagnostics import ProgressBar

input_fn = r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse\kartering\resultaten\oxidatie_data_coarse_2022.nc"
input = xr.open_dataset(input_fn).chunk({'x': 1000, 'y': 1000, 'layer': -1})

# output_fn = r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse\kartering\resultaten\impact_factor_kem_teun.tif"
# output = xr.open_dataarray(output_fn).squeeze('band', drop=True).compute()

lithoklasse_colors = {
    0: "#C0C0C0",  # Grijs (a)
    1: "#8B4513",  # SaddleBrown (v)
    2: "#008000",# Groenachtig (k)
    3: "#6CA36C",  # Minder fel groen (kz)
    4: "#FFFF00",  # Geel (zf)
    5: "#FFD700",  # Goud (zm)
    6: "#FFA500",  # Oranje (zg)
    7: "#DAA520",  # GoldenRod (g)
    8: "#7212DF"   # Blauw (she)

}

doorlatend_enums = [0, 1, 5, 6, 7, 8, 9,11]

hatch_dict = {True: '///', False: None}

def make_title(input_ps, i):
    return f'x={int(input_ps.x[i])}\ny={int(input_ps.y[i])}\nps: {input_ps.yearly_points.isel(points=i).item():.2f}'

#%%
num_samples_per_value = 64 # Set the number of sample points per unique value
rows, cols = utils.best_grid(num_samples_per_value)
size_factor = 3
figsize = (rows*size_factor, cols*size_factor)

# Get all non-null x, y coordinates
valid_mask = input.yearly_points>0
y_coords, x_coords = np.where(valid_mask)
coords = [(input.x[x].item(), input.y[y].item()) for x, y in zip(x_coords, y_coords)]

# Sample coordinates randomly
sample_indices = np.random.choice(len(coords), size=min(num_samples_per_value, len(coords)), replace=False)
coords = [coords[i] for i in sample_indices]

x= xr.DataArray([c[0] for c in coords], dims="points")
y = xr.DataArray([c[1] for c in coords], dims="points")

with ProgressBar():
    input_ps = input.sel(x=x, y=y, method = 'nearest').compute()
input_ps['time'] = input_ps.time.dt.dayofyear
#%% 
fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True)
flatax = axes.flatten()
for i in range(len(input_ps.points)):
    ax = flatax[i]
    input_p = input_ps.isel(points=i)
    lith = input_p.lithology
    tops = input_p.tops_onder_mv
    bottoms = input_p.bots_onder_mv
    doorlat = input_p.organisch_mask.values

    mask = ~np.isnan(lith)
    for j, (top, bottom, litho, doorla) in enumerate(zip(tops[mask], bottoms[mask], lith[mask], doorlat[mask])):
        if np.isnan(top) or np.isnan(bottom):
            continue
        color = lithoklasse_colors.get(int(litho), "#FFFFFF")
        ax.bar(365/2, top - bottom, bottom=bottom, 
                    color=color, edgecolor='k', 
                    width=365, alpha=1,
                    hatch = hatch_dict[doorla])
    
    #plot points per day
    ps = (input_p.total_points / input_p.total_points.max())*0.3
    ps.plot(ax=ax, color='k', label='Punten boven GW')

    #plot grondwater
    if (input_p.gw_stand > 0).all():
        input_p['gw_stand'] = np.minimum(input_p.gw_stand, -0.03)
    elif (input_p.gw_stand < -1.5).all():
        input_p['gw_stand'] = np.maximum(input_p.gw_stand, -1.47)
    input_p.gw_stand.plot(ax=ax, color='blue', label='GW stand')
    title = make_title(input_ps, i)
    ax.set_title(title)
    
    ax.set_xticks([])
    ax.set_ylabel('')
    ax.set_xlabel('')
   
    # ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.4, -1.5)
    ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-0.7, color='red', linestyle='--', alpha=0.7)
    ax.invert_yaxis()

    plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
fig.suptitle('Oxidatie', y=1)
fig.savefig(Path(input_fn).parent / f'oxidatie_example_columns.png', dpi = 800)

#%%

# %%
