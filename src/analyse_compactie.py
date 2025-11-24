#%%
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import utils
from dask.diagnostics import ProgressBar
#%%
compactie_data_fn = r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse\kartering\resultaten\compactie_data_coarse_2022.nc"
input = xr.open_dataset(compactie_data_fn).chunk({'x': 500, 'y': 500, 'layer': -1})

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

# def make_title(input_ps, i):
#     return f'Point {i+1}\nx={int(input_ps.x[i])}\ny={int(input_ps.y[i])}\nOndiep: {input_ps.perc_shallow_doorlatend.isel(points=i).item():.2f}\nDiep: {input_ps.perc_deep_doorlatend.isel(points=i).item():.2f}'
# %%
num_samples_per_value = 50# Set the number of sample points per unique value
rows = num_samples_per_value
cols = 5
size_factor = 2
figsize= (cols, rows*size_factor)

# Get all non-null x, y coordinates
valid_mask = input.lithology.notnull().any('layer')
y_coords, x_coords = np.where(valid_mask)
coords = np.array([input.x[x_coords], input.y[y_coords]]).T

# Sample coordinates randomly
sample_indices = np.random.choice(len(coords), size=min(num_samples_per_value, len(coords)), replace=False)
coords = [coords[i] for i in sample_indices]

x = xr.DataArray([c[0] for c in coords], dims="points")
y = xr.DataArray([c[1] for c in coords], dims="points")
with ProgressBar():
    input_ps = input.sel(x=x, y=y, method = 'nearest').compute()
input_ps['center_z'] = (input_ps.tops_onder_mv + input_ps.bots_onder_mv) / 2

#%%
fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True, sharex  = 'col')
for i in range(len(input_ps.points)):
    input_p = input_ps.isel(points=i)
    glg_value = input_p.glg_mv.item()
    if glg_value > 0:
        glg_value = -0.05
    elif glg_value < -8:
        glg_value = -7.95
    score = input_p.compactie_score.item()
    input_p = input_p.where(input_p.center_z >= -8, drop=True)

    ax = axes[i,0]
    lith = input_p.lithology
    tops = input_p.tops_onder_mv
    bottoms = input_p.bots_onder_mv

    mask = ~np.isnan(lith)
    for j, (top, bottom, litho) in enumerate(zip(tops[mask], bottoms[mask], lith[mask])):
        if np.isnan(top) or np.isnan(bottom):
            continue
        color = lithoklasse_colors.get(int(litho), "#FFFFFF")
        ax.bar(0, top - bottom, bottom=bottom, 
                    color=color, edgecolor='k', 
                    width=0.8, alpha=1)
    ax.set_xticks([])
    ax.set_ylabel('m onder mv')
    ax.axhline(y=glg_value, color='r', linestyle='--', label='GLG')

    barh_height = abs(input_p.tops_onder_mv - input_p.bots_onder_mv)
    axes[i,1].barh(input_p.center_z, input_p.sw, height=barh_height)
    axes[i,1].set_xlim(-0.1,1.1)
    axes[i,1].set_title('SW')

    axes[i,2].plot(input_p.eff_stress, input_p.center_z)
    axes[i,2].set_title('Eff. stress')

    axes[i,3].barh(input_p.center_z, input_p.cr+0.03, height = barh_height)
    axes[i,3].set_xlim(-0.1,1.1)
    axes[i,3].set_title('CR')

    axes[i,4].barh(input_p.center_z, input_p.layer_compactie_score, height = barh_height)
    axes[i,4].axhline(y=glg_value, color='r', linestyle='--', label='GLG')
    axes[i,4].set_title(f'{score:.2f}')

    #title = make_title(input_ps, iz
    #ax.set_title(title)
    #ax.set_xticks([])
    #ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, -8)
    ax.invert_yaxis()

plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
fig.suptitle('Compactie', y=1)
fig.savefig(Path(compactie_data_fn).parent / f'compactie_example.png', dpi = 800)

# %%
