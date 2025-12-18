#%%
from pathlib import Path
import xarray as xr
import utils
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from matplotlib.patches import Patch

result_folder = Path(r"N:\Projects\11211500\11211556\B. Measurements and calculations\kartering\resultaten")
map_folder = result_folder / "final_maps"
ras_folder = result_folder / "masked_maps"
legend_folder = result_folder / "legends"
fig_folder = result_folder / "figures"
fig_folder.mkdir(exist_ok=True)

drainage_gpkg_fn = map_folder / "drainage_in_peil.gpkg"

map_fns_gpkg = list([l for l in map_folder.iterdir() if np.isin(l.stem.split("_")[0], ['kem', 'oxidatie', 'compactie', 'krimp'])])
map_fns_tif = list(ras_folder.glob("*.tif"))
map_fns = map_fns_gpkg + map_fns_tif

drainage_gdf = gpd.read_file(drainage_gpkg_fn)
#%%
for map_fn in map_fns:
    # Determine the map category (e.g., 'kem', 'oxidatie', etc.)
    cat = map_fn.stem.split("_")[0]
    legend_fn = legend_folder / f"{cat}_legend.txt"
    
    # Load legend and create colormap
    legend = utils.read_qgis_raster_legend(legend_fn)
    cutoffs = np.array(legend[0])  # Value boundaries for color bins
    colors = np.array(legend[1])/255  # Normalize RGB colors to [0, 1]
    labels = legend[2]  # Legend labels

    cmap = ListedColormap(colors)
    if cat == 'kem':
        # Add an upper boundary if needed for BoundaryNorm
        cutoffs = np.append(cutoffs, cutoffs[-1] + 1)
    norm = BoundaryNorm(cutoffs, len(colors))

    # Plot the map
    fig, ax = plt.subplots(figsize=(10, 10))
    if map_fn.suffix == '.gpkg':
        # Read and plot vector data
        data = gpd.read_file(map_fn)
        data.plot(ax=ax,
                  column='agg_value',
                  norm=norm,
                  cmap=cmap)
        # Overlay drainage layer for 'kem' maps
        if cat == 'kem':
            drainage_gdf.plot(ax=ax, color='#ffee01')

        crs = data.crs
        
    elif map_fn.suffix == '.tif':
        # Read and plot raster data
        data = xr.open_dataarray(map_fn).squeeze()
        data.plot(ax=ax,
                  cmap=cmap,
                  norm=norm,
                  add_colorbar=False)
        crs = data.rio.crs
        ax.set_title('')

    # Set aspect ratio and map extent
    ax.set_aspect('equal')
    ax.set_xlim(0, 300000)
    ax.set_ylim(300000, 625000)

    # Add basemap for context
    ctx.add_basemap(
        ax,
        crs=crs,
        source=ctx.providers.CartoDB.Positron
    )

    # Build legend handles for each class
    handles = [
        Patch(facecolor=cmap(i), label=labels[i])
        for i in range(len(labels))
    ]
    # Add drainage legend entry if needed
    if cat == 'kem' and map_fn.suffix == '.gpkg':
        handles.append(Patch(facecolor='#ffee01', label='Drainage aanwezig'))
    ax.legend(handles=handles[::-1], loc='lower right', frameon=True)

    #make title
    year = '' if cat != 'oxidatie' else map_fn.stem.split("_")[1]
    title_args = {'kem': 'KEM',
                  'oxidatie': f'Oxidatie {year}',
                  'compactie': 'Compactie',
                  'krimp': 'Krimp'}
    plt.title(f'Gevoeligheid voor {title_args[cat]}', fontsize=13)

    # Save figure
    fig.savefig(fig_folder / f"{map_fn.stem}_map.png", dpi=300, bbox_inches='tight')
    plt.show()


# %%
#make histograms of both rasters and gpkg maps
for gpkg_fn in map_fns_gpkg:
    cat = '_'.join(gpkg_fn.stem.split("_")[:-2])
    ras_fn = ras_folder / f"{cat}_masked.tif"

    gdf = gpd.read_file(gpkg_fn)
    ras = xr.open_dataarray(ras_fn).squeeze()

    bins = 100

    fig, ax = plt.subplots(figsize=(6,4))
    gdf['agg_value'].hist(ax=ax, bins=bins, density=True, label='GPKG', alpha=0.7)
    ras.plot.hist(ax=ax, bins=bins, alpha=0.7, density=True, label='Raster')
    ax.legend()
    ax.set_title(f'Histogram van waarden voor {cat} kaart')
    ax.set_xlabel('Waarde')
    ax.set_ylabel('Aantal cellen')
    # fig.savefig(fig_folder / f"{gpkg_fn.stem}_histogram.png", dpi=300, bbox_inches='tight')
    plt.show()

# %%
#visualize combined maps
# Color legend for dominance logic
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Define color mapping for each code using primary colors
comb_colors = {
    1: (1, 0, 0),        # Oxidatie dominant - pure red
    2: (0, 0, 1),        # Compactie dominant - pure blue
    3: (1, 1, 0),        # Krimp dominant - pure yellow
    4: (0.5, 0, 0.5),    # Oxidatie & Compactie equal - magenta (red + blue)
    5: (1, 0.5, 0),      # Oxidatie & Krimp equal - orange (red + yellow)
    6: (0.3, 0.8, 0.3),        # Compactie & Krimp equal - green (blue + yellow)
    7: (0, 0, 0),  # All equal dominant - grey
}

comb_labels = {
    1: 'Oxidatie',
    2: 'Compactie',
    3: 'Krimp',
    4: 'Oxidatie & Compactie gelijk',
    5: 'Oxidatie & Krimp gelijk',
    6: 'Compactie & Krimp gelijk',
    7: 'Alle gelijk',
}

# Example: create legend handles for matplotlib
legend_handles = [
    mpatches.Patch(color=comb_colors[i], label=comb_labels[i]) for i in range(1, 8)
]

data_fn = map_folder / "combined_kartering_2015.tif"
data = xr.open_dataarray(data_fn).squeeze()

fig, ax = plt.subplots(figsize=(10, 10))
data.plot(
    ax=ax,
    cmap=ListedColormap([comb_colors[i] for i in range(1, 8)]),
    levels=np.arange(0.5, 8.5, 1),
    add_colorbar=False
)
ax.set_aspect('equal')
ax.set_xlim(0, 300000)
ax.set_ylim(300000, 625000)
ctx.add_basemap(
    ax,
    crs=data.rio.crs,
    source=ctx.providers.CartoDB.Positron
)

ax.legend(handles=legend_handles, loc='lower right', frameon=True)
ax.set_title('Gevoeligheid dominantie kaart', fontsize=13)
fig.savefig(fig_folder / f"{data_fn.stem}_map.png", dpi=300, bbox_inches='tight')
# Usage in a plot:
# plt.legend(handles=legend_handles, title="Dominantie type")
