#%%
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from shapely.strtree import STRtree
from matplotlib.colors import ListedColormap
from itertools import combinations
import matplotlib.patches as mpatches
#%%
base_folder = Path(r"N:\Projects\11211500\11211556\B. Measurements and calculations\kartering")
fin_map_folder = base_folder / "resultaten" / "final_maps"

compactie_fn = fin_map_folder / "compactie_score_agg_masked.gpkg"
krimp_fn = fin_map_folder / "krimp_score_agg_masked.gpkg"
oxidatie_fn = fin_map_folder / "oxidatie_2018_score_agg_masked.gpkg"
kem_fn = fin_map_folder / "kem_score_agg_masked.gpkg"

#load additional overlays
drainage_fn = r"N:\Projects\11211500\11211556\B. Measurements and calculations\kartering\uitsluitingsgebieden\drainage.shp"
nl_shape_fn = r"P:\gis-data\provincie\2021_provincies_zonder_water.shp"

#load all data
compactie_gdf = gpd.read_file(compactie_fn)
krimp_gdf = gpd.read_file(krimp_fn)
oxidatie_gdf = gpd.read_file(oxidatie_fn)

sens_dict = {'compactie': compactie_gdf,
             'krimp': krimp_gdf,
             'oxidatie': oxidatie_gdf}

kem_gdf = gpd.read_file(kem_fn)
drainage_gdf = gpd.read_file(drainage_fn)
nl_shape_gdf = gpd.read_file(nl_shape_fn)

kem_sens = kem_gdf[kem_gdf['agg_value']==0.8]
kem_shape = kem_sens.union_all()
drainage_shape = drainage_gdf.union_all()
mask_shape = kem_shape.difference(drainage_shape)
mask_shape = mask_shape.simplify(10)
#%%
comb_gdf = []
quantile = 0.95
for name, gdf in sens_dict.items():
    tresh_q = np.quantile(gdf['agg_value'], 
                q=quantile, 
                method = 'inverted_cdf',
                weights = gdf.area)
    gdf = gdf[gdf['agg_value'] >= tresh_q]
    gdf['type'] = name
    comb_gdf.append(gdf)
comb_gdf = pd.concat(comb_gdf).reset_index()

comb_gdf['geometry'] = comb_gdf.intersection(mask_shape)
comb_gdf = comb_gdf[~comb_gdf.is_empty]
comb_gdf = comb_gdf[comb_gdf.type.isin(['Polygon', 'MultiPolygon'])]
comb_gdf = comb_gdf.explode(index_parts=False).reset_index(drop=True)

#%%
# Map colors to geometries based on code
fig, ax = plt.subplots(figsize=(12, 10))

comb_colors = {
    'compactie': (0,0,1),        # Compactie dominant
    'krimp': (1,1,0),        # Krimp dominant
    'oxidatie': (1,0,0)}        # Oxidatie dominant}

for code in sorted(comb_gdf['type'].unique()):
    subset = comb_gdf[comb_gdf['type'] == code]
    subset.plot(
        ax=ax,
        color=comb_colors[code],
        alpha = 1.0,
        label = code.capitalize(),
    )

nl_shape_gdf.boundary.plot(ax=ax, color='black', linewidth=1)

ax.set_aspect('equal')
ax.legend(loc='best')
ax.set_title('Meest gevoelige peilgebieden\nmet mogelijkheid tot effectieve peilaanpassing')

# Add basemap for context
ctx.add_basemap(ax, crs=28992, source=ctx.providers.CartoDB.Positron)

plt.tight_layout()
plt.show()


#%%
# # Find all geometries, including non-overlapping and overlapping (pairwise and triple) intersections



#%%
#     for t in other_types:
#         other_gdf = comb_gdf[comb_gdf['type'] == t]
#         intersections = gpd.overlay(type_gdf, other_gdf, how='intersection')
#         comb_gdf = pd.concat([comb_gdf, intersections]).reset_index(drop=True)

    



# # Convert intersections to GeoDataFrame
# intersections_gdf = gpd.GeoDataFrame(intersections, geometry='geometry')
# intersections_gdf = intersections_gdf[~intersections_gdf.is_empty]
# intersections_gdf = intersections_gdf[intersections_gdf.type.isin(['Polygon', 'MultiPolygon'])]

# for i, row in intersections_gdf.iterrows():
#     types = set([row[col] for col in row.index if (col.startswith('type_') & (str(row[col])!= 'nan'))])
#     types_sorted = sorted(types)
#     if types_sorted == ['compactie']:
#         code = 1
#     elif types_sorted == ['krimp']:
#         code = 2
#     elif types_sorted == ['oxidatie']:
#         code = 3
#     elif types_sorted == ['compactie', 'oxidatie']:
#         code = 4
#     elif types_sorted == ['krimp', 'oxidatie']:
#         code = 5
#     elif types_sorted == ['compactie', 'krimp']:
#         code = 6
#     elif types_sorted == ['compactie', 'krimp', 'oxidatie']:
#         code = 7
#     intersections_gdf.at[i, 'code'] = code

# #%%
# comb_colors = {
#     1: (1, 0, 0),        # Oxidatie dominant - pure red
#     2: (0, 0, 1),        # Compactie dominant - pure blue
#     3: (1, 1, 0),        # Krimp dominant - pure yellow
#     4: (0.5, 0, 0.5),    # Oxidatie & Compactie equal - magenta (red + blue)
#     5: (1, 0.5, 0),      # Oxidatie & Krimp equal - orange (red + yellow)
#     6: (0.3, 0.8, 0.3),  # Compactie & Krimp equal - green (blue + yellow)
#     7: (0.6, 0.6, 0.6),  # All three - gray
# }

# comb_labels = {
#     1: 'Oxidatie',
#     2: 'Compactie',
#     3: 'Krimp',
#     4: 'Oxidatie & Compactie',
#     5: 'Oxidatie & Krimp',
#     6: 'Compactie & Krimp',
#     7: 'Oxidatie & Compactie & Krimp'
# }

# # Map colors to geometries based on code
# fig, ax = plt.subplots(figsize=(12, 10))

# for code in sorted(intersections_gdf['code'].unique()):
#     subset = intersections_gdf[intersections_gdf['code'] == code]
#     subset.plot(
#         ax=ax,
#         color=comb_colors[code],
#         label=comb_labels[code],
#     )

# nl_shape_gdf.boundary.plot(ax=ax, color='black', linewidth=1)

# ax.set_aspect('equal')
# ax.legend(loc='best')
# ax.set_title('Dominante bodemdal processen')

# # Add basemap for context
# ctx.add_basemap(ax, crs=28992, source=ctx.providers.CartoDB.Positron)

# plt.tight_layout()
# plt.show()

# %%
