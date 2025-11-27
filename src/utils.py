
#%%
import numpy as np
import xarray as xr
from rasterio.features import rasterize
import pandas as pd
import xugrid as xu
from pathlib import Path
import geopandas as gpd
import shapely.geometry as gmt
import rasterio

SNAKEFILE_PATH = r"C:\git_repos\bodemdaling\workflow_kem"

def read_snakemake_rule(path, name: str) -> "snakemake.rules.Rule":
    """
    Parameters
    ----------
    path: str, pathlib.Path
        The path to the snakefile.
    name: str
        Name of the rule in the snakefile that runs this script.
        
    Returns
    -------
    rule: snakemake.rules.Rules
    
    Examples
    --------
    To run an example both interactively and in a workflow, e.g.:
    
    >>> if "snakemake" not in globals():
    >>>     snakemake = read_snakemake_rule("snakefile", rule="my_rule")
    >>> modelname = snakemake.params.modelname
    >>> template = snakemake.input["template"]
    """
    from snakemake.settings.types import ResourceSettings
    from snakemake.api import SnakemakeApi

    with SnakemakeApi() as snakemake_api:
        workflow = snakemake_api.workflow(
            resource_settings=ResourceSettings(),
            snakefile=Path(path),
        )
        rules = {rule.name: rule for rule in workflow._workflow.rules}
    
    rule = rules.get(name)
    if rule is None:
        raise ValueError(
            f"Rule {name} not in snakefile. Available rules: {', '.join(rules.keys())}")
    return rule

def mask_to_geodataframe(mask_da):
    """Convert a boolean mask DataArray to a GeoDataFrame of polygons."""
    shapes = rasterio.features.shapes(
        mask_da.astype('uint8').values,
        transform=rasterio.transform.from_bounds(
            mask_da.x.min(), mask_da.y.min(),
            mask_da.x.max(), mask_da.y.max(),
            mask_da.sizes['x'], mask_da.sizes['y']
        )
    )
    return gpd.GeoDataFrame([{'geometry': gmt.shape(shape), 'value': value} for shape, value in shapes if value == 1])

def data_to_geodataframe(data_da):
    """Convert a DataArray to a GeoDataFrame of polygons with values."""
    shapes = rasterio.features.shapes(
        data_da.values,
        transform=rasterio.transform.from_bounds(
            data_da.x.min(), data_da.y.min(),
            data_da.x.max(), data_da.y.max(),
            data_da.sizes['x'], data_da.sizes['y']
        )
    )
    return gpd.GeoDataFrame([{'geometry': gmt.shape(shape), 'value': value} for shape, value in shapes if not np.isnan(value)])

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
    if type(input_gdf) is gpd.GeoSeries:
        input_du = xu.earcut_triangulate_polygons(input_gdf.to_frame())
    elif type(input_gdf) is gpd.GeoDataFrame:
        input_du = xu.earcut_triangulate_polygons(input_gdf)
    elif type(input_gdf) is xu.core.wrap.UgridDataArray:
        input_du = input_gdf
    elif type(input_gdf) is xr.DataArray:
        input_du = input_gdf

    rel_overlap_regr = xu.RelativeOverlapRegridder(template_da, input_du)
    weights = rel_overlap_regr.weights_as_dataframe()
    weights_source = weights.groupby('source_index').sum()
    weights_source['weight'] = np.minimum(weights_source['weight'], 1.0)

    dummy_df = template_da.to_dataframe(name='dummy').reset_index()
    overlap = pd.merge(weights_source, dummy_df, left_on='source_index', right_index=True, how='right')

    fraction_per_cell = (type(template_da)).from_series(overlap.set_index(list(template_da.dims))['weight'])
    fraction_per_cell.rio.set_crs(28992, inplace=True)
    return fraction_per_cell

def select_layers(lith_col, layer_bottoms, layer_tops, startdepth, enddepth):
    adapted_tops = xr.apply_ufunc(
        np.minimum, startdepth, layer_tops,
        dask='parallelized', keep_attrs=True
    )

    adapted_bottoms = xr.apply_ufunc(
        np.maximum, enddepth, layer_bottoms,
        dask='parallelized', keep_attrs=True
    )
    adapted_thickness = adapted_tops - adapted_bottoms
    adapted_thickness = adapted_thickness.where(adapted_thickness > 0)

    adapted_lith = lith_col.where(adapted_thickness.notnull())
    return adapted_tops, adapted_thickness, adapted_lith

def best_grid(n):
    """Return the best (rows, cols) for n subplots."""
    if n == 0:
        return (0, 0)
    best_diff = n
    best_rows = 1
    best_cols = n
    for rows in range(1, n + 1):
        cols = int(np.ceil(n / rows))
        diff = abs(rows - cols)
        if diff < best_diff:
            best_diff = diff
            best_rows = rows
            best_cols = cols
    return best_rows, best_cols

def sample_coordinates_by_value(output, num_samples_per_value):
    """
    Sample coordinates for each unique value in the output array.
    
    Parameters:
    output: xarray.DataArray - The data array to sample from
    num_samples_per_value: int - Maximum number of samples per unique value
    
    Returns:
    dict: Dictionary mapping values to lists of [x, y] coordinates
    """
    unique_values = np.unique(output.values)
    unique_values = unique_values[~np.isnan(unique_values)]
    
    coords_dict = {}
    for val in unique_values:
        mask = output.values == val
        if not np.any(mask):
            coords_dict[val] = []
            continue
        idxs = np.where(mask)
        coords = np.array(list(zip(output.x[idxs[1]].values, output.y[idxs[0]].values)))
        coords_dict[val] = coords[np.random.choice(len(coords), min(num_samples_per_value, len(coords)), replace=False)].tolist()
    
    return coords_dict

def calc_avg_lowest_three(da):
    """
    Calculate the average of the three lowest values in the data array.
    
    Parameters:
    da (xarray.DataArray): The data array.
    
    Returns:
    xarray.DataArray: The average of the three lowest values.
    """
    # Compute rank
    da = da.chunk({'time': -1}) # Ilja: added chunk to get it ranked in chunks
    rank = da.rank(dim='time')#.compute()
    
    # Select lowest three values
    lowest_values = da.where(rank <= 3)
    
    # Compute mean of the lowest three values
    mean_lowest_values = lowest_values.mean(dim='time')
    
    return mean_lowest_values

def write_qgis_legend(filename, x_values, colors, labels, interpolation="DISCRETE"):
    """
    Write a QGIS legend export file.

    Parameters:
    ----------
    filename : str
        Path to the output file.
    x_values : list of float
        Cutoff values for each class (last value can be 'inf').
    colors : list of tuple
        List of (R, G, B, A) tuples (0–255).
    labels : list of str
        Legend labels for each class.
    interpolation : str
        Interpolation type, default "DISCRETE".
    """
    if not (len(x_values) == len(colors) == len(labels)):
        raise ValueError("x_values, colors, and labels must have the same length")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"INTERPOLATION:{interpolation}\n")
        for val, color, label in zip(x_values, colors, labels):
            r, g, b, a = color
            f.write(f"{val},{r},{g},{b},{a},{label}\n")

def read_qgis_legend(filename):
    """
    Read cutoff values, colors, and labels from a QGIS legend export file.

    Parameters:
    ----------
    filename : str
        Path to the legend file.

    Returns:
    -------
    tuple
        (x_values, colors, labels, interpolation) where:
        - x_values: list of float (cutoff values)
        - colors: list of tuple (R, G, B, A) values
        - labels: list of str (legend labels)
        - interpolation: str (interpolation type)
    """
    x_values = []
    colors = []
    labels = []
    interpolation = "DISCRETE"
    
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("INTERPOLATION:"):
                interpolation = line.split(":", 1)[1]
            elif line and not line.startswith("INTERPOLATION:"):
                parts = line.split(",", 5)
                if len(parts) >= 6:
                    val = float(parts[0]) if parts[0] != 'inf' else float('inf')
                    r, g, b, a = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    label = parts[5]
                    
                    x_values.append(val)
                    colors.append((r, g, b, a))
                    labels.append(label)
    
    return x_values, colors, labels, interpolation
# %%
