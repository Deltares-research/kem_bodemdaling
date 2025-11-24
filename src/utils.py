
#%%
import numpy as np
import xarray as xr

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

 