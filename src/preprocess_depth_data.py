#%%
import xarray as xr
from pathlib import Path
from dask.diagnostics import ProgressBar
import numpy as np
from tqdm import tqdm
base_folder = Path(r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse")

#load data
subsurface_model_fn = base_folder / "AtlantisRuns/atlans_subsurface_model_Knaake.nc"
sm = xr.open_dataset(subsurface_model_fn).chunk(chunks={"x": 500, "y": 500, 'layer': -1})

# test_area = [[110000, 120000], [450000, 460000]]
# sm = sm.sel(x=slice(test_area[0][0], test_area[0][1]), 
#             y=slice(test_area[1][1], test_area[1][0]))


tot_thickness = sm.thickness.cumsum('layer').where(sm.thickness.notnull())
sm['tops_onder_mv'] = tot_thickness - tot_thickness.max('layer')

#%%
upper_cells = sm.tops_onder_mv > -2.5
upper_data = sm.where(upper_cells)
upper_data = upper_data[['lithology','thickness', 'mass_fraction_organic', 'tops_onder_mv']]
#%%
def shift_nans_to_end(arr):
    # arr: xarray.DataArray with 'layer' as one dimension
    def shift_func(a):
        if np.isnan(a).all() or not np.isnan(a).any():
            return a
        valid = a[~np.isnan(a)]
        nans = np.isnan(a).sum()
        return np.concatenate([valid, np.full(nans, np.nan)])
    return xr.apply_ufunc(
        shift_func,
        arr,
        input_core_dims=[['layer']],
        output_core_dims=[['layer']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[arr.dtype]
    )

shifted = upper_data.map(shift_nans_to_end)
#last_layer = shifted.thickness.notnull().argmin(dim='layer').max()
mask = ~shifted.thickness.isnull().all(dim=[d for d in shifted.dims if d != 'layer'])


# with ProgressBar():
#     #data_layers = shifted.sel(layer=slice(1, last_layer + 1))
#     shifted_masked = shifted.sel(layer=mask)
#     shifted_masked.to_netcdf(base_folder / "kartering"/ "input" / "base_data_shifted_testarea.nc", encoding={v: {"zlib": True, "complevel": 4} for v in shifted_masked.data_vars})
#     s = shifted.isel(x=1000,y=1000).compute()
# mask = ~shifted.thickness.isnull().all(dim=[d for d in shifted.dims if d != 'layer'])
# f


# with ProgressBar():
#     shifted = xr.Dataset()
#     for var in upper_data.data_vars:
#         if 'layer' in upper_data[var].dims:
#             shifted_var = shift_nans_to_end(upper_data[var])
#             # Remove layers where all values are nan
#             mask = ~shifted_var.isnull().all(dim=[d for d in shifted_var.dims if d != 'layer'])
#             shifted[var] = shifted_var.sel(layer=mask)
#         else:
#             shifted[var] = upper_data[var]

# s = shifted.isel(x=1000,y=1000).compute()

#%%
with ProgressBar():
    shifted.to_netcdf(base_folder / "kartering"/ "resultaten" / "base_data_shifted.nc", encoding={v: {"zlib": True, "complevel": 4} for v in shifted.data_vars})



# %%
