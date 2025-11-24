#%%
from pathlib import Path
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar


#%%
base_folder = Path(r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse")
subsurface_model_fn = base_folder / "AtlantisRuns/atlans_subsurface_model_Knaake.nc"
sm = xr.open_dataset(subsurface_model_fn).chunk(chunks={"x": 500, "y": 500, 'layer': -1})

with ProgressBar():
    sm = sm.coarsen(x=10, y=10, boundary='trim').median().compute()
sm.to_netcdf(base_folder / "kartering"/ "resultaten" / "atlans_subsurface_model_Knaake_coarse.nc", encoding={v: {"zlib": True, "complevel": 4} for v in sm.data_vars})

tot_thickness = sm.thickness.cumsum('layer').where(sm.thickness.notnull())
sm['tops_onder_mv'] = tot_thickness - tot_thickness.max('layer')

##%%
upper_cells = sm.tops_onder_mv > -2.5
upper_data = sm.where(upper_cells)
upper_data = upper_data[['lithology','thickness', 'mass_fraction_organic', 'tops_onder_mv']]
##%%
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
with ProgressBar():
    shifted.to_netcdf(base_folder / "kartering"/ "resultaten" / "base_data_shifted_coarse.nc", encoding={v: {"zlib": True, "complevel": 4} for v in shifted.data_vars})

# %%
folder = Path(r"N:\Projects\11209000\11209258\B. Measurements and calculations\GWS analyse\kartering\lhm_data")
newfolder = folder.parent
for head_fn in folder.glob("lhm_433_heads_l1_*.nc"):
    year = head_fn.stem.split('_')[-1]
    print(year)
    gw_stand_nap = xr.open_dataarray(head_fn)#.chunk(x=200, y = 200, time = 50)
    print(gw_stand_nap.time.size)
    gw_stand_nap_year = gw_stand_nap.sel(time = (gw_stand_nap.time.dt.year == int(year)))
    with ProgressBar():
        gw_stand_nap_year.to_netcdf(newfolder / head_fn.name)
# %%
