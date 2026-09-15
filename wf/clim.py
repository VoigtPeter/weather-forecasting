import xarray as xr

from weatherbench2.utils import compute_hourly_climatology_mean_fast



if __name__ == '__main__':
    ds = xr.open_zarr("../cached-data/era5_2p8_1996_2015_3l.zarr")

    out = compute_hourly_climatology_mean_fast(ds, window_size=61, clim_years=slice("1996", "2015"), hour_interval=6)
    print(out)
    print("computing and writing to disk...")
    out.to_zarr("../cached-data/era5_2p8_1996_2015_3l_clim.zarr", compute=True)
