import datetime
import os
import time

from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch
import xarray as xr
import numpy as np
import yaml
from einops import rearrange

from torch.utils.data import Dataset


_DATA_PATHS = {
    "era5_1p5": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
    "era5_2p8": "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr",
    "era5_5p6": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
}

_T_dataset_name = Literal["era5_1p5", "era5_2p8", "era5_5p6"]


def _convert_longitude_360_to_180(ds: xr.Dataset, check: bool = False) -> xr.Dataset:
    # check if necessary (loads longitude coords)
    if check:
        longitude_coords = ds["longitude"].to_numpy()
        if longitude_coords.min() < 0.0 or longitude_coords.max() <= 180.0:
            return ds  # nop
    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
    smallest_negative_lon = np.argmin(ds.longitude.data).item()
    if ds.longitude.data[smallest_negative_lon] < 0:
        ds = ds.roll(longitude=len(ds.longitude) - smallest_negative_lon, roll_coords=True)
    return ds


class WeatherDataset(Dataset):
    def __init__(
            self,
            name: _T_dataset_name,
            variables: ERA5VariableConfig,
            in_memory: bool = False,
            seq_len: int = 2,
            seq_stride: int = 1,
            data_path: str | None = None,
            stats_path: str | None = None,
            clim_path: str | None = None,
            static_path: str | None = None,
            time_slice: dict | None = None,
            lat_slice: dict | None = None,
            lon_slice: dict | None = None,
            convert_lon_360_to_180: bool = True,
            extra_features: dict | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.in_memory = in_memory
        self._variables = variables
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.convert_lon_360_to_180 = convert_lon_360_to_180

        # 1. load the dataset (either local or remote)
        self.data_path = data_path
        self.is_offline: bool = True
        if data_path is None or not os.path.exists(data_path):
            # dataset not cached yet, load from remote
            self.ds = xr.open_zarr(_DATA_PATHS[name], storage_options={"token": "anon"}, chunks={})
            self.is_offline = False
        elif os.path.exists(data_path):
            # load cached data
            if data_path.lower().endswith(".nc"):
                self.ds = xr.open_dataset(data_path)
            elif data_path.lower().endswith(".zarr"):
                self.ds = xr.open_zarr(data_path)
            else:
                raise ValueError(f"data_path must end with .nc or .zarr, not {repr(data_path)}")
        else:
            raise ValueError(f"Invalid data_path {repr(data_path)}")

        # 2. select time period
        if time_slice is not None:
            self.ds = self.ds.sel(time=slice(time_slice.get("start"), time_slice.get("stop"), time_slice.get("step")))

        # 3. select variables
        if not self.is_offline:
            # we are working with the remote ERA5 data, thus we must first filter
            # the variables to fit in our data-structure
            self.ds = xr.Dataset(variables.sel_fields(self.ds)).transpose("time", "latitude", "longitude").chunk({"time": 100})
        self.ds = self.ds[variables.keys()]  # ensure variable ordering to be consistent

        # 4. change longitude coord from 0..360 to -180..180 (optional)
        if convert_lon_360_to_180:
            self.ds = _convert_longitude_360_to_180(self.ds, check=True)

        # 5. latitude / longitude slicing
        if lat_slice is not None:
            self.ds = self.ds.sel(latitude=slice(lat_slice.get("start"), lat_slice.get("stop"), lat_slice.get("step")))
        if lon_slice is not None:
            self.ds = self.ds.sel(longitude=slice(lon_slice.get("start"), lon_slice.get("stop"), lon_slice.get("step")))

        if not self.is_offline:
            print(self.ds.values())

        # load or generate stats
        self.stats = None  # tensor of shape (vars, (mean, std))
        if self.is_offline and stats_path is not None:
            if os.path.exists(stats_path):
                # load stats from disk
                if stats_path.lower().endswith(".nc"):
                    stats_ds = xr.open_dataset(stats_path)
                elif stats_path.lower().endswith(".zarr"):
                    stats_ds = xr.open_zarr(stats_path)
                else:
                    raise ValueError(f"stats_path must end with .nc or .zarr, not {repr(stats_path)}")
            else:
                # generate and store stats
                stats_ds = xr.concat(
                    [self.ds[self.variables].mean(), self.ds[self.variables].std()],
                    dim=xr.DataArray(["mean", "std"], dims="statistic", name="statistic"),
                ).compute()
                if stats_path.lower().endswith(".nc"):
                    stats_ds.to_netcdf(stats_path, mode="w")
                elif stats_path.lower().endswith(".zarr"):
                    stats_ds.to_zarr(stats_path, mode="w")
                else:
                    raise ValueError(f"stats_path must end with .nc or .zarr, not {repr(stats_path)}")
            self.stats = torch.from_numpy(stats_ds[self.variables].to_dataarray(dim="variables").to_numpy()).to(dtype=torch.float32)
            stats_ds.close()

        # load into tensor if we work in memory
        self.data = None  # shape -> (variables, time, latitude, longitude)
        if self.in_memory and self.is_offline:
            self.data = self._to_tensor(self.ds)

        # register extra features
        if extra_features is None:
            extra_features = dict()
        self._return_time: bool = extra_features.get("time", False)
        self._time = self._load_time()

        # load static features
        self.static_path = static_path
        self.latlon: torch.Tensor | None = None
        self.land_sea_mask: torch.Tensor | None = None
        self.geopotential_at_surface: torch.Tensor | None = None
        if self.static_path is not None:
            self.download_static_features()

        # load clim data
        self.clim: xr.Dataset | None = None
        if clim_path is not None:
            assert os.path.exists(clim_path)
            if clim_path.lower().endswith(".nc"):
                self.clim = xr.open_dataset(clim_path)[self.variables]
            elif clim_path.lower().endswith(".zarr"):
                self.clim = xr.open_zarr(clim_path)[self.variables]
            else:
                raise ValueError(f"clim_path must end with .nc or .zarr, not {repr(clim_path)}")

    @property
    def variables(self) -> list[str]:
        return self._variables.keys()

    @property
    def num_vars(self) -> int:
        return len(self.variables)

    @property
    def num_split_vars(self) -> tuple[int, int]:
        return self._variables.num_vars()

    @property
    def field_size(self) -> tuple[int, int]:
        return self.ds["latitude"].shape[0], self.ds["longitude"].shape[0]

    @property
    def time_delta(self) -> float:
        return (self._time[1] - self._time[0]).item() * self.seq_stride

    def load_to_memory(self) -> None:
        assert self.is_offline
        self.data = self._to_tensor(self.ds)
        self.in_memory = True

    def download_static_features(self) -> None:
        assert self.static_path is not None
        if os.path.isfile(self.static_path):
            # load from disk
            static_features = np.load(self.static_path)
            latlon = static_features["latlon"]
            land_sea_mask = static_features["land_sea_mask"]
            geopotential_at_surface = static_features["geopotential_at_surface"]
        else:
            # load from cloud
            print("Downloading static features...")
            ds = xr.open_zarr(_DATA_PATHS[self.name], storage_options={"token": "anon"}, chunks={})
            ds = ds[["land_sea_mask", "geopotential_at_surface"]]
            if self.convert_lon_360_to_180:
                ds = _convert_longitude_360_to_180(ds, check=True)
            latlon = rearrange(
                np.stack(
                    np.meshgrid(ds["latitude"].data, ds["longitude"].data),
                    axis=-1,
                ),
                "lon lat c -> lat lon c",
            )
            land_sea_mask = ds["land_sea_mask"].data.compute().T
            geopotential_at_surface = ds["geopotential_at_surface"].data.compute().T
            np.savez(
                self.static_path,
                latlon=latlon,
                land_sea_mask=land_sea_mask,
                geopotential_at_surface=geopotential_at_surface,
            )
        self.latlon = torch.from_numpy(latlon).to(dtype=torch.float32)
        self.land_sea_mask = torch.from_numpy(land_sea_mask).to(dtype=torch.float32)
        self.geopotential_at_surface = torch.from_numpy(geopotential_at_surface).to(dtype=torch.float32)

        # post-process
        self.geopotential_at_surface = (self.geopotential_at_surface - self.geopotential_at_surface.mean()) / self.geopotential_at_surface.std()

    def download(self) -> None:
        if self.is_offline:
            return  # already working from disk
        if self.data_path is None:
            raise ValueError("data_path must not be None")
        if os.path.exists(self.data_path):
            raise ValueError(f"data_path {repr(self.data_path)} already exists")
        # load and write to disk
        if self.data_path.lower().endswith(".nc"):
            self.ds.to_netcdf(self.data_path, mode="w")
        elif self.data_path.lower().endswith(".zarr"):
            self.ds.drop_encoding().to_zarr(self.data_path, mode="w")  # drop_encoding is a workaround for: https://github.com/zarr-developers/zarr-python/issues/2964
        else:
            raise ValueError(f"data_path must end with .nc or .zarr, not {repr(self.data_path)}")

    def to_xarray(self, x: torch.Tensor, **coords):
        x = x.detach().cpu().numpy()
        var_mean = self.stats[:, 0].numpy()
        var_std = self.stats[:, 1].numpy()

        if x.ndim == 3:  # (variables, latitude, longitude)
            x *= var_std.reshape(-1, 1, 1)
            x += var_mean.reshape(-1, 1, 1)
            assert len(coords) == 0, "For input with 3 dimensions (variables, latitude, longitude) there are no extra coords allowed!"

        elif x.ndim > 3:  # (variables, ..., latitude, longitude)
            extra_dims = x.ndim - 3
            assert extra_dims == len(coords), "For input with >3 dimensions (variables, ..., latitude, longitude) you need to specify the additional coords!"
            broadcast_shape = [-1] + [1 for _ in range(extra_dims)] + [1, 1]
            x *= var_std.reshape(broadcast_shape)
            x += var_mean.reshape(broadcast_shape)
        else:
            raise NotImplementedError

        _coords = {
            "latitude": self.ds["latitude"].data,
            "longitude": self.ds["longitude"].data,
        }
        _coords.update(coords)
        dims = list(coords.keys()) + ["latitude", "longitude"]
        ds = xr.Dataset(
            {var: xr.DataArray(x[i], dims=dims) for i, var in enumerate(self.variables)},
            coords=_coords,
        )
        return ds

    def denormalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        var_mean = self.stats[:, 0]
        var_std = self.stats[:, 1]
        if not torch.is_tensor(x):
            var_mean = var_mean.numpy()
            var_std = var_std.numpy()
        ndim = len(x.shape)
        if ndim == 3:  # (variables, latitude, longitude)
            var_std = var_std.reshape(-1, 1, 1)
            var_mean = var_mean.reshape(-1, 1, 1)
        elif ndim > 3:  # (variables, ..., latitude, longitude)
            extra_dims = ndim - 3
            broadcast_shape = [-1] + [1 for _ in range(extra_dims)] + [1, 1]
            var_std = var_std.reshape(broadcast_shape)
            var_mean = var_mean.reshape(broadcast_shape)
        else:
            raise NotImplementedError()
        return (x * var_std) + var_mean

    def _to_tensor(self, ds: xr.Dataset) -> torch.Tensor:
        data = torch.from_numpy(
            ds.to_array().astype("float32").values
        ).share_memory_().nan_to_num(nan=0.0)
        # standardize data
        assert self.stats is not None, "stats must be computed before data tensor can be built"
        data -= self.stats[:, 0].view(-1, 1, 1, 1)
        data /= self.stats[:, 1].view(-1, 1, 1, 1)
        # sanity check
        #assert torch.allclose(data.view(data.shape[0], -1).mean(dim=1), data.new_zeros(data.shape[0]), atol=0.1, rtol=0.1)
        #assert torch.allclose(data.view(data.shape[0], -1).std(dim=1), data.new_ones(data.shape[0]), atol=0.1, rtol=0.1)
        return data  # shape -> (variables, time, latitude, longitude)

    def _load_time(self) -> torch.Tensor:
        sample_times = list()
        for i, time in enumerate(self.ds["time"].data):
            time: datetime.datetime = time.astype('datetime64[us]').astype('O')
            sample_times.append((time.timetuple().tm_yday - 1) * 24 + time.hour)
        return torch.tensor(sample_times, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ds.time) - (self.seq_len * self.seq_stride) + self.seq_stride  # number of windows

    def __getitem__(self, index: int) -> torch.Tensor | Sequence[torch.Tensor]:
        out = list()
        if self.in_memory:
            out.append(self.data[:, index:index+(self.seq_len*self.seq_stride):self.seq_stride, :, :])
        else:
            out.append(self._to_tensor(self.ds.isel(time=slice(index, index + (self.seq_len * self.seq_stride), self.seq_stride))))
        if self._return_time:
            out.append(self._time[index : index+(self.seq_len*self.seq_stride) : self.seq_stride])

        if len(out) == 1:
            return out[0]
        return out

    @classmethod
    def from_config(cls, config: WeatherDatasetConfig) -> tuple[WeatherDataset, WeatherDataset | None, WeatherDataset | None]:
        variables = ERA5VariableConfig(**config.variables)
        return tuple(
            [WeatherDataset(
                name=config.name,
                data_path=data_path,
                stats_path=config.stats_path,
                clim_path=config.clim_path,
                static_path=config.static_path,
                variables=variables,
                time_slice=time_slice,
                seq_len=seq_len,
                seq_stride=seq_stride,
                in_memory=config.in_memory,
                lat_slice=config.lat_slice,
                lon_slice=config.lon_slice,
                convert_lon_360_to_180=config.convert_lon_360_to_180,
                extra_features=config.extra_features,
            ) if (i == 0 or (time_slice is not None and data_path is not None)) else None
            for i, (data_path, time_slice, seq_len, seq_stride) in enumerate((
                (config.train_data_path, config.train_time_slice, config.train_seq_len, config.train_seq_stride),
                (config.val_data_path, config.val_time_slice, config.val_seq_len, config.val_seq_stride),
                (config.test_data_path, config.test_time_slice, config.test_seq_len, config.test_seq_stride)
            ))
            ]
        )

@dataclass
class ERA5VariableConfig:
    T2M: bool = False
    U10M: bool = False
    V10M: bool = False
    TP6h: bool = False
    Q: Sequence[int] | int | None = None  # lab default: 700
    U: Sequence[int] | int | None = None  # lab default: 250
    V: Sequence[int] | int | None = None  # lab default: 250
    # we need at least Z500 & T850 for the evaluation
    Z: Sequence[int] | int = 500
    T: Sequence[int] | int = 850

    def sel_fields(self, ds: xr.Dataset) -> dict[str, xr.DataArray]:
        """ Selects the configured variables (at specified levels) from an ERA5
        dataset and returns the selection as dict.
        """
        fields = dict()
        if self.T2M:
            fields["T2M"] = ds["2m_temperature"]
        if self.U10M:
            fields["U10M"] = ds["10m_u_component_of_wind"]
        if self.V10M:
            fields["V10M"] = ds["10m_v_component_of_wind"]
        if self.TP6h:
            fields["TP6h"] = ds["total_precipitation_6hr"]

        def _sel_levels(variable: str, name: str, d: xr.Dataset, levels: int | Sequence[int] | None):
            if levels is None:
                return
            if isinstance(levels, int):
                levels = [levels]
            for level in levels:
                fields[f"{name}{level}"] = d[variable].sel(level=level, drop=True)

        _sel_levels("geopotential", "Z", ds, self.Z)
        _sel_levels("temperature", "T", ds, self.T)
        _sel_levels("specific_humidity", "Q", ds, self.Q)
        _sel_levels("u_component_of_wind", "U", ds, self.U)
        _sel_levels("v_component_of_wind", "V", ds, self.V)

        return fields

    def keys(self) -> list[str]:
        keys = list()
        if self.T2M:
            keys.append("T2M")
        if self.U10M:
            keys.append("U10M")
        if self.V10M:
            keys.append("V10M")
        if self.TP6h:
            keys.append("TP6h")

        def _level_keys(name: str, levels: int | Sequence[int] | None):
            if levels is None:
                return
            if isinstance(levels, int):
                levels = [levels]
            for level in levels:
                keys.append(f"{name}{level}")

        _level_keys("Z", self.Z)
        _level_keys("T", self.T)
        _level_keys("Q", self.Q)
        _level_keys("U", self.U)
        _level_keys("V", self.V)

        return keys

    def num_vars(self) -> tuple[int, int]:
        surface_vars = 0
        if self.T2M:
            surface_vars += 1
        if self.U10M:
            surface_vars += 1
        if self.V10M:
            surface_vars += 1
        if self.TP6h:
            surface_vars += 1

        atmosphere_vars = 0
        def _count_levels(levels: int | Sequence[int] | None):
            if levels is None:
                return 0
            if isinstance(levels, int):
                return 1
            return len(levels)

        atmosphere_vars += _count_levels(self.Z)
        atmosphere_vars += _count_levels(self.T)
        atmosphere_vars += _count_levels(self.Q)
        atmosphere_vars += _count_levels(self.U)
        atmosphere_vars += _count_levels(self.V)

        return surface_vars, atmosphere_vars



@dataclass
class WeatherDatasetConfig:
    name: _T_dataset_name
    stats_path: str
    train_data_path: str
    val_data_path: str | None = None
    test_data_path: str | None = None
    variables: dict[str, bool | int | Sequence[int]] = field(default_factory=lambda: dict(T=850, Z=500))
    in_memory: bool = False
    train_seq_len: int = 2
    train_seq_stride: int = 1
    train_time_slice: dict | None = None
    val_seq_len: int = 4
    val_seq_stride: int = 1
    val_time_slice: dict | None = None
    test_seq_len: int = 4
    test_seq_stride: int = 1
    test_time_slice: dict | None = None

    lat_slice: dict | None = None
    lon_slice: dict | None = None
    convert_lon_360_to_180: bool = False

    extra_features: dict | None = None
    clim_path: str | None = None
    static_path: str | None = None


if __name__ == "__main__":
    config = WeatherDatasetConfig(**yaml.safe_load(open("../../configs/data/era5_1p5.yml", "r")))
    config.in_memory = False
    train_dataset, val_dataset, _ = WeatherDataset.from_config(config)
    #time.sleep(5)
    print("downloading train...")
    train_dataset.download()
    #print("downloading validation...")
    #val_dataset.download()
    print("downloading DONE")
    #print(len(train_dataset.ds.time))
    #train_dataset.download_static_features()
    #print(train_dataset.ds.values())
    #print(dataset.data.shape, dataset.data.is_shared())

    #print(len(train_dataset), train_dataset.variables)
    #sample = train_dataset[6]
    #print(sample)
    #train_dataset.land_sea_mask

