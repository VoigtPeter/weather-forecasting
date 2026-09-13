import math
import tqdm

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader
from einops import rearrange

from wf.scaffold import ForecastModule
from wf.utils.config import Config
from wf.utils.ensemble import ensemble_batch, reverse_ensemble_batch


def _w_mean(x: np.ndarray, w: np.ndarray, axis = None) -> np.ndarray:
    return (x * w).sum(axis=axis) / w.sum(axis=axis)


def _slice_clim(clim: xr.Dataset, hod: np.ndarray, doy: np.ndarray) -> np.ndarray:
    if len(hod.shape) == 2:
        batch_slices = list()
        for h, d in zip(hod, doy):
            batch_slices.append(_slice_clim(clim, h, d))
        return np.stack(batch_slices, axis=0)

    clim_slices = list()
    for h, d in zip(hod, doy):
        clim_slices.append(np.expand_dims(clim.sel(dayofyear=d, hour=h).to_array(dim="variable").data, axis=1))
    return np.concatenate(clim_slices, axis=1)


def anomaly_correlation_coef(forecast: np.ndarray, true: np.ndarray, clim: np.ndarray, lat_weights: np.ndarray, axis = None) -> np.ndarray:
    forecast_delta = forecast - clim
    true_delta = true - clim
    return _w_mean(forecast_delta * true_delta, lat_weights, axis=axis) / (np.sqrt(_w_mean(forecast_delta ** 2,
                                                                                           lat_weights,
                                                                                           axis=axis) * _w_mean(
        true_delta ** 2, lat_weights, axis=axis)))

def rmse(forecast: np.ndarray, true: np.ndarray, lat_weights: np.ndarray, axis = None) -> np.ndarray:
    return np.sqrt(_w_mean((forecast - true) ** 2, lat_weights, axis=axis))


def activity(x: np.ndarray, clim: np.ndarray, lat_weights: np.ndarray, axis = None) -> np.ndarray:
    return rmse(x, clim, lat_weights, axis=axis)


def var_time_rank_histogram(x: np.ndarray) -> np.ndarray:
    # x -> (b, var, ens, time, h, w)
    x_sorted = np.argsort(x, axis=2)
    x_min = np.argmin(x_sorted, axis=2)
    # x_min -> (b, var, time, h, w)
    x_min = np.transpose(x_min, (1, 2, 3, 4, 0))
    # x_min -> (var, time, h, w, b)
    x_min = x_min.reshape(x_min.shape[0], x_min.shape[1], -1)  # flatten last 3 dims
    # x_min -> (var, time, h*w*b)
    total_counts = list()
    for var_slice in x_min:
        # var_slice -> (time, h*w*b)
        time_counts = list()
        for time_slice in var_slice:
            # time_slice -> (h*w*b,)
            _, counts = np.unique(time_slice, return_counts=True, sorted=True)
            time_counts.append(counts)  # -> (bins,)
        total_counts.append(np.stack(time_counts, axis=0))  # -> (time, bins)
    return np.stack(total_counts, axis=0)  # -> (var, time, bins)


def classify_clim_above_below(y_pred: np.ndarray, y_true: np.ndarray, y_clim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # y_pred -> (b, var, ens, time, h, w)
    # y_true -> (b, var, 1, time, h, w)
    # y_clim -> (b, var, 1, time, h, w)
    pred_delta = y_pred - y_clim
    true_delta = y_true - y_clim
    ret_pred = np.empty_like(y_pred, dtype=np.int8)
    ret_pred[pred_delta < 0] = 0
    ret_pred[pred_delta >= 0] = 1
    ret_true = np.empty_like(y_true, dtype=np.int8)
    ret_true[true_delta < 0] = 0
    ret_true[true_delta >= 0] = 1
    return ret_pred, ret_true


def var_time_reliability(y_pred_cls: np.ndarray, y_true_cls: np.ndarray):
    # y_pred_cls -> (b, var, ens, time, h, w)
    # y_true_cls -> (b, var, 1, time, h, w)
    n_ens = y_pred_cls.shape[2]
    n_bins = n_ens + 1
    n_vars = y_pred_cls.shape[1]
    n_times = y_pred_cls.shape[3]
    eye = np.eye(n_bins, dtype=np.int8)
    eye2 = np.eye(2, dtype=np.int8)

    # (2, var, time, bins)
    pred_histogram = np.zeros((2, n_vars, n_times, n_bins), dtype=np.int64)  # -> (2 {c0/c1}, ...)
    true_count_0 = np.zeros((n_vars, n_times, n_bins, 2), dtype=np.int64)  # -> (var, time, bins, 2)
    true_count_1 = np.zeros((n_vars, n_times, n_bins, 2), dtype=np.int64)  # -> (var, time, bins, 2)

    # pred: (var, time, b*h*w, ens) -> count -> (var, time, b*h*w) -> index_eye -> (var, time, b*h*w, bins) -> sum -> (var, time, bins)
    #y_pred_cls = np.transpose(np.transpose(y_pred_cls, (1, 3, 2, 0, 4, 5)).reshape(n_vars, n_times, -1, n_ens), (0, 1, 3, 2))
    y_pred_cls = rearrange(y_pred_cls, "batch var ens time h w -> var time (batch h w) ens")
    y_pred_counts_1 = np.count_nonzero(y_pred_cls, axis=-1)  # -> (var, time, b*h*w)
    y_pred_counts_0 = n_ens - y_pred_counts_1
    pred_histogram[0] = eye[y_pred_counts_0].sum(axis=2)  # -> (var, time, bins)
    pred_histogram[1] = eye[y_pred_counts_1].sum(axis=2)  # -> (var, time, bins)

    # true: (var, time, b*h*w)
    y_true_cls = rearrange(y_true_cls, "batch var i time h w -> var time (batch i h w)")
    np.add.at(true_count_0, (
        np.arange(n_vars)[:, None, None, None],
        np.arange(n_times)[None, :, None, None],
        y_pred_counts_0[:, :, :, None],
        np.arange(2)[None, None, None, :]
    ), eye2[y_true_cls])
    np.add.at(true_count_1, (
        np.arange(n_vars)[:, None, None, None],
        np.arange(n_times)[None, :, None, None],
        y_pred_counts_1[:, :, :, None],
        np.arange(2)[None, None, None, :]
    ), eye2[y_true_cls])

    return pred_histogram, np.stack((true_count_0, true_count_1), axis=-1)


def compute_acc(checkpoint_path: str, config_path: str, rollout_steps: int = 5, ensemble_size: int = 2):
    config = Config.from_yaml(config_path)
    config.dataset.in_memory = False
    config.dataset.extra_features = dict(time=True)

    # remove ._orig_mod from state dict (torch.compile artifact)
    checkpoint = torch.load(checkpoint_path)
    state_dict = dict()
    for key, value in checkpoint["state_dict"].items():
        key: str = key.replace("._orig_mod", "")
        state_dict[key] = value.to("cpu")
    model: ForecastModule = ForecastModule(config)
    model.load_state_dict(state_dict)
    model = model.to("cuda")

    model.train_dataset = None

    model.model = torch.compile(model.model, fullgraph=True, mode="max-autotune")

    dataset = model.val_dataset  # TODO: make configurable
    #dataset.load_to_memory()

    dataset.seq_len = rollout_steps + 1

    # compute latitude weights
    latitudes = np.abs(dataset.ds["latitude"].data) * (math.pi / 180.0)
    lat_weights = np.cos(latitudes) / np.mean(np.cos(latitudes))
    assert np.isclose(lat_weights.sum(), len(latitudes)) and np.isclose(lat_weights.mean(), 1.0)
    lat_weights = lat_weights.reshape(-1, 1)  # -> (lat, 1)

    # gather metrics
    ACC: np.ndarray | None = None
    RMSE: np.ndarray | None = None
    A_pred: np.ndarray | None = None
    A_true: np.ndarray | None = None
    rank_histogram_bins: np.ndarray | None = None
    clim_pred_histogram_bins: np.ndarray | None = None
    clim_true_histogram_bins: np.ndarray | None = None
    num_samples = 0
    loader = DataLoader(dataset, batch_size=4, num_workers=0)
    with torch.no_grad():
        for (steps, times) in tqdm.tqdm(loader, total=len(loader)):
            # 1. split inputs / gt_outputs
            x_field = steps[:, :, 0, ...] # TODO: adapt when model supports >1 field as input
            x_time = times[:, 0]
            y_true_field = steps[:, :, 1:, ...].numpy()
            y_time = times[:, 1:]

            # 2. convert step times (y_time) to dayofyear & hourofday (used to determine clim slice)
            doy = (y_time // 24).detach().cpu().numpy().astype(int) + 1
            hod = (y_time % 24).detach().cpu().numpy().astype(int)
            clim = _slice_clim(dataset.clim, hod, doy).compute()

            # 3. make forecast
            torch.compiler.cudagraph_mark_step_begin()
            x_field = ensemble_batch(x_field.to(model.device), ensemble_size)
            x_time = ensemble_batch(x_time.to(model.device), ensemble_size)
            y_pred_field = reverse_ensemble_batch(model.forecast(x_field, rollout_steps, x_time, dataset.time_delta), ensemble_size)  # un-batch
            y_pred_field = y_pred_field.transpose(2, 1)
            y_pred_field = y_pred_field.detach().cpu().numpy()

            # 4. denorm
            y_pred_field_denorm = dataset.denormalize(y_pred_field.transpose(1, 0, 2, 3, 4, 5)).transpose(1, 0, 2, 3, 4, 5)
            y_true_field_denorm = dataset.denormalize(np.expand_dims(y_true_field, axis=2).transpose(1, 0, 2, 3, 4, 5)).transpose(1, 0, 2, 3, 4, 5)
            lat_weights_reshaped = lat_weights.reshape(1, 1, 1, 1, -1, 1)
            clim_reshaped = np.expand_dims(clim, axis=2)

            # rank-histogram binning
            true_pred_cat = np.concatenate(
                (y_true_field_denorm, y_pred_field_denorm),
                axis=2,
            )
            bin_counts = var_time_rank_histogram(true_pred_cat)  # the rank-target must be the first item at the specified axis, i.e., y_true
            if rank_histogram_bins is None:
                rank_histogram_bins = bin_counts
            else:
                rank_histogram_bins += bin_counts

            # reliability curve binning
            y_pred_cls, y_true_cls = classify_clim_above_below(y_pred_field_denorm, y_true_field_denorm, clim_reshaped)
            cls_pred_hist, cls_true_hist = var_time_reliability(y_pred_cls, y_true_cls)
            if clim_pred_histogram_bins is None:
                clim_pred_histogram_bins = cls_pred_hist
            else:
                clim_pred_histogram_bins += cls_pred_hist
            if clim_true_histogram_bins is None:
                clim_true_histogram_bins = cls_true_hist
            else:
                clim_true_histogram_bins += cls_true_hist

            # compute other metrics (for ensemble mean & all members)
            y_pred_field_denorm = np.concatenate((
                y_pred_field_denorm.mean(axis=2, keepdims=True),
                y_pred_field_denorm,
            ), axis=2)

            num_samples += steps.shape[0]  # batch-size
            acc_ = anomaly_correlation_coef(
                forecast=y_pred_field_denorm,
                true=y_true_field_denorm,
                clim=clim_reshaped,
                lat_weights=lat_weights_reshaped,
                axis=(-1, -2),  # (lat, lon)
            ).sum(axis=0)
            if ACC is not None:
                ACC += acc_
            else:
                ACC = acc_

            rmse_ = rmse(
                forecast=y_pred_field_denorm,
                true=y_true_field_denorm,
                lat_weights=lat_weights_reshaped,
                axis=(-1, -2),  # (lat, lon)
            ).sum(axis=0)
            if RMSE is not None:
                RMSE += rmse_
            else:
                RMSE = rmse_

            a_pred = activity(
                x=y_pred_field_denorm,
                clim=clim_reshaped,
                lat_weights=lat_weights_reshaped,
                axis=(-1, -2),  # (lat, lon)
            ).sum(axis=0)
            if A_pred is not None:
                A_pred += a_pred
            else:
                A_pred = a_pred

            a_true = activity(
                x=y_true_field_denorm,
                clim=clim_reshaped,
                lat_weights=lat_weights_reshaped,
                axis=(-1, -2),  # (lat, lon)
            ).sum(axis=0)
            if A_true is not None:
                A_true += a_true
            else:
                A_true = a_true

            # TODO: tmp remove!!!
            #if num_samples > 32:
            #    break

    ACC = ACC / num_samples
    RMSE = RMSE / num_samples
    A_pred = A_pred / num_samples
    A_true = A_true / num_samples

    np.savez(
        "../../logs/2p8_extra_features/val_metrics_3.npz",
        ACC=ACC,
        RMSE=RMSE,
        A_pred=A_pred,
        A_true=A_true,
        rank_histogram=rank_histogram_bins,
        clim_pred_histogram=clim_pred_histogram_bins,
        clim_true_histogram=clim_true_histogram_bins,
    )


if __name__ == "__main__":
    compute_acc("../../logs/2p8_extra_features/checkpoints/step_3/epoch=6-step=12782.ckpt", "../../configs/train.yml", rollout_steps=16, ensemble_size=20)
