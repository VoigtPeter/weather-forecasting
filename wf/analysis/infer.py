import math
import tqdm

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from wf.scaffold import ForecastModule
from wf.utils.config import Config
from wf.utils.ensemble import ensemble_batch, reverse_ensemble_batch


def compute_acc(checkpoint_path: str, config_path: str, rollout_steps: int = 5, ensemble_size: int = 2):
    config = Config.from_yaml(config_path)
    config.dataset.in_memory = False
    config.dataset.extra_features = dict(time=True)

    model: ForecastModule = ForecastModule.load_from_checkpoint(checkpoint_path, config=config)
    model.train_dataset = None

    dataset = model.val_dataset  # TODO: make configurable

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
    num_samples = 0
    loader = DataLoader(dataset, batch_size=1)
    with torch.no_grad():
        for (steps, times) in tqdm.tqdm(loader, total=len(loader)):
            # 1. split inputs / gt_outputs
            x_field = steps[:, :, 0, ...] # TODO: adapt when model supports >1 field as input
            x_time = times[:, 0]
            y_true_field = steps[:, :, 1:, ...].numpy()
            y_time = times[:, 1:]

            # 2. convert step times (y_time) to dayofyear & hourofday (used to determine clim slice)
            #doy = (y_time // 24).detach().cpu().numpy().astype(int) + 1
            #hod = (y_time % 24).detach().cpu().numpy().astype(int)
            #clim = _slice_clim(dataset.clim, hod, doy).compute()

            # 3. make forecast
            x_field = ensemble_batch(x_field.to(model.device), ensemble_size)
            x_time = ensemble_batch(x_time.to(model.device), ensemble_size)
            y_pred_field = reverse_ensemble_batch(model.forecast(x_field, rollout_steps, x_time, dataset.time_delta), ensemble_size)  # un-batch
            y_pred_field = y_pred_field.transpose(2, 1)
            y_pred_field = y_pred_field.transpose(1, 0)

            ds = dataset.to_xarray(y_pred_field, time=y_time[:, 0].cpu().numpy(), number=np.arange(ensemble_size), lead=np.arange(rollout_steps))

            print(ds.values())


            break


    np.savez("../../logs/2p8/val_metrics.npz", ACC=ACC, RMSE=RMSE, A_pred=A_pred, A_true=A_true)


if __name__ == "__main__":
    compute_acc("../checkpoints_3_02/epoch=19-step=18280.ckpt", "../../configs/train.yml", rollout_steps=16, ensemble_size=20)
