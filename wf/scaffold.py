from typing import Any, Self

import lightning as L

import torch
from torch.utils.data import DataLoader

from wf.models.wet import WeTConfig, WeT
from wf.data.dataset import WeatherDataset
from wf.models.vit import ViTConfig, ViT
from wf.modules.loss.crps import CRPSLoss
from wf.modules.loss.mse import MSELoss
from wf.utils.config import Config
from wf.utils.ensemble import ensemble_batch, reverse_ensemble_batch


class ForecastModule(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # prepare data
        self.train_dataset, self.val_dataset, _ = WeatherDataset.from_config(config.dataset)

        # static features we don't need to batch
        self.register_buffer("data_latlon", self.train_dataset.latlon, persistent=False)
        self.register_buffer("data_land_sea_mask", self.train_dataset.land_sea_mask, persistent=False)
        self.register_buffer("data_surface_geopotential", self.train_dataset.geopotential_at_surface, persistent=False)

        # prepare model
        if isinstance(self.config.model, ViTConfig):
            self.model = ViT.from_config(
                self.config.model,
                num_vars=self.train_dataset.num_vars,
                field_size=self.train_dataset.field_size
            )
        elif isinstance(self.config.model, WeTConfig):
            self.model = WeT.from_config(
                self.config.model,
                num_surface_vars=self.train_dataset.num_split_vars[0],
                num_atmosphere_vars=self.train_dataset.num_split_vars[1],
                nlat=self.train_dataset.field_size[0],
                nlon=self.train_dataset.field_size[1],
            )
        else:
            raise NotImplementedError()

        # prepare loss
        loss_kwargs = self.config.loss.kwargs if self.config.loss.kwargs is not None else dict()
        if self.config.loss.name.lower() == "mse":
            self.loss = MSELoss(**loss_kwargs)
        elif self.config.loss.name.lower() == "crps":
            self.loss = CRPSLoss(**loss_kwargs)
        else:
            raise NotImplementedError()

        # ensemble options
        self.ensemble_size = self.config.trainer.ensemble_size
        if self.ensemble_size > 1:
            assert self.config.loss.name.lower() in ("crps", ), "Must use ensemble loss when ensemble_size > 1"

        # time options
        self.time_delta = self.train_dataset.time_delta

        # training options
        self.train_forecast_steps: int = 1

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        torch.compiler.cudagraph_mark_step_begin()

        if self.config.model.time_embed is not None:
            state, time = batch
        else:
            state = batch
            time = None
        steps = min(self.train_forecast_steps, state.shape[2] - 1)

        x = state[:, :, 0, :, :]
        x_time = time[:, 0] if time is not None else None
        if self.ensemble_size > 1:
            x = ensemble_batch(x, self.ensemble_size)
            if x_time is not None:
                x_time = ensemble_batch(x_time, self.ensemble_size)
        y_pred = self.forecast(x, steps=steps, time=x_time)
        # -> (batch * ensemble, vars, steps, lat, lon)
        y_true = state[:, :, 1:(steps + 1), :, :]
        if self.ensemble_size > 1:
            y_pred = reverse_ensemble_batch(y_pred, self.ensemble_size)
            # -> (batch, ensemble, vars, steps, lat, lon)

        loss = self.loss(y_pred, y_true)
        self.log("train/loss", loss.item())
        return loss

        """x, y = state[:, :, 0, :, :], state[:, :, 1, :, :]
        x_time = time[:, 0] if time is not None else None
        if self.ensemble_size > 1:
            x = ensemble_batch(x, self.ensemble_size)  # -> (batch * ensemble, ...)
            if x_time is not None:
                x_time = ensemble_batch(x_time, self.ensemble_size)
        y_pred = self.model(x, time=x_time)
        if self.ensemble_size > 1:
            y_pred = reverse_ensemble_batch(y_pred, self.ensemble_size)  # -> (batch, ensemble, ...)
        loss = self.loss(y_pred, y)
        self.log("train/loss", loss.item())
        return loss"""


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        torch.compiler.cudagraph_mark_step_begin()

        if self.config.model.time_embed is not None:
            state, time = batch
        else:
            state = batch
            time = None

        x = state[:, :, 0, :, :]
        x_time = time[:, 0] if time is not None else None
        if self.ensemble_size > 1:
            x = ensemble_batch(x, self.ensemble_size)
            if x_time is not None:
                x_time = ensemble_batch(x_time, self.ensemble_size)
        y_pred = self.forecast(x, steps=state.shape[2] - 1, time=x_time)
        if self.ensemble_size > 1:
            y_pred = reverse_ensemble_batch(y_pred, self.ensemble_size)
        y_true = state[:, :, 1:, :, :]
        for i in range(state.shape[2] - 1):
            if self.ensemble_size > 1:
                y_pred_step_i = y_pred[:, :, :, i, :, :]
            else:
                y_pred_step_i = y_pred[:, :, i, :, :]
            loss = self.loss(y_pred_step_i, y_true[:, :, i, :, :])
            self.log(f"val/loss_step{i+1}", loss.item())

    def forecast(self, x: torch.Tensor, steps: int = 1, time: torch.Tensor | None = None, time_delta: float | None = None) -> torch.Tensor:
        cur_x = x
        cur_time = time
        if time_delta is None:
            time_delta = self.time_delta
        y = list()
        noise_cond = torch.randn((x.shape[0], self.model.noise_dim), device=x.device)
        for i in range(steps):
            next_x = self.model(
                cur_x,
                time=cur_time,
                # static features (same for every sample at any timestep):
                latlon=self.data_latlon,
                land_sea_mask=self.data_land_sea_mask,
                surface_geopotential=self.data_surface_geopotential,
                noise_cond=noise_cond,
            )
            if cur_time is not None:
                cur_time = cur_time + time_delta
            y.append(torch.unsqueeze(next_x, dim=2))
            cur_x = next_x
        return torch.cat(y, dim=2)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.trainer.lr,
            weight_decay=self.config.trainer.weight_decay,
            betas=self.config.trainer.betas,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.trainer.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.trainer.batch_size,
            num_workers=self.config.trainer.num_data_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.trainer.batch_size,
            num_workers=self.config.trainer.num_data_workers,
        )
