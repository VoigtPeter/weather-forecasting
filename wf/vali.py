import torch

import xarray as xr
from analysis.plots import _var_descriptor
from wf.scaffold import ForecastModule
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from wf.utils.config import Config
from wf.utils.ensemble import ensemble_batch, reverse_ensemble_batch

MODELS = {
    "vit": ("../configs/ViT.yml", "../logs/ViT_step4ft.ckpt", "ViT"),
    "afno": ("../configs/WeT_afno.yml", "../logs/WeT_afno_step4ft.ckpt", r"WeT$_{\text{AFNO}}$"),
    "afno_gcn": ("../configs/WeT_afno_gcn.yml", "../logs/WeT_afno_gcn_step4ft.ckpt", r"WeT$_{\text{AFNO+}}$"),
    "sfno": ("../configs/WeT_sfno.yml", "../logs/WeT_sfno_step4ft.ckpt", r"WeT$_{\text{SFNO}}$"),
    "sfno_gcn": ("../configs/WeT_sfno_gcn.yml", "../logs/WeT_sfno_gcn_step4ft.ckpt", r"WeT$_{\text{SFNO+}}$"),
}


if __name__ == "__main__":
    var_map = _var_descriptor(MODELS["vit"][0])  # model doesnt matter, as it only looksup dataset confif


    # forecast control
    steps = 20
    idx = 234
    ensemble_size = 1
    keep_last_n: int | None = None  # when we do looong rollouts but only care about the last steps

    # plot control
    times = (0, 3, 7, 19)
    time_unit = "hour"
    #times = (1,)
    var = "U250"
    out_name = "U250_forecast"
    proj = ccrs.EqualEarth(central_longitude=180)
    # proj = ccrs.EqualEarth(central_longitude=180)
    # proj = ccrs.Orthographic(central_latitude=-90)


    cmap_scale = 0.6 if isinstance(proj, ccrs.Orthographic) else 0.9
    # -- make forecasts
    gt_dataset: xr.Dataset | None = None
    pred_datasets = dict()
    show_models = ("afno",)
    for model in show_models:
        config = Config.from_yaml(MODELS[model][0])
        config.dataset.in_memory = False
        module = ForecastModule.load_from_checkpoint(MODELS[model][1], config=config).to("cpu")

        print(f"Predicting with model {model} ...")
        with torch.no_grad():
            x, time_x = module.test_dataset[idx]
            x = x[:, 0, :, :].unsqueeze(0)
            time_x = time_x[0].view(1)
            x = ensemble_batch(x, ensemble_size)
            time_x = ensemble_batch(time_x, ensemble_size)
            pred = reverse_ensemble_batch(module.forecast(x, steps=steps, time=time_x), ensemble_size)[0]

        _steps = steps
        if keep_last_n is not None:
            _steps = keep_last_n
            pred = pred[:, :, -keep_last_n:, :, :]
        if gt_dataset is None:
            gt = torch.cat([module.test_dataset[idx + i][0][:, 1, :, :].unsqueeze(dim=1) for i in range(steps)], dim=1)
            if keep_last_n is not None:
                gt = gt[:, -keep_last_n:, :, :]
            gt_dataset = module.val_dataset.to_xarray(gt, time=np.arange(_steps))
        pred_dataset = module.val_dataset.to_xarray(pred.transpose(1, 0), m=np.arange(ensemble_size),
                                                    time=np.arange(_steps))
        pred_datasets[model] = pred_dataset

    # -- plot forecasts
    titles_h = [(t + 1) * 6 for t in times]
    if keep_last_n is not None:
        titles_h = [(steps - keep_last_n + t + 1) * 6 for t in times]
    if time_unit == "hour":
        titles = [f"{t}h" for t in titles_h]
    elif time_unit == "day":
        titles = [f"{t // 24}d" for t in titles_h]
    elif time_unit == "year":
        titles = [f"{t // 365}y" for t in titles_h]

    vmin, vmax = 1e100, -1e100
    for i, t in enumerate(times):
        sample = gt_dataset[var].sel(time=t)
        if sample.min() < vmin:
            vmin = sample.min()
        if sample.max() > vmax:
            vmax = sample.max()
    if var_map[var].cmap == "RdBu":
        vabsmax = max(abs(vmin), abs(vmax))
        vmin = -vabsmax
        vmax = vabsmax

    fig, axs = plt.subplots(len(times), figsize=(3, (len(times) + 0.5) * 1.7),
                            subplot_kw={"projection": proj},
                            dpi=350,
                            layout='constrained')
    if len(times) == 1:
        axs = [axs]
    for i, t in enumerate(times):
        a = gt_dataset[var].sel(time=t).plot(
            ax=axs[i],
            transform=ccrs.PlateCarree(),
            add_labels=False,
            add_colorbar=False,
            cmap=var_map[var].cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        axs[i].coastlines()
        axs[i].set_title(titles[i])
    cbar = fig.colorbar(
        a,
        ax=axs if len(times) > 1 else axs[0],
        pad=0.04,
        orientation="horizontal",
        label=f"{var_map[var].name} [{var_map[var].unit}]",
        shrink=cmap_scale,
    )
    fig.suptitle(f"Ground truth")
    plt.savefig(f"../{out_name}_plot_{var}_gt.pdf")
    plt.show()

    # model pred
    for model in show_models:
        fig, axs = plt.subplots(len(times), figsize=(3, (len(times) + 0.5) * 1.7),
                                subplot_kw={"projection": proj},
                                dpi=350,
                                layout='constrained')
        if len(times) == 1:
            axs = [axs]
        for i, t in enumerate(times):
            a = pred_datasets[model][var].sel(time=t, m=0).plot(
                ax=axs[i],
                transform=ccrs.PlateCarree(),
                add_labels=False,
                add_colorbar=False,
                cmap=var_map[var].cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            axs[i].coastlines()
            #axs[i].set_title(f"{(t + 1) * 6}h")
            axs[i].set_title(" ")
        cbar = fig.colorbar(a, ax=axs if len(times) > 1 else axs[0], pad=0.04, orientation="horizontal", label=f"{var_map[var].name} [{var_map[var].unit}]", shrink=cmap_scale,)
        fig.suptitle(MODELS[model][2])
        plt.savefig(f"../{out_name}_plot_{var}_{model}.pdf")
        plt.show()


    # model error
    emin, emax = -58, 58  # manual override
    if emin is None or emax is None:
        emin, emax = 1e100, -1e100
        for model in show_models:
            for i, t in enumerate(times):
                sample = (pred_datasets[model][var] - gt_dataset[var]).sel(time=t, m=0)
                if sample.min() < emin:
                    emin = float(sample.min())
                if sample.max() > emax:
                    emax = float(sample.max())
    eabsmax = max(abs(emin), abs(emax))
    emin = -eabsmax
    emax = eabsmax

    for model in show_models:
        fig, axs = plt.subplots(len(times), figsize=(3, (len(times) + 0.5) * 1.7),
                                subplot_kw={"projection": proj},
                                dpi=350,
                                layout='constrained')
        if len(times) == 1:
            axs = [axs]
        for i, t in enumerate(times):
            a = (pred_datasets[model][var] - gt_dataset[var]).sel(time=t, m=0).plot(
                ax=axs[i],
                transform=ccrs.PlateCarree(),
                add_labels=False,
                add_colorbar=False,
                cmap="RdBu",
                vmin=emin,
                vmax=emax,
                rasterized=True,
            )
            axs[i].coastlines()
            #axs[i].set_title(f"{(t + 1) * 6}h")
            axs[i].set_title(" ")

        cbar = fig.colorbar(a, ax=axs if len(times) > 1 else axs[0], pad=0.04, orientation="horizontal",
                            label=f"{var_map[var].name} [{var_map[var].unit}]", shrink=cmap_scale,)
        #fig.suptitle(MODELS[model][2])
        fig.suptitle(" ")
        plt.savefig(f"../{out_name}_plot_{var}_{model}_err.pdf")
        plt.show()
