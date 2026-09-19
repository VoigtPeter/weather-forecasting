import torch
from wf.scaffold import ForecastModule
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from wf.utils.config import Config
from wf.utils.ensemble import ensemble_batch, reverse_ensemble_batch

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    device = torch.device(device)

    config = Config.from_yaml("../configs/WeT_sfno.yml")
    #config = Config.from_yaml("../configs/wet_train_1p5_sfno.yml")
    #config = Config.from_yaml("../configs/train.yml")
    #config = Config.from_yaml("../configs/train_mhsa.yml")
    config.dataset.in_memory = False
    #test_model_statedict = torch.load("./test_model_5p6_test.pt")
    module = ForecastModule.load_from_checkpoint("../logs/2p8_WeT_sfno/checkpoints/step_4_ft/epoch=0-step=4300.ckpt", config=config).to("cpu")
    #module = ForecastModule.load_from_checkpoint("../logs/1p5_sfno_WeT/checkpoints/step_4_ft/epoch=30-step=56575.ckpt", config=config).to("cpu")
    #module = ForecastModule.load_from_checkpoint("../logs/2p8_sfno_4/checkpoints/step_2_ft/epoch=18-step=34694.ckpt", config=config).to("cpu")
    #module = ForecastModule.load_from_checkpoint("../logs/2p8_mhsa/checkpoints/step_1/epoch=17-step=16434.ckpt", config=config).to("cpu")
    #module.load_state_dict(test_model_statedict, strict=True)

    steps = 100
    idx = 23
    ensemble_size = 1

    with torch.no_grad():
        x, time_x = module.test_dataset[idx]
        x = x[:, 0, :, :].unsqueeze(0)
        time_x = time_x[0].view(1)
        print(x.shape, time_x.shape)
        x = ensemble_batch(x, ensemble_size)
        time_x = ensemble_batch(time_x, ensemble_size)
        print(x.shape, time_x.shape)
        gt = torch.cat([module.val_dataset[idx + i][0][:, 1, :, :].unsqueeze(dim=1) for i in range(steps)], dim=1)
        pred = reverse_ensemble_batch(module.forecast(x.to(device), steps=steps, time=time_x.to(device)), ensemble_size)[0].detach().cpu()

        print(pred.shape, gt.shape)

    """rmse_ensemble = list()
    for i in range(steps):
        pred_i = pred[:, :, i, :, :]
        gt_i = gt[:, i, :, :].unsqueeze(0)
        rmse = torch.sqrt(torch.mean((pred_i - gt_i)**2, dim=(1, 2, 3)))
        rmse_ensemble.append(rmse)
    rmse_ensemble = torch.stack(rmse_ensemble).detach().cpu().numpy().T

    fig, ax = plt.subplots()
    for i in range(ensemble_size):
        ax.plot(np.arange(steps), rmse_ensemble[i])
    plt.show()

    var = 3
    lat, lon = 40, 64
    #lat, lon = 15, 50
    true_forecast = list()
    pred_forecast_mean = list()
    pred_forecast_std = list()
    for i in range(steps):
        pred_i = pred[:, var, i, lat, lon]
        gt_i = gt[var, i, lat, lon].unsqueeze(0)
        true_forecast.append(gt_i.item())
        pred_forecast_mean.append(pred_i.mean().item())
        pred_forecast_std.append(pred_i.std().item())
    true_forecast = np.array(true_forecast)
    pred_forecast_mean = np.array(pred_forecast_mean)
    pred_forecast_std = np.array(pred_forecast_std)

    fig, ax = plt.subplots()
    ax.plot(np.arange(steps), true_forecast, label="True", linewidth=2, c="k")
    ax.plot(np.arange(steps), pred_forecast_mean, label="Predicted", linestyle="dashed", linewidth=2, c="tab:blue")
    ax.fill_between(np.arange(steps), pred_forecast_mean - pred_forecast_std, pred_forecast_mean + pred_forecast_std, color="tab:blue", alpha=0.2)
    plt.show()





    #x = module.val_dataset[idx][:, 0, :, :].unsqueeze(0)
    #gt = torch.cat([module.val_dataset[idx+i][:, 1, :, :].unsqueeze(dim=1) for i in range(steps)], dim=1)
    #pred = module.forecast(x, steps=steps)[0]

    gt_dataset = module.val_dataset.to_xarray(gt, time=np.arange(steps))
    pred_dataset = module.val_dataset.to_xarray(pred.transpose(1, 0), m=np.arange(ensemble_size), time=np.arange(steps))

    times = (0, 1, 2, 3, 4, 5)

    fig, axs = plt.subplots(len(times), figsize=(6, len(times)*6),
        subplot_kw={"projection": ccrs.EqualEarth()}
    )
    for i, t in enumerate(times):
        (pred_dataset["T2M"] - gt_dataset["T2M"]).sel(time=t, m=0).plot(ax=axs[i], transform=ccrs.PlateCarree(), cmap="RdBu", cbar_kwargs={"shrink": 0.6, "orientation": "horizontal"})
        axs[i].coastlines()
    plt.show()


    times = (0, 1, 2, 3, 4, 5, 6)

    fig, axs = plt.subplots(len(times), figsize=(6, len(times)*6),
        subplot_kw={"projection": ccrs.EqualEarth()}
    )
    for i, t in enumerate(times):
        pred_dataset["T2M"].sel(time=t, m=0).plot(ax=axs[i], transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs={"shrink": 0.6, "orientation": "horizontal"})
        axs[i].coastlines()
    plt.show()


    # animation"""
    gt_dataset = module.val_dataset.to_xarray(gt, time=np.arange(steps))
    pred_dataset = module.val_dataset.to_xarray(pred.transpose(1, 0), m=np.arange(ensemble_size), time=np.arange(steps))

    var = "Q500"
    north_pole = ccrs.Orthographic(0, 90)

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.EqualEarth()})
    def update(frame: int):
        #ax.clear()
        pred_dataset[var].sel(time=frame, m=0).plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, cmap="viridis")
        #gt_dataset[var].sel(time=frame).plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, cmap="viridis")
        ax.coastlines()
        #ax.gridlines()

    ani = animation.FuncAnimation(fig=fig, func=update, frames=steps, interval=100)
    ani.save("./anim_mhsa.gif")
