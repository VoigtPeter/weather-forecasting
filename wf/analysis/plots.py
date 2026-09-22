import os
from typing import NamedTuple

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from IPython.core.pylabtools import figsize
from matplotlib.ticker import MaxNLocator

from utils.config import Config
from wf.data.dataset import ERA5VariableConfig


MODELS = {
    "SFNO":  (
        r"WeT$_{\text{SFNO}}$",
        "../../logs_final/WeT_sfno_step4ft_metrics.npz",
        ["T2M", "TP6h", 'T850', 'Z500'],  # subset vars
    ),
    "SFNO+": (
        r"WeT$_{\text{SFNO+}}$",
        "../../logs_final/WeT_sfno_gcn_step4ft_metrics.npz",
        ["T2M", "TP6h", 'T850', 'Z500'],  # subset vars
    ),
    "AFNO+1": (
        r"WeT$_{\text{AFNO+ (1)}}$",
        "../../logs_final/WeT_afno_gcn_step1_metrics.npz",
        ["T2M", "TP6h", 'T850', 'Z500'],  # subset vars
    ),
    "AFNO":  (
        r"WeT$_{\text{AFNO}}$",
        "../../logs_final/WeT_afno_step4ft_metrics.npz",
        ["T2M", "TP6h", 'T850', 'Z500'],  # subset vars
    ),
    "vit":  (
        r"ViT",
        "../../logs_final/ViT_step4ft_metrics.npz",
        ["T2M", "TP6h", 'T850', 'Z500'],  # subset vars
    ),
    "AFNO+":  (
        r"WeT$_{\text{AFNO+}}$",
        "../../logs_final/WeT_afno_gcn_step4ft_metrics.npz",
        ["T2M", "TP6h", 'T850', 'Z500'],  # subset vars
    ),
    "persistence": (
        r"Persistence",
        "../../logs_final/persistence_metrics.npz",
        ["T2M", "TP6h", 'T850', 'Z500'],  # subset vars
    ),
    "climatology": (
        r"Climatology",
        "../../logs_final/clim_metrics.npz",
        ["T2M", "TP6h", 'T850', 'Z500'],  # subset vars
    ),
}


class VarLabel(NamedTuple):
    name: str
    unit: str
    cmap: str | None = None

VAR_LEGEND = {
    "T2M": VarLabel("2m temperature", "$K$", "viridis"),
    "TP6h": VarLabel("Total precipitation", "$m$"),
    "U10M": VarLabel("10m U wind component", r"$m \cdot s^{-1}$", "RdBu"),
    "V10M": VarLabel("10m V wind component", r"$m \cdot s^{-1}$", "RdBu"),
    "Z...": VarLabel("Geopotential", r"$m^{2} \cdot s^{-2}$", "viridis"),
    "T...": VarLabel("Temperature", r"$K$", "viridis"),
    "Q...": VarLabel("Specific humidity", r"$kg \cdot kg^{-1}$", "viridis"),
    "U...": VarLabel("U component of wind", r"$m \cdot s^{-1}$", "RdBu"),
    "V...": VarLabel("V component of wind", r"$m \cdot s^{-1}$", "RdBu"),
}

class VarMapping:
    def __init__(self, var_keys: list[str]) -> None:
        self._var_keys = var_keys
        self._var_legend = dict()
        for vk in self._var_keys:
            if vk in VAR_LEGEND:
                self._var_legend[vk] = VAR_LEGEND[vk]
            elif f"{vk[0]}..." in VAR_LEGEND:
                src_label = VAR_LEGEND[f"{vk[0]}..."]
                self._var_legend[vk] = VarLabel(f"{src_label.name} ({vk})", src_label.unit, src_label.cmap)

    def __getitem__(self, item: int | str) -> VarLabel:
        if isinstance(item, int):
            item = self._var_keys[item]
        return self._var_legend[item]

    def idx(self, item: str, subset: list[str] | None = None) -> int:
        if subset is not None:
            return subset.index(item)
        return self._var_keys.index(item)


def _var_descriptor(config_path: str):
    config: Config = Config.from_yaml(config_path)
    vars: ERA5VariableConfig = ERA5VariableConfig(**config.dataset.variables)
    return VarMapping(vars.keys())


def _information_noise_diagram_helper(ax: plt.Axes, true_activity: float| tuple[float, float], sci_ticks: bool = False) -> plt.Axes:
    '''Information, noise, and correlation diagram after Bonavita and Geer (2026), Figure 3.

    A forecast is a point at (noise error, information).
    Its distance from the origin is the forecast activity, the angle to the vertical axis
    has cosine equal to the anomaly correlation, and its distance to the point (0, true_activity)
    is the error.  The dashed quarter circle is the locus of forecasts with the true activity;
    the dashed half circle is the locus of lowest error for a given correlation.
    '''
    a_min, a_max = None, true_activity
    if isinstance(true_activity, tuple):
        a_min, a_max = true_activity

    # setup ticks
    minor_tick_bins = 3
    tick_locator = MaxNLocator(nbins=5)
    ticks = tick_locator.tick_values(0, 1.1 * a_max)
    mticks = np.linspace(ticks[0], ticks[-1], (len(ticks) * minor_tick_bins) - (minor_tick_bins - 1))
    tick_max = ticks[-1]

    # plot grid
    grid_color = "tab:blue"
    grid_linestyle = (0, (3, 5))
    grid_linewidth = 0.8
    grid_alpha = 0.4
    theta = np.linspace(0, np.pi / 2, 200)
    for grid_xy in ticks:
        ax.plot(grid_xy * np.sin(theta), grid_xy * np.cos(theta), linestyle=grid_linestyle, linewidth=grid_linewidth, c=grid_color, alpha=grid_alpha, zorder=-10)

    ax.plot(a_max * np.sin(theta), a_max * np.cos(theta), "k--", lw=0.8)                       # forecast activity == true activity
    ax.plot(a_max / 2 * np.sin(2 * theta), a_max / 2 + a_max / 2 * np.cos(2 * theta), "k:", lw=0.8)  # lowest error for a given correlation
    if a_min is not None:
        ax.plot(a_min * np.sin(theta), a_min * np.cos(theta), "k--", lw=0.8)  # forecast activity == true activity
        ax.plot(a_min / 2 * np.sin(2 * theta), a_min / 2 + a_min / 2 * np.cos(2 * theta), "k:",
                lw=0.8)  # lowest error for a given correlation

    # plot ACC grid
    tick_font = ax.get_xticklabels()[0].get_fontproperties()
    for acc in (0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        ang = np.arccos(acc)
        ax.plot([ticks[1] * np.sin(ang), tick_max * np.sin(ang)], [ticks[1] * np.cos(ang), tick_max * np.cos(ang)], c=grid_color, linestyle=grid_linestyle, linewidth=grid_linewidth, alpha=grid_alpha, zorder=-10)
        #ax.annotate(f"{acc}", (tick_max * np.sin(ang), tick_max * np.cos(ang)), fontproperties=tick_font, bbox=dict(boxstyle='square,pad=0', fc='none', ec='none'),)
        ax.text(tick_max * np.sin(ang), tick_max * np.cos(ang), f"{acc}", fontsize="small", bbox=None)
    ax.annotate(f"ACC", (0.9, 0.55), xycoords="axes fraction", fontsize="medium", color="k", rotation=-55)

    ax.plot(0, a_max, "o", ms=9, c="k", mfc="w")                                                          # the perfect forecast
    #ax.plot(noise_error, information, "o-", label=label, **plot_kwargs)
    ax.set_aspect("equal")

    # format ticks & spines
    ax.set_xlim(0, 1.1 * a_max)
    ax.set_ylim(0, 1.1 * a_max)
    ax.set_xticks(ticks)
    ax.set_xticks(mticks, minor=True)
    ax.set_yticks(ticks)
    ax.set_yticks(mticks, minor=True)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set(linewidth=1.5)
    ax.spines.bottom.set(linewidth=1.5)
    ax.tick_params(which='major', direction="out", width=1.5, length=4)
    ax.tick_params(which='minor', direction="out", width=1.0, length=3)

    if sci_ticks:
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        tx = ax.xaxis.get_offset_text()
        ty = ax.yaxis.get_offset_text()
        tx.set_fontsize(9)
        ty.set_fontsize(9)
        ty.set_x(-0.06)

    ax.set_xlabel("noise")
    ax.set_ylabel("information")
    return ax


def plot_info_noise_acc(
        var: str,
        model_config: str,
        models: str | list[str] | None = None,
        sci_ticks: bool = False,
        max_step: int = None,
        ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    if isinstance(models, str):
        models = [models]
    elif models is None:
        models = MODELS.keys()
    var_map = _var_descriptor(model_config)
    markers = ("o", "D", "s", "v", "X", "P")

    ne_p_vals: dict[str, tuple[np.ndarray, np.ndarray]] = dict()
    for model in models:
        assert model in MODELS
        _, model_path, model_subset = MODELS[model]

        metrics = np.load(model_path)
        ACC = metrics["ACC"]
        RMSE = metrics["RMSE"]
        A_pred = metrics["A_pred"]
        A_true = metrics["A_true"]
        p = A_pred * ACC
        IE = np.abs(A_true - p)
        NE = np.sqrt((RMSE ** 2) - (IE ** 2))
        ne_p_vals[model] = (NE, p)

        var_idx = var_map.idx(var, subset=model_subset)

    mean_true_activity = float(A_true[var_idx, 0].mean())
    _information_noise_diagram_helper(ax, true_activity=mean_true_activity, sci_ticks=sci_ticks)
    for i, model in enumerate(models):
        model_name, _, model_subset = MODELS[model]
        ne, p = ne_p_vals[model]
        var_idx = var_map.idx(var, subset=model_subset)

        x = ne[var_idx, 0]
        y = p[var_idx, 0]
        if max_step is not None:
            x = ne[var_idx, 0, :max_step]
            y = p[var_idx, 0, :max_step]

        ax.plot(x, y, f"{markers[i]}-", label=model_name, markersize=5)

    unit = var_map[var].unit
    ax.set_xlabel(f"Noise [{unit}]")
    ax.set_ylabel(f"Information [{unit}]")
    ax.set_title(var_map[var].name, pad=20)

    leg = ax.legend(
        frameon=True,
        framealpha=0.8,
        facecolor='white',
        edgecolor='white',
        bbox_to_anchor=(0.8, 0.08, 1, 1),
        loc="upper left",
        fontsize=8,
        ncol=1,
    )
    #leg.set_visible(False)

    return ax


def plot_lead_time_rmse(
        var: str,
        model_config: str,
        models: str | list[str] | None = None,
        ax: plt.Axes | None = None,
        show_std: bool = False,
        as_days: bool = False,
        ) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    if isinstance(models, str):
        models = [models]
    elif models is None:
        models = MODELS.keys()
    var_map = _var_descriptor(model_config)

    def _plot(model: str, legend: bool = True, std: bool = True, alpha: float = 0.2, _ret_max_val: bool = False, **plot_kwargs) -> int | tuple[int, float]:
        model_name, model_path, model_subset = MODELS[model]
        metrics = np.load(model_path)
        rmse = metrics["RMSE"][var_map.idx(var, subset=model_subset), 0]
        rmse_std = metrics["RMSE_std"][var_map.idx(var, subset=model_subset), 0]
        x = np.arange(rmse.shape[0])
        line = ax.plot(x, rmse, label=model_name if legend else None, **plot_kwargs)
        c = line[0].get_color()
        if std:
            ax.fill_between(x, rmse - rmse_std, rmse + rmse_std, alpha=alpha, facecolor=c)
        if _ret_max_val:
            return len(x), float((rmse + rmse_std).max()), line[0]
        return len(x)

    # clim & persistence
    c_steps, c_ymax, c_line = _plot("climatology", legend=False, alpha=0.1, c="#444", linestyle="dashdot", _ret_max_val=True)
    p_steps, p_ymax, p_line = _plot("persistence", legend=False, alpha=0.1, c="#444", linestyle="dotted", _ret_max_val=True)
    steps = max(c_steps, p_steps)
    ymax = max(c_ymax, p_ymax)

    # models
    for model in models:
        steps = max(_plot(model, alpha=0.2, std=show_std), steps)

    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    if not as_days:
        ticks = np.arange(steps+1, step=4) - 1
        ticks[0] = 0
        ax.set_xticks(
            ticks,
            labels=[f"{int(t)}" for t in (np.array(ticks) + 1) * 6],
        )
    else:
        ticks = np.arange(steps + 1, step=12) - 1
        ticks[0] = 0
        ax.set_xticks(
            ticks,
            labels=[f"{int(t)}" for t in (np.array(ticks) + 1) * 0.25],
        )
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set(linewidth=1.5)
    ax.spines.bottom.set(linewidth=1.5)
    ax.tick_params(which='major', direction="out", width=1.5, length=4)
    ax.tick_params(which='minor', direction="out", width=1.0, length=3)
    ax.set_xlabel("Lead time [$h$]")
    if as_days:
        ax.set_xlabel("Lead time [$d$]")
    var = var_map[var]
    ax.set_ylabel(f"RMSE [{var.unit}]")
    ax.set_title(var.name)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_xlim(0, steps-1)

    sub_legend = ax.legend(
        handles=[c_line, p_line],
        labels=["Climatology", "Persistence"],
        frameon=False,
        framealpha=0.0,
        loc="lower center",
        fontsize=9,
        ncol=1,
    )
    ax.legend(
        frameon=False,
        framealpha=0.0,
        loc="lower right",
        fontsize=9,
        ncol=1,
    )
    ax.add_artist(sub_legend)

    return ax


def plot_rank_histogram(
        models: list[str],
        steps: list[int],
        ) -> plt.Axes:
    fig, axes = plt.subplots(ncols=len(steps), figsize=(3 * len(steps), 2.5), dpi=300)

    ymax = 0
    bins = 0
    for model in models:
        model_name, model_path, _ = MODELS[model]
        metrics = np.load(model_path)
        time_rank_hist = (
                metrics["rank_histogram"].sum(axis=0) /
                metrics["rank_histogram"].sum(axis=0).sum(axis=1).reshape(-1, 1)
        )

        for ax, step in zip(axes, steps):
            data = time_rank_hist[step]
            bins = data.shape[0]
            ax.step(np.arange(data.shape[0]), data, where="mid", label=f"{model_name}")
            ymax = max(ymax, np.max(data))

    for ax, step in zip(axes, steps):
        ax.set_ylim(0, ymax * 1.05)
        ax.set_xlim(0, bins - 1)
        ticks = np.arange(0, bins, step=2)
        ax.set_xticks(ticks, labels=[str(int(t + 1)) for t in ticks])
        ax.axhline(1.0 / bins, color="k", linestyle="--", zorder=-10)
        ax.spines.top.set_visible(False)
        ax.set_title(f"{(step + 1) * 6}h", pad=0)
        ax.spines.left.set(linewidth=1.5)
        ax.spines.bottom.set(linewidth=1.5)
        ax.spines.right.set(linewidth=1.5)
        ax.tick_params(which='major', direction="out", width=1.5, length=4)

    for ax in axes[1:]:
        ax.yaxis.set_tick_params(which='both', labelbottom=False)

    axes[0].legend(
        frameon=False,
        framealpha=0.0,
        loc="lower right",
        fontsize=9,
        ncol=1 if len(models) <= 3 else 2,
    )
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel(r"$p(\text{Rank})$")

    return axes


def plot_train_losses(
        model: str,
        log_dir: str,
        loss_name: str,
        legend: bool = True,
        hide_yaxis: bool = False,
        ) -> None:
    phase_1_logs = pd.read_csv(os.path.join(log_dir, "version_0/metrics.csv"), sep=",")
    phase_2_logs = pd.read_csv(os.path.join(log_dir, "version_1/metrics.csv"), sep=",")

    def _get_col(df: pd.DataFrame, col: str) -> np.ndarray:
        return df[["step", col]][pd.notna(df[col])].to_numpy()

    # phase 1
    p1_train_loss = _get_col(phase_1_logs, "train/loss")
    p1_val_loss_1 = _get_col(phase_1_logs, "val/loss_step1")
    p1_val_loss_2 = _get_col(phase_1_logs, "val/loss_step2")
    p1_val_loss_3 = _get_col(phase_1_logs, "val/loss_step3")

    # phase 2
    p2_train_loss = _get_col(phase_2_logs, "train/loss")
    p2_val_loss_1 = _get_col(phase_2_logs, "val/loss_step1")
    p2_val_loss_2 = _get_col(phase_2_logs, "val/loss_step2")
    p2_val_loss_3 = _get_col(phase_2_logs, "val/loss_step3")

    # time
    p1_time = _get_col(phase_1_logs, "time")
    p2_time = _get_col(phase_2_logs, "time")
    p1_time = p1_time[-1, 1] - p1_time[0, 1]
    p2_time = p2_time[-1, 1] - p2_time[0, 1]
    total_seconds = p1_time + p2_time
    print("total_seconds:", total_seconds, "total_hours:", total_seconds / 3600)

    tmp_all = np.concatenate((
        p1_train_loss, p1_val_loss_1, p1_val_loss_2, p1_val_loss_3,
        p2_train_loss, p2_val_loss_1, p2_val_loss_2, p2_val_loss_3,
    ), axis=0)
    y_min = float(tmp_all[:, 1].min())
    y_max = float(tmp_all[:, 1].max())
    x1_max = float(p1_train_loss[:, 0].max())
    x2_max = float(p2_train_loss[:, 0].max())

    y_min = 0.05
    y_max = 0.25

    train_c = "k"
    train_lw = 1
    train_alpha = 0.3

    val_c = (
        "indigo",
        "darkviolet",
        "violet"
    )
    val_lw = 1.5

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(5, 3), gridspec_kw={'width_ratios': [1, x2_max / x1_max]}, dpi=300)

    ax1.plot(p1_val_loss_3[:, 0], p1_val_loss_3[:, 1], label="Validation @ 18h", c=val_c[0], linewidth=val_lw)
    ax1.plot(p1_val_loss_2[:, 0], p1_val_loss_2[:, 1], label="Validation @ 12h", c=val_c[1], linewidth=val_lw)
    ax1.plot(p1_val_loss_1[:, 0], p1_val_loss_1[:, 1], label="Validation @ 6h", c=val_c[2], linewidth=val_lw)
    ax1.plot(p1_train_loss[:, 0], p1_train_loss[:, 1], label="Train", alpha=train_alpha, zorder=-10, c=train_c, linewidth=train_lw)

    p1_min_idx = np.argmin(p1_val_loss_3[:, 1])
    p1_min_x = p1_val_loss_3[p1_min_idx, 0]
    p1_min_y = p1_val_loss_3[p1_min_idx, 1]
    pp1 = ax1.scatter([p1_min_x], [p1_min_y], marker="o", s=40, lw=1, edgecolor=val_c[0], fc="None", label="min. loss")
    #ax1.arrow(p1_min_x, p1_min_y*1.3, 0, p1_min_y - p1_min_y*1.3, arrowprops=dict(arrowstyle="->"))
    #ax1.annotate("best", xytext=(p1_min_x, p1_min_y*1.3), xy=(p1_min_x, p1_min_y),
    #            arrowprops=dict(arrowstyle="->", color=val_c[0], linewidth=val_lw))

    ax2.plot(p2_val_loss_3[:, 0], p2_val_loss_3[:, 1], c=val_c[0], linewidth=val_lw)
    ax2.plot(p2_val_loss_2[:, 0], p2_val_loss_2[:, 1], c=val_c[1], linewidth=val_lw)
    ax2.plot(p2_val_loss_1[:, 0], p2_val_loss_1[:, 1], c=val_c[2], linewidth=val_lw)
    ax2.plot(p2_train_loss[:, 0], p2_train_loss[:, 1], alpha=train_alpha, zorder=-10, c=train_c, linewidth=train_lw)
    p2_min_idx = np.argmin(p2_val_loss_3[:, 1])
    p2_min_x = p2_val_loss_3[p2_min_idx, 0]
    p2_min_y = p2_val_loss_3[p2_min_idx, 1]
    pp2 = ax2.scatter([p2_min_x], [p2_min_y], marker="o", s=40, lw=1, edgecolor=val_c[0], fc="None")


    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    ax1.set_xlim(0, x1_max)
    ax2.set_xlim(0, x2_max)
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax1.spines.left.set(linewidth=1.5)
    ax1.spines.bottom.set(linewidth=1.5)
    ax2.spines.bottom.set(linewidth=1.5)
    ax1.tick_params(which='major', direction="out", width=1.5, length=4)
    ax2.tick_params(which='major', direction="out", width=1.5, length=4)

    ax1.set_ylabel(loss_name)
    ax1.set_xlabel(r"Phase 1 [$step$]")
    ax2.set_xlabel(r"Phase 2 [$step$]")

    if legend:
        fig.legend(
            frameon=False,
            loc="center right",
            fontsize=9,
            bbox_to_anchor=(1., 0.7)
        )
    if hide_yaxis:
        ax1.yaxis.set_ticklabels([])
        ax1.set_ylabel(" ")
    #fig.add_artist(leg)
    pp1.set_clip_on(False)
    pp2.set_clip_on(False)

    fig.suptitle(MODELS[model][0])



if __name__ == "__main__":
    plot_train_losses(
        model = "AFNO",
        log_dir=f"../../logs_final/SFNO_P",
        loss_name="fCRPS",
        legend=True,
        hide_yaxis=False,
    )
    plt.tight_layout()
    #plt.savefig(f"../../losses_showcase_afno.pdf")
    plt.show()

    #plot_info_noise_acc(
    #    "T850",
    #    model_config="../../configs/WeT_afno.yml",
    #    models=["vit", "AFNO", "AFNO+", "SFNO", "SFNO+", "persistence"],
    #    #sci_ticks=True,
    #    max_step=20
    #)
    #plt.tight_layout()
    #plt.savefig("../../info_noise_T850.pdf")
    #plt.show()

    #plot_lead_time_rmse(
    #    "Z500",
    #    model_config="../../configs/WeT_afno.yml",
    #    models=[ "AFNO", "AFNO+"],
    #    show_std=True,
    #    as_days=True,
    #)
    #plt.tight_layout()
    #plt.savefig("../../lead_RMSE_Z500_30d.pdf")
    #plt.show()

    #plot_rank_histogram(
    #    models=["vit", "AFNO", "AFNO+", "SFNO", "SFNO+"],
    #    steps=[0, 7, 19],
    #)
    #plt.tight_layout()
    #plt.savefig("../../rank_histogram_all.pdf")
    #plt.show()