from typing import NamedTuple

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from utils.config import Config
from wf.data.dataset import ERA5VariableConfig


MODELS = {
    "SFNO_": (
        r"WeT$_{\text{SFNO_}}$",
        "../../logs/WeT_sfno_base_metrics.npz",
    ),
    "SFNO":  (
        r"WeT$_{\text{SFNO}}$",
        "../../logs/WeT_sfno_step4ft_metrics.npz",
    ),
    "SFNO+_": (
        r"WeT$_{\text{SFNO+_}}$",
        "../../logs/WeT_sfno_gcn_base_metrics.npz",
    ),
    "SFNO+": (
        r"WeT$_{\text{SFNO+}}$",
        "../../logs/WeT_sfno_gcn_step4ft_metrics.npz",
    ),
    "AFNO_": (
        r"WeT$_{\text{AFNO_}}$",
        "../../logs/WeT_afno_base_metrics.npz",
    ),
    "AFNO":  (
        r"WeT$_{\text{AFNO}}$",
        "../../logs/WeT_afno_step4ft_metrics.npz",
    ),
    "persistence": (
        r"Persistence",
        "../../logs/persistence_metrics.npz",
    ),
    "climatology": (
        r"Climatology",
        "../../logs/clim_metrics.npz",
    ),
}


class VarLabel(NamedTuple):
    name: str
    unit: str

VAR_LEGEND = {
    "T2M": VarLabel("2m temperature", "$K$"),
    "TP6h": VarLabel("Total precipitation", "$m$"),
    "U10M": VarLabel("10m U wind component", r"$m \cdot s^{-1}$"),
    "V10M": VarLabel("10m V wind component", r"$m \cdot s^{-1}$"),
    "Z...": VarLabel("Geopotential", r"$m^{2} \cdot s^{-2}$"),
    "T...": VarLabel("Temperature", r"$K$"),
    "Q...": VarLabel("Specific humidity", r"$kg \cdot kg^{-1}$"),
    "U...": VarLabel("U component of wind", r"$m \cdot s^{-1}$"),
    "V...": VarLabel("V component of wind", r"$m \cdot s^{-1}$"),
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
                self._var_legend[vk] = VarLabel(f"{src_label.name} ({vk})", src_label.unit)

    def __getitem__(self, item: int | str) -> VarLabel:
        if isinstance(item, int):
            item = self._var_keys[item]
        return self._var_legend[item]

    def idx(self, item: str) -> int:
        return self._var_keys.index(item)


def _var_descriptor(config_path: str):
    config: Config = Config.from_yaml(config_path)
    vars: ERA5VariableConfig = ERA5VariableConfig(**config.dataset.variables)
    return VarMapping(vars.keys())


def _information_noise_diagram_helper(ax: plt.Axes, true_activity: float| tuple[float, float]) -> plt.Axes:
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
    ticks = tick_locator.tick_values(0, 1.25 * a_max)
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
    ax.annotate(f"ACC", (0.75, 0.75), xycoords="axes fraction", fontsize="medium", color="k", rotation=-45)

    ax.plot(0, a_max, "o", ms=9, c="k", mfc="w")                                                          # the perfect forecast
    #ax.plot(noise_error, information, "o-", label=label, **plot_kwargs)
    ax.set_aspect("equal")

    # format ticks & spines
    ax.set_xlim(0, 1.25 * a_max)
    ax.set_ylim(0, 1.25 * a_max)
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

    ax.set_xlabel("noise error")
    ax.set_ylabel("information")
    return ax


def plot_info_noise_acc(
        var: int | str,
        model_config: str,
        models: str | list[str] | None = None,
        ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    if isinstance(models, str):
        models = [models]
    elif models is None:
        models = MODELS.keys()

    ne_p_vals: dict[str, tuple[np.ndarray, np.ndarray]] = dict()
    for model in models:
        assert model in MODELS

        metrics = np.load(MODELS[model][1])
        ACC = metrics["ACC"]
        RMSE = metrics["RMSE"]
        A_pred = metrics["A_pred"]
        A_true = metrics["A_true"]
        p = A_pred * ACC
        IE = np.abs(A_true - p)
        NE = np.sqrt((RMSE ** 2) - (IE ** 2))
        ne_p_vals[model] = (NE, p)

    var_map = _var_descriptor(model_config)
    var_idx = var_map.idx(var) if isinstance(var, str) else var

    markers = ("o", "D", "s", "v")
    mean_true_activity = float(A_true[var_idx, 0].mean())
    _information_noise_diagram_helper(ax, true_activity=mean_true_activity)
    for i, model in enumerate(models):
        ne, p = ne_p_vals[model]
        ax.plot(ne[var_idx, 0], p[var_idx, 0], f"{markers[i]}-", label=MODELS[model][0], markersize=3)

    unit = var_map[var].unit
    ax.set_xlabel(f"Noise error [{unit}]")
    ax.set_ylabel(f"Information [{unit}]")
    ax.set_title(var_map[var].name, pad=20)

    ax.legend(
        frameon=False,
        framealpha=0.0,
        bbox_to_anchor=(0.8, 0.05, 1, 1),
        loc="upper left",
        fontsize=8,
        ncol=1,
    )

    return ax



if __name__ == "__main__":
    plot_info_noise_acc(
        "T2M",
        model_config="../../configs/WeT_afno.yml",
        models=["AFNO", "SFNO", "SFNO+"],
    )
    plt.tight_layout()
    plt.show()
