import os
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import torch
from ldns.utils.eval_utils import average_rates, correlation_matrix
from sklearn.metrics import roc_curve


def cm2inch(cm, cm2=None, INCH=2.54):
    """Convert cm to inch"""
    if isinstance(cm, tuple):
        return tuple(i / INCH for i in cm)
    elif cm2 is not None:
        return cm / INCH, cm2 / INCH
    else:
        return cm / INCH


@dataclass
class FigureLayout:
    """
    Dataclass to define the layout of a figure. That is, width and font size.

    Args:
        width_in_pt: Width of the figure in pt.
        width_grid: Width of the grid in which the figure is placed.
        base_font_size: Base font size of the figure.
        scale_factor: Scale factor of the font size.
            This exposes the factor by which the Figure will be downscaled when included document.
    """

    width_in_pt: float
    width_grid: int
    base_font_size: int = 10
    scale_factor: float = 1.0

    def get_grid_in_inch(self, w_grid, h_grid):
        pt_to_inch = 1 / 72
        assert w_grid <= self.width_grid
        return (
            (w_grid / self.width_grid) * self.width_in_pt * pt_to_inch,
            (h_grid / self.width_grid) * self.width_in_pt * pt_to_inch,
        )

    def get_rc(self, w_grid, h_grid):
        return {
            "figure.figsize": self.get_grid_in_inch(w_grid, h_grid),
            "font.size": self.base_font_size * self.scale_factor,
        }


def basic_plotting(
    fig,
    ax,
    x_label=None,
    x_label_fontsize="medium",
    y_label=None,
    y_label_fontsize="medium",
    x_lim=None,
    y_lim=None,
    x_ticks=None,
    y_ticks=None,
    x_ticklabels=None,
    y_ticklabels=None,
    x_axis_visibility=None,
    y_axis_visibility=None,
):
    """
    Provide some basic plotting functionality.

    Args:
        fig: Figure object.
        ax: Axis object.
        x_label: Label for the x-axis.
        x_label_fontsize: Fontsize of the x-axis label.
        y_label: Label for the y-axis.
        y_label_fontsize: Fontsize of the y-axis label.
        x_lim: Limits of the x-axis.
        y_lim: Limits of the y-axis.
        x_ticks: Ticks of the x-axis.
        y_ticks: Ticks of the y-axis.
        x_ticklabels: Ticklabels of the x-axis.
        y_ticklabels: Ticklabels of the y-axis.
        x_axis_visibility: Visibility of the x-axis.
        y_axis_visibility: Visibility of the y-axis.
    """

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=x_label_fontsize)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=y_label_fontsize)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if x_axis_visibility is not None:
        ax.get_xaxis().set_visible(x_axis_visibility)
        try:
            ax.spines["bottom"].set_visible(x_axis_visibility)
        except KeyError:
            print("No bottom spine exists!")
    if y_axis_visibility is not None:
        ax.get_yaxis().set_visible(y_axis_visibility)
        try:
            ax.spines["left"].set_visible(y_axis_visibility)
        except KeyError:
            print("No left spine exists!")

    return fig, ax


def plot_sd(
    fig,
    ax,
    arr_one,
    arr_two,
    fs,
    nperseg,
    agg_function=np.median,
    color_one="red",
    color_two="blue",
    label_one="gt",
    label_two="rec",
    with_quantiles=False,
    alpha=0.1,
    lower_quantile=0.25,
    upper_quantile=0.75,
    alpha_boundary=1.0,
    x_ss=slice(None),
    add_legend=False,
    lw=1,
):
    """
    Plot the spectral density of two arrays with pointwise uncertainty.

    Args:
        fig: Figure object.
        ax: Axis object.
        arr_one: First array.
        arr_two: Second array.
        fs: Sampling frequency.
        nperseg: Number of samples per segment.
        agg_function: Aggregation function.
        color_one: Color for the first array.
        color_two: Color for the second array.
        with_quantiles: Whether to plot the quantiles.
        alpha: Alpha value for the quantiles.
        lower_quantile: Lower quantile.
        upper_quantile: Upper quantile.
        alpha_boundary: Alpha value for the percentile boundary.
        x_ss: Frequencies to plot.
    """

    ff_one, Pxy_one = signal.csd(arr_one, arr_one, axis=1, nperseg=nperseg, fs=fs)
    ff_two, Pxy_two = signal.csd(arr_two, arr_two, axis=1, nperseg=nperseg, fs=fs)
    if with_quantiles:
        ax.fill_between(
            ff_one[x_ss],
            np.quantile(Pxy_one, lower_quantile, axis=0)[x_ss],
            np.quantile(Pxy_one, upper_quantile, axis=0)[x_ss],
            color=color_one,
            alpha=alpha,
        )
        ax.fill_between(
            ff_two[x_ss],
            np.quantile(Pxy_two, lower_quantile, axis=0)[x_ss],
            np.quantile(Pxy_two, upper_quantile, axis=0)[x_ss],
            color=color_two,
            alpha=alpha,
        )
        ax.loglog(
            ff_one[x_ss],
            np.quantile(Pxy_one, lower_quantile, axis=0)[x_ss],
            color=color_one,
            alpha=alpha_boundary,
            label=label_one,
            lw=lw,
        )
        ax.loglog(
            ff_one[x_ss],
            np.quantile(Pxy_one, upper_quantile, axis=0)[x_ss],
            color=color_one,
            alpha=alpha_boundary,
            lw=lw,
        )
        ax.loglog(
            ff_two[x_ss],
            np.quantile(Pxy_two, lower_quantile, axis=0)[x_ss],
            color=color_two,
            alpha=alpha_boundary,
            label=label_two,
            lw=lw,
        )
        ax.loglog(
            ff_two[x_ss],
            np.quantile(Pxy_two, upper_quantile, axis=0)[x_ss],
            color=color_two,
            alpha=alpha_boundary,
            lw=lw,
        )

    ax.loglog(
        ff_one[x_ss],
        agg_function(Pxy_one, axis=0)[x_ss],
        color=color_one,
        label=label_one,
        lw=lw,
    )
    ax.loglog(
        ff_two[x_ss],
        agg_function(Pxy_two, axis=0)[x_ss],
        color=color_two,
        label=label_two,
        lw=lw,
    )
    # set unique legend
    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    return fig, ax


def plot_phase_line(fig, ax, bin_phase, pac, color="black"):
    """
    Polar plot used for the phase-amplitude coupling.

    Args:
        fig: Figure object.
        ax: Axis object.
        bin_phase: Phase bins.
        pac: Phase-amplitude coupling.
        color: Color of the plotted line.
    """

    ax.plot(np.append(bin_phase, bin_phase[0]), np.append(pac, pac[0]), color=color)
    return fig, ax


def polar_hist(
    fig,
    ax,
    values,
    p_bins,
    grid=10,
    fill_alpha=0.2,
    fillcolor="red",
    spinecolor="black",
    full_spines=False,
):
    """
    Polar histogram used for phase-count coupling.

    Args:
        fig: Figure object.
        ax: Axis object.
        values: Values to plot.
        p_bins: Phase bins.
        grid: Grid size.
        fill_alpha: Alpha value for the fill.
        fillcolor: Color for the fill.
        spinecolor: Color for the spines.
        full_spines: Whether to plot the full spines.
    """
    num_values = len(values)
    assert num_values + 1 == len(p_bins)
    for idx in range(num_values):
        ax.fill_between(
            np.linspace(p_bins[idx], p_bins[idx + 1], grid),
            np.zeros(grid),
            values[idx] * np.ones(grid),
            color=fillcolor,
            alpha=fill_alpha,
        )
        ax.plot(
            np.linspace(p_bins[idx], p_bins[idx + 1], grid),
            values[idx] * np.ones(grid),
            color=spinecolor,
        )
        if full_spines:
            ax.plot([p_bins[idx], p_bins[idx]], [0.0, values[idx]], color=spinecolor)
            ax.plot(
                [p_bins[idx + 1], p_bins[idx + 1]], [0.0, values[idx]], color=spinecolor
            )
        else:
            ax.plot(
                [p_bins[idx], p_bins[idx]],
                [values[idx - 1], values[idx]],
                color=spinecolor,
            )
    return fig, ax


def plot_overlapping_signal(fig, ax, sig, colors=["C0"]):
    """
    Plot signals of a given signal array.

    Args:
        fig: Figure object.
        ax: Axis object.
        sig: Signal array.
        colors: Colors for the individual signal channels.
    """

    if len(colors) == 1:
        colors = len(sig) * colors
    else:
        assert len(colors) == len(sig)

    for chan, col in zip(sig, colors):
        ax.plot(chan, color=col)

    return fig, ax


def plot_density(fig, ax, values, x_range, bw_method=None, d_alpha=0.2, color="C0"):
    """
    Plot the Gaussian kernel density estimate of a given array.

    Args:
        fig: Figure object.
        ax: Axis object.
        values: Values to plot.
        x_range: Values to evalualte the KDE at.
        bw_method: Bandwidth method.
        d_alpha: Alpha value for the fill.
        color: Color of the plot.
    """

    kde = gaussian_kde(values, bw_method=bw_method)
    ax.fill_between(x_range, kde(x_range), alpha=d_alpha, color=color)
    ax.plot(x_range, kde(x_range), color=color)
    return fig, ax


def plot_roc_curve(fig, ax, y_true, y_score, rand_base_col=None):
    """
    Plot the ROC curve.

    Args:
        fig: Figure object.
        ax: Axis object.
        y_true: True labels.
        y_score: Predicted labels.
        rand_base_col: Color for the random baseline. If None, no baseline is plotted.
    """

    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    if rand_base_col is not None:
        ax.plot(
            np.linspace(0.0, 1.0, 100),
            np.linspace(0.0, 1.0, 100),
            color=rand_base_col,
            linestyle="--",
        )
    return fig, ax


# -------------------------- Auguste added plots --------------------------


def plot_losses(return_dict, save=False, save_path=None, cutoff=10):
    """plot train and val losses"""
    plt.figure(figsize=cm2inch((12, 4)))
    plt.subplot(1, 3, 1)
    plt.plot(return_dict["train_losses"][cutoff:], label="train")
    plt.plot(return_dict["val_losses"][cutoff:], label="val")
    plt.title("total loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(return_dict["train_recon_losses"][cutoff:], label="train")
    plt.plot(return_dict["val_recon_losses"][cutoff:], label="val")
    plt.title("rec loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(return_dict["train_kl_losses"][cutoff:], label="train")
    plt.plot(return_dict["val_kl_losses"][cutoff:], label="val")
    plt.title("kl loss")
    plt.legend()

    plt.tight_layout()

    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_dataset_visualizations(
    dataset, indices=[0, -1], figsize=cm2inch((6, 4)), save=False, save_path=None
):
    """
    Plots samples, rates, and latents from the dataset for specified indices.

    Parameters:
    ----------
    dataset : object
        Dataset object containing 'samples', 'rates', and 'latents' attributes.
    indices : list, optional
        List of indices to plot. Default is [0, -1], the first and last samples.
    """
    for n_sa in indices:
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Plot samples with colorbar
        img0 = axs[0].imshow(dataset.samples[n_sa].T, cmap="gray_r", aspect="auto")
        cb1 = fig.colorbar(img0, ax=axs[0], fraction=0.1, pad=0.04)
        cb1.outline.set_linewidth(0.1)
        axs[0].set_ylabel("samples")

        # Plot rates with colorbar
        img1 = axs[1].imshow(dataset.rates[n_sa].T, aspect="auto")
        cb2 = fig.colorbar(img1, ax=axs[1], fraction=0.1, pad=0.04)
        cb2.outline.set_linewidth(0.1)
        axs[1].set_ylabel("rates")

        # Plot latents
        axs[2].plot(dataset.latents[n_sa, :, :])
        axs[2].set_xlabel("time (a.u.)")
        axs[2].set_ylabel("latents")
        cb3 = fig.colorbar(img1, ax=axs[2], fraction=0.1, pad=0.04)
        axs[2].set_xlabel("time (a.u.)")
        cb3.outline.set_linewidth(0.1)

        plt.tight_layout()
        if save and save_path is not None:
            plt.savefig(save_path + f"_{n_sa}.png")
            plt.savefig(save_path + f"_{n_sa}.pdf")


def plot_dataset_visualizations_order(
    dataset,
    indices=[0, -1],
    figsize=cm2inch((6, 4)),
    save=False,
    save_path=None,
    green=False,
):
    """
    Plots samples, rates, and latents from the dataset for specified indices.

    Parameters:
    ----------
    dataset : object
        Dataset object containing 'samples', 'rates', and 'latents' attributes.
    indices : list, optional
        List of indices to plot. Default is [0, -1], the first and last samples.
    """
    for n_sa in indices:
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Plot samples with colorbar
        img0 = axs[2].imshow(dataset.samples[n_sa].T, cmap="gray_r", aspect="auto")
        cb1 = fig.colorbar(img0, ax=axs[2], fraction=0.1, pad=0.04)
        cb1.outline.set_linewidth(0.1)
        axs[2].set_ylabel("samples")

        # Plot rates with colorbar
        img1 = axs[1].imshow(dataset.rates[n_sa].T, aspect="auto")
        cb2 = fig.colorbar(img1, ax=axs[1], fraction=0.1, pad=0.04)
        # add title to colorbar
        cb2.set_label("rate (Hz)")
        cb2.outline.set_linewidth(0.1)
        axs[1].set_ylabel("rates")

        # Plot latents
        # set the colormap to different shades of green
        if green:
            axs[0].plot(dataset.latents[n_sa, :, 0], color="#394932")
            axs[0].plot(dataset.latents[n_sa, :, 1], color="#AFCB90")
            axs[0].plot(dataset.latents[n_sa, :, 2], color="#5E7953")

        else:
            axs[0].plot(dataset.latents[n_sa, :, :].T)

        axs[0].set_ylabel("latents")
        cb3 = fig.colorbar(img1, ax=axs[0], fraction=0.1, pad=0.04)
        axs[2].set_xlabel("time (s)")
        cb3.outline.set_linewidth(0.1)
        cb1.set_label("spike count")

        plt.tight_layout()
        if save and save_path is not None:
            plt.savefig(save_path + f"_{n_sa}.png")
            plt.savefig(save_path + f"_{n_sa}.pdf")


def plot_rate_comparisons(
    gt_rates,
    model_rates_list,
    fn=average_rates,
    mode="neur",
    fps=None,
    figsize=cm2inch((10, 5 * 2)),
    colors=None,
    labels=["ae", "diffusion"],
    ms=1,
    save=False,
    save_path=None,
    xlabel="firing rate",
    switch_condition="diffusion",
):
    """
    Plots comparisons of firing rates (variance) between ground truth data and model predictions.
    Each model comparison is plotted in its own row, showing histograms of firing rates
    and scatter plots against GT rates.
    """
    n_models = len(model_rates_list)
    gt_averaged = fn(gt_rates, mode=mode, fps=fps)

    fig, axs = plt.subplots(n_models, 2, figsize=figsize)
    if n_models == 1:
        axs = np.expand_dims(axs, 0)

    for i, model_rates in enumerate(model_rates_list):
        model_averaged = fn(model_rates, mode=mode, fps=fps)

        min_val = min(gt_averaged.min(), model_averaged.min())
        max_val = max(gt_averaged.max(), model_averaged.max())
        bins = np.linspace(min_val, max_val, 25)
        # Histograms
        axs[i, 0].hist(gt_averaged, bins=bins, alpha=1, label="gt", color="grey")
        axs[i, 0].hist(
            model_averaged,
            bins=bins,
            alpha=0.5,
            label=labels[i],
            color=colors[i] if colors is not None else None,
        )
        axs[i, 0].set_title(mode + " avg " if fn == average_rates else mode + " std ")
        axs[i, 0].set_xlabel(xlabel)
        axs[i, 0].set_ylabel("count")
        axs[i, 0].legend()

        if mode != "neur" and labels[i] == switch_condition:
            # remove sumplot make empty
            axs[i, 1].axis("off")
        else:
            # Scatter plots
            axs[i, 1].plot(
                gt_averaged,
                model_averaged,
                ".",
                alpha=0.5,
                color=colors[i] if colors is not None else None,
                ms=ms,
            )
            axs[i, 1].plot(
                [gt_averaged.min(), gt_averaged.max()],
                [gt_averaged.min(), gt_averaged.max()],
                "k--",
            )
            # ensure equal axis
            axs[i, 1].axis("equal")
            axs[i, 1].set_title(mode + " avg " if fn == average_rates else " std ")
            axs[i, 1].set_xlabel("gt " + xlabel)
            axs[i, 1].set_ylabel(labels[i])

    plt.tight_layout()
    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_correlation_matrices(
    gt_rates,
    model_rates_list,
    model_labels=None,
    model_colors=None,
    mode="average",
    sample=0,
    figsize=cm2inch((12, 4)),
    save=False,
    save_path=None,
    ms=1,
):
    """
    Plots correlation matrices for ground truth and model predictions,
    and a scatter plot comparing model correlations with the ground truth.

    Parameters:
    ----------
    gt_rates : np.ndarray
        3D numpy array containing the ground truth rates with shape (n_samples, n_seqlen, n_neurons).
    model_rates_list : list
        List of 3D numpy arrays containing the model predictions, each with the same shape as gt_rates.
    sample : int, optional
        The index of the sample to use for generating the correlation matrices.
    """
    n_models = len(model_rates_list)
    fig, axs = plt.subplots(1, n_models + 2, figsize=figsize)

    # Plot gt correlation matrix
    gt_corr = correlation_matrix(gt_rates, sample=sample, mode=mode)
    axs[0].imshow(gt_corr, vmax=1, vmin=-1, cmap="coolwarm")
    axs[0].set_title("gt correlation")
    axs[0].set_xlabel("neuron id")
    axs[0].set_ylabel("neuron id")

    last_im = None  # Placeholder for the last imshow object
    for i, model_rates in enumerate(model_rates_list):
        # Plot model correlation matrix
        model_corr = correlation_matrix(model_rates, sample=sample, mode=mode)
        last_im = axs[i + 1].imshow(model_corr, vmax=1, vmin=-1, cmap="coolwarm")
        axs[i + 1].set_title(
            f"model {i+1} correlation" if model_labels is None else model_labels[i]
        )

        # Prepare data for scatter plot
        gt_corr_flat = gt_corr[np.triu_indices_from(gt_corr, k=1)]
        model_corr_flat = model_corr[np.triu_indices_from(model_corr, k=1)]

        # Scatter plot gt vs. model correlations
        axs[-1].plot(
            gt_corr_flat,
            model_corr_flat,
            ".",
            alpha=0.5,
            label=f"model {i+1}" if model_labels is None else model_labels[i],
            color=model_colors[i] if model_colors is not None else None,
            ms=ms,
        )
        for ax in axs:
            ax.axis("equal")
            ax.axis("square")

    # Final adjustments for scatter plot
    axs[-1].plot([-1, 1], [-1, 1], color="black", linestyle="--")
    axs[-1].set_xlim([-1, 1])
    axs[-1].set_ylim([-1, 1])
    axs[-1].set_xlabel("gt corr")
    axs[-1].set_ylabel("model corr")
    axs[-1].set_title("gt vs. model corr")
    axs[-1].legend()

    # Adjust layout and add colorbar to the last correlation matrix subplot
    plt.tight_layout()
    # fig.colorbar(last_im, ax=axs[0], fraction=0.046, pad=0.04)
    # fig.colorbar(last_im, ax=axs[1], fraction=0.046, pad=0.04)
    # fig.colorbar(last_im, ax=axs[2], fraction=0.046, pad=0.04)
    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_correlation_matrices_monkey(
    gt_rates,
    model_rates_list,
    model_labels=None,
    model_colors=None,
    mode="average",
    sample=0,
    figsize=cm2inch((12, 4)),
    save=False,
    save_path=None,
    ms=1,
    ticks=None,
):
    """
    Plots correlation matrices for ground truth and model predictions,
    and a scatter plot comparing model correlations with the ground truth.

    Parameters:
    ----------
    gt_rates : np.ndarray
        3D numpy array containing the ground truth rates with shape (n_samples, n_seqlen, n_neurons).
    model_rates_list : list
        List of 3D numpy arrays containing the model predictions, each with the same shape as gt_rates.
    sample : int, optional
        The index of the sample to use for generating the correlation matrices.
    """
    n_models = len(model_rates_list)
    max_corr_range = []
    fig, axs = plt.subplots(1, n_models + 2, figsize=figsize)

    # Plot gt correlation matrix
    gt_corr = correlation_matrix(gt_rates, sample=sample, mode=mode)
    np.fill_diagonal(gt_corr, 0)
    min_max_gt = np.nanmax(np.abs(gt_corr))
    max_corr_range.append(min_max_gt)

    # first get the range for all
    last_im = None  # Placeholder for the last imshow object
    for i, model_rates in enumerate(model_rates_list):
        # Plot model correlation matrix
        model_corr = correlation_matrix(model_rates, sample=sample, mode=mode)
        np.fill_diagonal(model_corr, 0)
        min_max_corr = np.nanmax(np.abs(model_corr))
        max_corr_range.append(min_max_corr)
        print(max_corr_range)

    max_corr_range = max(max_corr_range)
    axs[0].imshow(gt_corr, vmax=max_corr_range, vmin=-max_corr_range, cmap="coolwarm")
    axs[0].set_title("gt correlation")

    for i, model_rates in enumerate(model_rates_list):
        # Plot model correlation matrix
        model_corr = correlation_matrix(model_rates, sample=sample, mode=mode)
        np.fill_diagonal(model_corr, 0)
        last_im = axs[i + 1].imshow(
            model_corr, vmax=max_corr_range, vmin=-max_corr_range, cmap="coolwarm"
        )
        axs[i + 1].set_title(
            f"model {i+1} correlation" if model_labels is None else model_labels[i]
        )

        # Prepare data for scatter plot
        gt_corr_flat = gt_corr[np.triu_indices_from(gt_corr, k=1)]
        model_corr_flat = model_corr[np.triu_indices_from(model_corr, k=1)]

        # Scatter plot gt vs. model correlations
        axs[-1].plot(
            gt_corr_flat,
            model_corr_flat,
            ".",
            alpha=0.5,
            label=f"model {i+1}" if model_labels is None else model_labels[i],
            color=model_colors[i] if model_colors is not None else None,
        )
        for ax in axs:
            ax.axis("equal")
            ax.axis("square")

    # Final adjustments for scatter plot
    axs[-1].plot(
        [-max_corr_range, max_corr_range],
        [-max_corr_range, max_corr_range],
        color="black",
        linestyle="--",
    )
    axs[-1].set_xlim([-max_corr_range, max_corr_range])
    axs[-1].set_ylim([-max_corr_range, max_corr_range])
    axs[-1].set_xlabel("gt corr")
    axs[-1].set_ylabel("model corr")
    axs[-1].set_title("gt vs. model")
    if ticks is not None:
        axs[-1].set_xticks(ticks)
        axs[-1].set_yticks(ticks)
    axs[-1].legend()
    axs[0].set_xlabel("neuron id")
    axs[0].set_ylabel("neuron id")

    # Adjust layout and add colorbar to the last correlation matrix subplot
    plt.tight_layout()
    # fig.colorbar(last_im, ax=axs[0], fraction=0.046, pad=0.04)
    # fig.colorbar(last_im, ax=axs[1], fraction=0.046, pad=0.04)
    # fig.colorbar(last_im, ax=axs[2], fraction=0.046, pad=0.04)
    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_spiketrain_stats(
    spike_stats_1,
    spike_stats_2,
    save=False,
    save_path=None,
    labels=["gt", "recs"],
    figsize=(12, 6),
    color=None,
    to_plot=["mean_isi", "std_isi"],
):
    """
    Plot mean isi and std as scatter plots with square axes and consistent limits.
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for i, (key, value) in enumerate(spike_stats_1.items()):
        if key in to_plot:
            ax = axs.flat[i]
            vals_1 = value.flatten()
            vals_2 = spike_stats_2[key].flatten()
            x_min, x_max = (
            np.min([np.nanmin(vals_1), np.nanmin(vals_2)]),
                np.max([np.nanmax(vals_1), np.nanmax(vals_2)]),
            )
            x_min += -0.1 * x_max
            x_max += 0.1 * x_max
            ax.plot(vals_1, vals_2, ".", alpha=0.5, color=color)
            ax.set_aspect("equal", adjustable="box")
            ax.plot([x_min, x_max], [x_min, x_max], ls="--", c="black")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_title(key)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)

    plt.tight_layout()

    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_correlations(
    correlations, title, save=False, save_path=None, figsize=cm2inch((10, 6))
):
    """Plot correlations for multiple neurons."""
    plt.figure(figsize=figsize)
    for i, corr in enumerate(correlations, 1):
        plt.plot(corr)
    plt.title(title)
    plt.xlabel("lag")
    plt.ylabel("correlation")
    plt.legend()



def plot_n_channel_sd(
    rates_1,
    rates_2,
    channels=[0, 1],
    colors=["grey", "C0"],
    labels=["gt", "recs"],
    fps=100,
    save=False,
    save_path=None,
    lw=1,
    ystack=1,
    figsize=cm2inch((12, 4)),  # cm2inch((4 * len(channels), 4))
):
    """plot the spectral density of n channels"""
    print("Caution: Ensure fps is correct here.... PSD fps:", fps)
    fig, axs = plt.subplots(ystack, int(len(channels) / ystack), figsize=figsize)
    axs = axs.flatten()
    for ch, idx in enumerate(channels):
        plot_sd(
            fig=fig,
            ax=axs[ch],
            arr_one=rates_1[:, :, idx],
            arr_two=rates_2[:, :, idx],
            fs=fps,
            nperseg=rates_1.shape[1],
            agg_function=np.median,
            with_quantiles=True,
            x_ss=slice(0, 60),
            color_one=colors[0],
            color_two=colors[1],
            label_one=labels[0],
            label_two=labels[1],
            add_legend=True if ch == 0 else False,
            lw=lw,
        )
        # ensure y axis has at least two ticks
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("power [dB]")

    # set legend for the first plot
    plt.tight_layout()
    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def get_group_colors(num_groups=4, cm="jet"):
    if cm == "jet":
        colors = plt.cm.jet(np.linspace(0.65, 1, num_groups))
    elif cm == "Reds":
        colors = plt.cm.Reds(np.linspace(0.65, 1, num_groups))
    elif cm == "Blues":
        colors = plt.cm.Blues(np.linspace(0.65, 1, num_groups))
    elif cm == "Greys":
        colors = plt.cm.Greys(np.linspace(0.65, 1, num_groups))
    elif cm == "Greens":
        colors = plt.cm.Greens(np.linspace(0.65, 1, num_groups))
    elif cm == "Purples":
        colors = plt.cm.Purples(np.linspace(0.65, 1, num_groups))
    elif cm == "RdPu":
        colors = plt.cm.RdPu(np.linspace(0.65, 1, num_groups))
    else:
        colors = plt.cm.cubehelix(np.linspace(0, 1, num_groups))
    return colors


def plot_temp_corr_summary(cross_corr_groups,
                           auto_corr_groups,
                           binWidth=None,
                           name="", lw=1, ms=1):
    num_groups = len(cross_corr_groups)
    g_colors = get_group_colors(num_groups)

    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    ax_corr.set_title("Temporal Cross-Correlation {}".format(name))

    fig_auto, ax_auto = plt.subplots(figsize=(6, 4))
    ax_auto.set_title("Temporal Auto-Correlation {}".format(name))

    nlag = int((len(cross_corr_groups[0]) - 1) / 2)
    x_ticks = np.arange(-nlag, nlag + 1, 1)
    x_name = "Lag"

    if binWidth is not None:
        x_ticks = x_ticks * binWidth
        x_name = "Lag Time (ms)"

    for ind in range(num_groups):
        ax_corr.plot(
            x_ticks,
            cross_corr_groups[ind],
            ".-",
            lw=lw,
            ms=ms, 
            color=g_colors[num_groups - ind - 1],
            label="Group {}".format(ind + 1),
        )
        ax_auto.plot(
            x_ticks,
            auto_corr_groups[ind],
            ".-",
            lw=lw,
            ms=ms, 
            color=g_colors[num_groups - ind - 1],
            label="Group {}".format(ind + 1),
        )

    ax_corr.set_xlabel(x_name)
    ax_corr.set_xticks(x_ticks)
    ax_corr.legend(loc=1, prop={"size": 11})

    ax_auto.set_xlabel(x_name)
    ax_auto.set_xticks(x_ticks)
    ax_auto.legend(loc=1, prop={"size": 11})
    return fig_corr, fig_auto


def plot_temp_corr_summary_stacked(
    cross_corr_groups, auto_corr_groups, binWidth=None, name=""
):
    num_groups = len(cross_corr_groups)
    g_colors = get_group_colors(num_groups)

    fig_temp, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].set_title("Temporal Cross-Correlation {}".format(name))
    axs[1].set_title("Temporal Auto-Correlation {}".format(name))

    nlag = int((len(cross_corr_groups[0]) - 1) / 2)
    x_ticks = np.arange(-nlag, nlag + 1, 1)
    x_name = "Lag"

    if binWidth is not None:
        x_ticks = x_ticks * binWidth
        x_name = "Lag Time (ms)"

    for ind in range(num_groups):
        axs[0].plot(
            x_ticks,
            cross_corr_groups[ind],
            ".-",
            lw=1,
            color=g_colors[num_groups - ind - 1],
            label="Group {}".format(ind + 1),
        )
        axs[1].plot(
            x_ticks,
            auto_corr_groups[ind],
            ".-",
            lw=1,
            color=g_colors[num_groups - ind - 1],
            label="Group {}".format(ind + 1),
        )

    for i in range(2):
        axs[i].set_xlabel(x_name)
        axs[i].set_ylabel("Correlation")
        axs[i].set_xticks(x_ticks)
        axs[i].legend(loc=1, prop={"size": 11})
    return fig_temp


def plot_cross_corr_summary(
    cross_corr_groups,
    binWidth=None,
    name="",
    cmap="Reds",
    figsize=(6, 4),
    linestyle=".-",
    ax_corr=None,
    ax_auto=None,
    ncol=2,
    labels="group",
    title="cross-corr",
    xlabel="lag",
    ylabel="x-corr",
    save=False,
    save_path=None,
):
    num_groups = len(cross_corr_groups)
    g_colors = get_group_colors(num_groups, cmap)

    if ax_corr is None:
        fig_corr, ax_corr = plt.subplots(figsize=figsize)
    ax_corr.set_title(f"temporal {title} {name}")

    nlag = int((len(cross_corr_groups[0]) - 1) / 2)
    x_ticks = np.arange(-nlag, nlag + 1, 1)

    if binWidth is not None:
        x_ticks = x_ticks * binWidth
        x_name = "lag time (ms)"

    for ind in range(num_groups):
        ax_corr.plot(
            x_ticks,
            cross_corr_groups[ind],
            linestyle,
            lw=1,
            color=g_colors[num_groups - ind - 1],
            label=f"{labels} {ind + 1}",
        )

    ax_corr.set_xlabel(xlabel)
    # ax_corr.legend(ncol=ncol, loc="upper left", bbox_to_anchor=(1, 1))
    ax_corr.set_ylabel(ylabel)

    ax_corr.locator_params(nbins=5)
    if save and save_path is not None:
        fig_corr.savefig(save_path + ".png")
        fig_corr.savefig(save_path + ".pdf")
    return ax_corr


# mode of just passing the model and the dataloader
# -----------------------------------------------------------


def plot_inferred_latents(
    model,
    dataloader,
    n_latents=16,
    y_stack=2,
    figsize=(10, 2),
    idx=0,
    true_data=True,
    color="midnightblue",
    save=False,
    save_path=None,
    indices=None
):
    """plot the inferred latents for a single sample idx"""
    model.eval()
    for batch in dataloader:
        signal = batch["signal"]
        if not true_data:
            real_rates = batch["rates"]
        with torch.no_grad():
            output_rates, z = model(signal)
            output_rates = output_rates.cpu()
            z = z.cpu()

        signal = signal.cpu()  # move signal to cpu
        if not true_data:
            real_rates = real_rates.cpu()
        break

    if indices is not None:

        n_latents = len(indices)
        fig, ax = plt.subplots(y_stack, int(n_latents / y_stack), figsize=figsize)
        ax = ax.flatten()
        for i in range(n_latents):
            ax[i].plot(z[idx, indices[i]].numpy(), color=color)
            ax[i].set_title(f"latent {indices[i]}")
            # ax[i].set_yticks([])
            #ax[i].set_ylim(z[idx].min() - 0.1, z[idx].max() + 0.1)
            if i < int(n_latents / y_stack):
                ax[i].set_xticks([])
            if i % int(n_latents / y_stack) != 0:
                ax[i].set_yticks([])
        fig.suptitle("inferred latents")
        # add xlabel to the last row to the left
        for i in range(int(n_latents / y_stack)):
            ax[-i - 1].set_xlabel("time (s)")
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(y_stack, int(n_latents / y_stack), figsize=figsize)
        ax = ax.flatten()
        for i in range(n_latents):
            ax[i].plot(z[idx, i].numpy(), color=color)
            ax[i].set_title(f"latent {i}")
            # ax[i].set_yticks([])
            # ax[i].set_ylim(z[idx].min() - 0.1, z[idx].max() + 0.1)
            if i < int(n_latents / y_stack):
                ax[i].set_xticks([])
            if i % int(n_latents / y_stack) != 0:
                ax[i].set_yticks([])
        fig.suptitle("inferred latents")
        # add xlabel to the last row to the left
        for i in range(int(n_latents / y_stack)):
            ax[-i - 1].set_xlabel("time (s)")
        plt.tight_layout()

    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_rec_rates(model, dataloader, figsize=(10, 8), save=False, save_path=None):
    model.eval()
    for batch in dataloader:
        signal = batch["signal"]
        real_rates = batch["rates"]
        with torch.no_grad():
            output_rates = model(signal)[0].cpu()

        signal = signal.cpu()  # move signal to cpu
        real_rates = real_rates.cpu()
        break

    fig, ax = plt.subplots(4, 1, figsize=figsize, dpi=300)

    im = ax[0].imshow(signal[0].cpu().numpy(), aspect="auto", cmap="Greys")
    cbar = plt.colorbar(im, ax=ax[0])
    ax[0].set_title("spikes")

    im = ax[1].imshow(real_rates[0].cpu().numpy(), aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, ax=ax[1])
    ax[1].set_title("real rates")

    im = ax[2].imshow(output_rates[0].cpu().numpy(), aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, ax=ax[2])
    ax[2].set_title("predicted rates")

    rate_errors = (output_rates[0] - real_rates[0]).abs().cpu().numpy()
    rate_errors_q99_clipped = np.clip(
        rate_errors, 0, np.quantile(rate_errors, 0.99)
    )  # /real_rates[0].cpu().numpy()
    im = ax[3].imshow(rate_errors_q99_clipped, aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, ax=ax[3])
    # limits of colorbar
    # cbar.set_clim(0, np.quantile(rate_errors, 0.99))
    ax[3].set_title("rate error (abs)")

    fig.tight_layout()

    return fig

    # fig = plt.figure(figsize=cm2inch((4, 2)), dpi=300)
    # plt.hist(rate_errors_q99_clipped.flatten(), bins=100)
    # plt.title("rate error histogram")
    # if save and save_path is not None:
    #     plt.savefig(save_path + ".png")
    #     plt.savefig(save_path + ".pdf")
    # else:


def plot_rate_traces(
    model,
    dataloader,
    figsize=(12, 5),
    idx=0,
    plot_real_rates=True,
    color="midnightblue",
    save=False,
    save_path=None,
    xlabel="time (a.u.)",
    ylabel="rate",
):
    model.eval()
    for batch in dataloader:
        signal = batch["signal"]

        with torch.no_grad():
            output_rates = model(signal)[0].cpu()

        signal = signal.cpu()  # move signal to cpu
        break

    fig, ax = plt.subplots(4, 1, figsize=figsize)
    channels = np.arange(0, 128, 32)

    for i, channel in enumerate(channels):

        L = batch["signal"][idx, channel].shape[0]
        ax[i].vlines(
            torch.arange(L),
            torch.zeros(L),
            torch.ones(L),
            # batch["signal"][0, channel].cpu().numpy(),
            # label="spikes",
            color="black",
            alpha=np.min(
                np.stack(
                    (np.ones(L), batch["signal"][idx, channel].cpu().numpy() * 0.2),
                    axis=1,
                ),
                axis=1,
            ),
        )
        # plot on different axis
        ax2 = ax[i].twinx()

        ax[i].set_title(f"channel {channel}")
        if plot_real_rates:
            ax2.plot(
                batch["rates"][idx, channel].cpu().numpy(), label="real", color="grey"
            )
        ax2.plot(output_rates[idx, channel].cpu().numpy(), label="pred", color=color)
    ax[-1].legend()
    ax[-1].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    # fig.suptitle("rate traces for channels")
    fig.tight_layout()
    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_3d_latent_trajectory(
    ae,
    val_dataloader,
    figsize=(6, 6),
    sample_idx=0,
    indices=[0, 1, 2],
    save=False,
    save_path=None,
    plot_real_rates=True,
):
    from matplotlib import cm

    # plot 3d latent trajectory
    ae.eval()
    for batch in val_dataloader:
        signal = batch["signal"]
        if plot_real_rates:
            real_rates = batch["rates"]
        with torch.no_grad():
            output_rates, z = ae(signal)
            output_rates = output_rates.cpu()
            z = z.cpu()

        signal = signal.cpu()  # move signal to cpu
        if plot_real_rates:
            real_rates = real_rates.cpu()
        break

    # Create a figure and an axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    z1 = z[sample_idx, indices[1], :].cpu().numpy()  # (1024,)
    z2 = z[sample_idx, indices[2], :].cpu().numpy()
    z3 = z[sample_idx, indices[0], :].cpu().numpy()

    # Initialize the scatter plot with correct color dimensions
    colors = cm.viridis(
        np.linspace(0, 1, len(z1))
    )  # Ensure colors are mapped from a colormap
    scatter = ax.scatter(z1, z2, z3, c=colors, s=3)
    (line,) = ax.plot(z1, z2, z3, color="k", alpha=0.3)
    # set x y and z label names
    ax.set_xlabel(f"lat {indices[1]}")
    ax.set_ylabel(f"lat {indices[2]}")
    ax.set_zlabel(f"lat {indices[0]}")

    plt.tight_layout()
    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_3d_latent_trajectory_direct(
    z,
    figsize=(6, 6),
    sample_idx=0,
    indices=[0, 1, 2],
    save=False,
    save_path=None,
    cmap="viridis",
    ticksoff=False,
    ms=1,
    lw=1,
):
    from matplotlib import cm

    # Create a figure and an axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    z1 = z[sample_idx, indices[1], :].cpu().numpy()  # (1024,)
    z2 = z[sample_idx, indices[2], :].cpu().numpy()
    z3 = z[sample_idx, indices[0], :].cpu().numpy()

    # Initialize the scatter plot with correct color dimensions
    if cmap == "viridis":
        colors = cm.viridis(
            np.linspace(0, 1, len(z1))
        )  # Ensure colors are mapped from a colormap
    elif cmap == "Reds":
        colors = cm.Reds(np.linspace(0, 1, len(z1)))
    elif cmap == 'Greens':
        colors = cm.Greens(
            np.linspace(0, 1, len(z1))
        )
    elif cmap == 'Greys':
        colors = cm.Greys(
            np.linspace(0, 1, len(z1))
        )
    else:
        colors = cm.viridis(
            np.linspace(0, 1, len(z1))
        )  # Ensure colors are mapped from a colormap
    scatter = ax.scatter(z1, z2, z3, c=colors, s=ms)
    (line,) = ax.plot(z1, z2, z3, color="k", alpha=0.3, lw=lw)
    # set x y and z label names
    # ax.set_xlabel(f"lat {indices[1]}")
    # ax.set_ylabel(f"lat {indices[2]}")
    # ax.set_zlabel(f"lat {indices[0]}")
    if ticksoff:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    ax.locator_params(nbins=4)
    plt.tight_layout()
    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_spikes_next_to_each_other(
    model,
    dataloader,
    idx=0,
    true_data=True,
    figsize=cm2inch(12, 5),
    save=False,
    save_path=None,
    xlabel="time (a.u.)",
    ylabel="neuron",
    binary=False,
):
    model.eval()
    for batch in dataloader:
        signal = batch["signal"]
        with torch.no_grad():
            output_rates = model(signal)[0].cpu()

        signal = signal.cpu()  # move signal to cpu
        break

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    channels = np.arange(0, 128, 32)
    output_spikes = np.random.poisson(output_rates)[idx]
    maxval = np.max(
        [output_spikes.flatten(), batch["signal"][idx].cpu().numpy().flatten()]
    )
    if binary:
        maxval = 1

    ax[0].imshow(
        batch["signal"][idx].cpu().numpy(),
        vmin=0,
        vmax=maxval,
        aspect="auto",
        cmap="Greys",
    )
    im = ax[1].imshow(output_spikes, aspect="auto", vmin=0, vmax=maxval, cmap="Greys")
    # add colorbar
    if not binary:
        cbar = plt.colorbar(im, ax=ax[1])
        cbar = plt.colorbar(im, ax=ax[0])
    ax[0].set_ylabel(ylabel)
    ax[0].set_xlabel(xlabel)
    ax[1].set_xlabel(xlabel)

    fig.tight_layout()
    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_spike_histogram(
    model,
    dataloader,
    idx=0,
    true_data=True,
    figsize=(12, 5),
    color="darkblue",
    labels=["gt", "ae"],
    max_bins=40,
    save=False,
    save_path=None,
):

    model.eval()
    all_outputs = []
    all_signals = []
    for batch in dataloader:
        signal = batch["signal"]
        with torch.no_grad():
            output_rates = model(signal)[0].cpu()
        all_outputs.append(output_rates)

        signal = signal.cpu()  # move signal to cpu
        all_signals.append(signal)
        break

    all_outputs = torch.cat(all_outputs, dim=0)
    all_signals = torch.cat(all_signals, dim=0)

    fig = plt.figure(figsize=figsize)

    sum_outputs = torch.sum(all_outputs, axis=2)
    sum_gt = torch.sum(all_signals, axis=2)

    max_val = torch.max(torch.stack((sum_gt, sum_outputs), dim=0).flatten())
    bins = np.linspace(0, max_val, min(int(max_val), max_bins))
    plt.hist(sum_gt.flatten(), bins=bins, alpha=0.5, label=labels[0], color="grey")
    plt.hist(sum_outputs.flatten(), bins=bins, alpha=0.5, label=labels[1], color=color)
    plt.xlabel("# of spikes summed over time")
    plt.ylabel("count")
    plt.legend()
    fig.tight_layout()
    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_population_spike_histogram(
    gt_spikes,
    model_spikes,
    labels=["gt", "ae"],
    colors=["grey", "midnightblue"],
    bins=np.linspace(0, 150, 50),
    figsize=cm2inch(8, 4),
    x_label="number of spikes per time bin",
    y_label="frequency",
    ax=None,
    save=False,
    save_path=None,
):
    """plot population spike hist"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(
        np.sum(gt_spikes, axis=2).flatten(),
        bins=bins,
        label=labels[0],
        color=colors[0],
        density=True,
    )
    ax.hist(
        np.sum(model_spikes, axis=2).flatten(),
        bins=bins,
        alpha=0.5,
        density=True,
        label=labels[1],
        color=colors[1],
    )
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim([min(bins), max(bins)])

    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_spike_histogram_list(
    list,
    idx=0,
    true_data=True,
    figsize=(12, 5),
    colors=["grey", "darkblue"],
    labels=["gt", "ae"],
    max_bins=40,
    xlabel="# spikes",
    save=False,
    save_path=None,
):
    """plot spike histogram for a given array of spikes"""
    fig = plt.figure(figsize=figsize)
    max_val = np.max([l.max() for l in list])
    bins = np.linspace(0, max_val, min(int(max_val), max_bins))
    for i, spikes in enumerate(list):
        plt.hist(
            spikes.flatten(), bins=bins, alpha=0.5, label=labels[i], color=colors[i]
        )

    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    fig.tight_layout()

    if save and save_path is not None:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_initial_conds(train_dataloader):
    """plot lorenz initial conditions"""
    all_init_conds = torch.stack(
        [
            train_dataloader.dataset[i]["latents"][:, 0]
            for i in range(len(train_dataloader.dataset))
        ]
    )
    all_lorenz = torch.stack(
        [
            train_dataloader.dataset[i]["latents"]
            for i in range(len(train_dataloader.dataset))
        ]
    )

    for i in range(3):
        plt.figure(figsize=(1, 1))
        plt.hist(all_init_conds[:, i], bins=100, color="C" + str(i))

    # make 3d scatter plot
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        all_init_conds[:, 0],
        all_init_conds[:, 1],
        all_init_conds[:, 2],
        s=0.3,
        color="C1",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # make 3d scatter plot
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        all_lorenz[:100, 0, :].flatten(),
        all_lorenz[:100, 1, :].flatten(),
        all_lorenz[:100, 2, :].flatten(),
        s=0.3,
    )
    ax.scatter(all_init_conds[:, 0], all_init_conds[:, 1], all_init_conds[:, 2], s=0.3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def visaulise_Lorenz_AE_latents(LORENZ_AE):
    """Visualise the Lorenz AE latents"""
    fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    ax = ax.flatten()
    for i in range(16):
        mean = np.mean(LORENZ_AE.data_array[:, :, i], axis=0)
        std = np.std(LORENZ_AE.data_array[:, :, i], axis=0)
        ax[i].plot(mean)
        ax[i].fill_between(np.arange(256), mean - std, mean + std, alpha=0.3)
        ax[i].plot(LORENZ_AE.data_array[0, :, i])
        ax[i].plot(LORENZ_AE.data_array[10, :, i])
        ax[i].plot(LORENZ_AE.data_array[3000, :, i])


def visualise_spikes_trains(
    spike_trains_gt,
    spike_trains_ae,
    spike_trains_diff,
    ae_rates,
    figsize=(cm2inch(6, 3)),
    ms=1,
    save=False,
    save_path=None,
):
    """Visualise the spike trains"""
    plt.figure(figsize=figsize)
    for (sample_idx, neuron_idx), spikes in spike_trains_gt.items():
        plt.plot(
            spikes, np.ones_like(spikes) * neuron_idx, "|", color="black", markersize=ms
        )

        if neuron_idx == ae_rates.shape[-1] - 1:
            break
    plt.xlabel("time [s]")
    plt.ylabel("neuron idx")
    plt.title("gt spikes")
    plt.locator_params(nbins=5)
    if save and save_path is not None:
        plt.savefig(save_path + "_gt.png")
        plt.savefig(save_path + "_gt.pdf")
    plt.figure(figsize=figsize)

    for (sample_idx, neuron_idx), spikes in spike_trains_ae.items():
        plt.plot(
            spikes,
            np.ones_like(spikes) * neuron_idx,
            "|",
            color="midnightblue",
            markersize=ms,
        )

        if neuron_idx == ae_rates.shape[-1] - 1:
            break
    plt.xlabel("time [s]")
    plt.ylabel("neuron idx")
    plt.title("ae spikes")
    plt.locator_params(nbins=5)
    if save and save_path is not None:
        plt.savefig(save_path + "_ae.png")
        plt.savefig(save_path + "_ae.pdf")
    plt.figure(figsize=figsize)

    for (sample_idx, neuron_idx), spikes in spike_trains_diff.items():
        plt.plot(
            spikes,
            np.ones_like(spikes) * neuron_idx,
            "|",
            color="darkred",
            markersize=ms,
        )

        if neuron_idx == ae_rates.shape[-1] - 1:
            break

    plt.xlabel("time [s]")
    plt.ylabel("neuron idx")
    plt.title("diff spikes")
    plt.locator_params(nbins=5)

    if save and save_path is not None:
        plt.savefig(save_path + "_diffusion.png")
        plt.savefig(save_path + "_diffusion.pdf")


def angle_to_color(angle, cmap_s="hsv"):
    """Convert an angle in radians to a color using the HSV colormap."""
    cmap = plt.get_cmap(cmap_s)
    # Normalize angle from [-π, π] to [0, 1]
    normalized_angle = (angle + np.pi) / (2 * np.pi)
    return cmap(normalized_angle)


# -------------------------- composite plotting functions --------------------------


def evaluate_autoencoder(
    ae,
    val_dataloader,
    val_dataloader_longer,
    n_latents=8,
    save=False,
    save_path=None,
    idx=0,
    indices=[
        5,
        4,
        3,
    ],  # which latents to plot in 3d
    plot_real_rates=True,
):
    if save_path is not None:
        save_path = save_path + "/ae_figures/"
        os.makedirs(save_path, exist_ok=True)

    from ldns.utils.plotting_utils import (
        plot_rate_traces,
        plot_inferred_latents,
        plot_3d_latent_trajectory,
        plot_spikes_next_to_each_other,
        cm2inch,
    )

    plot_rate_traces(
        ae,
        val_dataloader,
        idx=idx,
        figsize=cm2inch(10, 10),
        plot_real_rates=plot_real_rates,
        save=save,
        save_path=(save_path + "rate_traces" if save_path is not None else None),
    )

    plot_spikes_next_to_each_other(
        ae,
        val_dataloader,
        idx=idx,
        figsize=cm2inch(15, 5),
        save=save,
        save_path=(save_path + "spikes_next_to_each_other" if save_path is not None else None),
    )

    # run all sorts of analyses
    plot_inferred_latents(
        ae,
        val_dataloader,
        n_latents=8,
        y_stack=4,
        figsize=cm2inch(10, 10),
        color="royalblue",
        idx=idx,
        save=save,
        save_path=(save_path + "inferred_latents" if save_path is not None else None),
    )

    plot_3d_latent_trajectory(
        ae,
        val_dataloader,
        figsize=cm2inch(8, 8),
        sample_idx=idx,
        save=save,
        indices=indices,
        plot_real_rates=plot_real_rates,
        save_path=(save_path + "3d_latent_trajectory" if save_path is not None else None),
    )

    # long validation dataloader
    plot_rate_traces(
        ae,
        val_dataloader_longer,
        figsize=cm2inch(20, 10),
        idx=idx,
        plot_real_rates=plot_real_rates,
        save=save,
        save_path=(save_path + "rate_traces_longer" if save_path is not None else None),
    )

    plot_3d_latent_trajectory(
        ae,
        val_dataloader_longer,
        figsize=cm2inch(8, 8),
        sample_idx=idx,
        save=save,
        indices=indices,
        plot_real_rates=plot_real_rates,
        save_path=(save_path + "3d_latent_trajectory_longer" if save_path is not None else None),
    )


# --------------------------- baseline support -------


def plot_hankel_stim(hankel, activity, figsize=cm2inch(10, 10)):
    # make gridspec that one is 0.9 and the oder one 0.1
    fig, ax = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [0.9, 0.1]}
    )
    ax[0].imshow(hankel, aspect="auto", cmap="coolwarm")
    ax[0].set_title("hankel matrix")
    ax[0].set_xlabel("lag")
    ax[0].set_ylabel("time point")
    ax[1].imshow(activity[:, None], aspect="auto", cmap="Greys")
    ax[1].set_title("activity")
    ax[1].set_ylabel("time point")
    ax[1].set_xticks([])
    plt.tight_layout()


def plot_spike_filter(weights, dt, show=True, **kws):
    """Plot estimated weights time lagged with dt bin width"""
    d = len(weights)
    t = np.arange(-d + 1, 1) * dt

    ax = plt.gca()
    ax.plot(t, weights, marker="o", **kws)
    ax.axhline(0, color=".2", linestyle="--", zorder=1)
    ax.set(
        xlabel="time before spike (s)",
        ylabel="filter weight",
    )
