"""
Visualization functions for simulation results.

Provides plotting utilities for spatial data and simulation outputs.
"""

from typing import Optional

import numpy as np


def plot_spatial_cells(
    coords: np.ndarray,
    cell_types: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
    sender_indices: Optional[np.ndarray] = None,
    receiver_indices: Optional[np.ndarray] = None,
    title: str = "Spatial Cell Distribution",
    figsize: tuple[float, float] = (10, 10),
    cmap: str = "viridis",
    alpha: float = 0.7,
    point_size: float = 10,
    ax=None,
):
    """
    Plot spatial cell distribution.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n_cells, 2).
    cell_types : np.ndarray, optional
        Cell type labels for coloring.
    expression : np.ndarray, optional
        Expression values for coloring (overrides cell_types).
    sender_indices : np.ndarray, optional
        Indices of sender cells to highlight.
    receiver_indices : np.ndarray, optional
        Indices of receiver cells to highlight.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for expression.
    alpha : float
        Point transparency.
    point_size : float
        Point size.
    ax : matplotlib axis, optional
        Existing axis to plot on.

    Returns
    -------
    matplotlib axis
        The plot axis.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    x, y = coords[:, 0], coords[:, 1]

    if expression is not None:
        scatter = ax.scatter(x, y, c=expression, cmap=cmap, alpha=alpha, s=point_size)
        plt.colorbar(scatter, ax=ax, label="Expression")
    elif cell_types is not None:
        unique_types = np.unique(cell_types)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        for i, ct in enumerate(unique_types):
            mask = cell_types == ct
            ax.scatter(x[mask], y[mask], c=[colors[i]], label=ct, alpha=alpha, s=point_size)
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.scatter(x, y, alpha=alpha, s=point_size, c="gray")

    # Highlight senders and receivers
    if sender_indices is not None:
        ax.scatter(
            x[sender_indices],
            y[sender_indices],
            facecolors="none",
            edgecolors="red",
            s=point_size * 3,
            linewidths=1.5,
            label="Senders",
        )

    if receiver_indices is not None:
        ax.scatter(
            x[receiver_indices],
            y[receiver_indices],
            facecolors="none",
            edgecolors="blue",
            s=point_size * 3,
            linewidths=1.5,
            label="Receivers",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")

    return ax


def plot_ind_curve(
    radii: list[float],
    ind_values: list[float],
    lambda_est: Optional[float] = None,
    title: str = "I_ND vs Radius",
    figsize: tuple[float, float] = (8, 6),
    ax=None,
):
    """
    Plot I_ND curve across radii.

    Parameters
    ----------
    radii : list of float
        Radius values.
    ind_values : list of float
        I_ND values at each radius.
    lambda_est : float, optional
        Estimated diffusion length to mark on plot.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib axis, optional
        Existing axis to plot on.

    Returns
    -------
    matplotlib axis
        The plot axis.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(radii, ind_values, "o-", linewidth=2, markersize=8, color="steelblue")

    if lambda_est is not None:
        ax.axvline(
            lambda_est, color="red", linestyle="--", linewidth=1.5, label=f"λ = {lambda_est:.0f}"
        )
        ax.legend()

    ax.set_xlabel("Radius (μm)")
    ax.set_ylabel("I_ND")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Find and annotate peak
    peak_idx = np.argmax(ind_values)
    peak_radius = radii[peak_idx]
    peak_ind = ind_values[peak_idx]
    ax.annotate(
        f"Peak: r={peak_radius:.0f}, I_ND={peak_ind:.3f}",
        xy=(peak_radius, peak_ind),
        xytext=(peak_radius + 50, peak_ind - 0.05),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )

    return ax


def plot_simulation_summary(
    result: dict,
    figsize: tuple[float, float] = (15, 5),
):
    """
    Plot comprehensive simulation summary.

    Parameters
    ----------
    result : dict
        Simulation result from UnifiedSimulation.run_single().
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib figure
        The figure object.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Spatial distribution
    plot_spatial_cells(
        result["positions"],
        sender_indices=result["sender_indices"],
        receiver_indices=result["receiver_indices"],
        title="Cell Positions",
        ax=axes[0],
    )

    # Plot 2: Factor expression
    plot_spatial_cells(
        result["positions"],
        expression=result["factor_expr"],
        title="Factor Expression",
        cmap="Reds",
        ax=axes[1],
    )

    # Plot 3: I_ND curve
    radii = [p["radius"] for p in result["ind_curve"]]
    ind_values = [p["I_ND"] for p in result["ind_curve"]]
    plot_ind_curve(
        radii,
        ind_values,
        lambda_est=result.get("lambda"),
        title="I_ND Curve",
        ax=axes[2],
    )

    plt.tight_layout()
    return fig


def plot_expression_histogram(
    expression: np.ndarray,
    cell_mask: Optional[np.ndarray] = None,
    title: str = "Expression Distribution",
    bins: int = 50,
    log_scale: bool = True,
    figsize: tuple[float, float] = (8, 6),
    ax=None,
):
    """
    Plot expression histogram.

    Parameters
    ----------
    expression : np.ndarray
        Expression values (1D array).
    cell_mask : np.ndarray, optional
        Boolean mask for cells to include.
    title : str
        Plot title.
    bins : int
        Number of histogram bins.
    log_scale : bool
        Use log scale for x-axis.
    figsize : tuple
        Figure size.
    ax : matplotlib axis, optional
        Existing axis to plot on.

    Returns
    -------
    matplotlib axis
        The plot axis.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if cell_mask is not None:
        expression = expression[cell_mask]

    # Filter out zeros for log scale
    if log_scale:
        expression = expression[expression > 0]
        ax.hist(expression, bins=bins, alpha=0.7, edgecolor="black")
        ax.set_xscale("log")
    else:
        ax.hist(expression, bins=bins, alpha=0.7, edgecolor="black")

    ax.set_xlabel("Expression")
    ax.set_ylabel("Count")
    ax.set_title(title)

    # Add statistics
    stats_text = (
        f"Mean: {np.mean(expression):.2f}\nStd: {np.std(expression):.2f}\nN: {len(expression)}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    return ax


def plot_receiver_fraction_comparison(
    results: dict[float, dict],
    figsize: tuple[float, float] = (10, 6),
):
    """
    Compare I_ND curves across receiver fractions.

    Parameters
    ----------
    results : dict
        Dictionary mapping receiver_fraction -> simulation result.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib figure
        The figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i, (frac, result) in enumerate(sorted(results.items())):
        radii = [p["radius"] for p in result["ind_curve"]]
        ind_values = [p["I_ND"] for p in result["ind_curve"]]

        ax.plot(
            radii,
            ind_values,
            "o-",
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=f"Receiver frac = {frac:.1%}",
        )

    ax.set_xlabel("Radius (μm)")
    ax.set_ylabel("I_ND")
    ax.set_title("I_ND Curves by Receiver Fraction")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig


def plot_diffusion_field(
    coords: np.ndarray,
    concentrations: np.ndarray,
    sender_pos: np.ndarray,
    title: str = "Diffusion Field",
    figsize: tuple[float, float] = (10, 10),
    cmap: str = "YlOrRd",
    ax=None,
):
    """
    Plot diffusion concentration field.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n_cells, 2).
    concentrations : np.ndarray
        Concentration values at each cell.
    sender_pos : np.ndarray
        Sender positions (n_senders, 2).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap.
    ax : matplotlib axis, optional
        Existing axis to plot on.

    Returns
    -------
    matplotlib axis
        The plot axis.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot concentration field
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=concentrations,
        cmap=cmap,
        s=10,
        alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label="Concentration")

    # Mark sender positions
    ax.scatter(
        sender_pos[:, 0],
        sender_pos[:, 1],
        c="black",
        s=100,
        marker="*",
        edgecolors="white",
        linewidths=1,
        label="Senders",
        zorder=5,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()

    return ax


def save_figure(fig, path: str, dpi: int = 150):
    """
    Save figure to file.

    Parameters
    ----------
    fig : matplotlib figure
        Figure to save.
    path : str
        Output path.
    dpi : int
        Resolution.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
