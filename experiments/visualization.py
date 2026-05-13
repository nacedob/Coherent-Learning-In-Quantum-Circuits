from .utils import import_root, find_root
import_root()

from src.models.history import TrainingHistory
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Optional, Dict, List


# =========================================================
# ----------- Generic data extraction layer ---------------
# =========================================================

def extract_series(
    history: TrainingHistory,
    metric_fn: Callable,
) -> Dict[str, List]:
    if not history.is_complete():
        raise RuntimeError("History is not complete, cannot extract")

    epochs = range(history.epochs + 1)

    epoch_x = []
    epoch_y = []

    batch_x = []
    batch_y = []

    xticks = []
    xticklabels = []

    width = 0.7

    for e in epochs:
        steps = history.get_epoch(e)
        if not steps:
            continue

        # -----------------------------
        # Epoch value
        # -----------------------------
        epoch_x.append(0 if e == 0 else e)
        epoch_y.append(metric_fn(steps[-1]))

        # -----------------------------
        # Batch values
        # -----------------------------
        n = len(steps)
        offsets = np.linspace(-width / 2, width / 2, n)

        if e == 0:
            xpos = 1 - width
            batch_x.append(xpos)
            batch_y.append(metric_fn(steps[-1]))

            xticks.append(xpos)
            xticklabels.append('0')
            continue

        for i, (offset, s) in enumerate(zip(offsets, steps)):
            xpos = e + offset
            val = metric_fn(s)

            batch_x.append(xpos)
            batch_y.append(val)

            if i == len(steps) - 1:
                xticks.append(xpos)
                xticklabels.append(str(e))

    return {
        "epoch_x": epoch_x,
        "epoch_y": epoch_y,
        "batch_x": batch_x,
        "batch_y": batch_y,
        "xticks": xticks,
        "xticklabels": xticklabels,
    }


# =========================================================
# ------------------ Plotting layer ------------------------
# =========================================================

def plot_series(
    ax: plt.Axes,
    data: Dict,
    label: Optional[str] = None,
    show_batches: bool = True,
    ylabel: str = "",
    title: str = "",
):
    epoch_x = data["epoch_x"]
    epoch_y = data["epoch_y"]
    batch_x = data["batch_x"]
    batch_y = data["batch_y"]
    xticks = data["xticks"]
    xticklabels = data["xticklabels"]

    # -----------------------------
    # Background shading
    # -----------------------------
    for e in epoch_x:
        if e % 2 == 0:
            ax.axvspan(e - 0.5, e + 0.5, alpha=0.08)

    # -----------------------------
    # Epoch curve
    # -----------------------------
    ax.plot(
        epoch_x,
        epoch_y,
        linewidth=2.5,
        marker="o",
        label=label,
        zorder=3,
    )

    # -----------------------------
    # Batch-level curve
    # -----------------------------
    if show_batches:
        ax.plot(
            batch_x,
            batch_y,
            alpha=0.35,
            linewidth=1.2,
            zorder=1,
        )
    else:
        xticks = list(epoch_x)
        xticklabels = [str(e) for e in epoch_x]

    # -----------------------------
    # Tick control (IDENTICAL)
    # -----------------------------
    max_ticks = 40
    if len(xticks) > max_ticks:
        step = len(xticks) // max_ticks
        xticks = xticks[::step]
        xticklabels = xticklabels[::step]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, fontsize=8)

    # -----------------------------
    # Styling (IDENTICAL)
    # -----------------------------
    ax.set_xlabel("Epoch-Batch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, alpha=0.25)
    ax.legend()


# =========================================================
# ----------- High-level wrappers (same API) ---------------
# =========================================================

def plot_training(
    history: TrainingHistory,
    ax: plt.Axes,
    show_batches: bool = True,
    metric: str = "loss",
    label: Optional[str] = None,
):
    if metric == "loss":
        train_fn = lambda s: s.train_loss
        test_fn = lambda s: s.test_loss
    elif metric == "accuracy":
        train_fn = lambda s: s.train_acc
        test_fn = lambda s: s.test_acc
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Train
    train_data = extract_series(history, train_fn)
    plot_series(
        ax,
        train_data,
        label=label or f"Train {metric}",
        show_batches=show_batches,
        ylabel=metric.capitalize(),
        title=f"{metric.capitalize()} evolution",
    )

    # Test (preserving original logic)
    test_data = extract_series(history, test_fn)

    valid_x = []
    valid_y = []

    for x, y in zip(test_data["epoch_x"], test_data["epoch_y"]):
        if y is not None:
            valid_x.append(x)
            valid_y.append(y)

    if valid_y:
        ax.plot(
            valid_x,
            valid_y,
            linestyle="--",
            linewidth=2.0,
            marker="s",
            label=(label + " (test)") if label else f"Test {metric}",
            zorder=3,
        )


def plot_fidelities(
    history: TrainingHistory,
    optimal_state: np.ndarray,
    ax: plt.Axes,
    label: Optional[str] = None,
):
    def fidelity_fn(step):
        return step.fidelity(optimal_state)

    data = extract_series(history, fidelity_fn)

    plot_series(
        ax,
        data,
        label=label or "Fidelity",
        show_batches=True,
        ylabel="Fidelity",
        title="Fidelity evolution",
    )


# =========================================================
# ------------------------ MAIN ----------------------------
# =========================================================

if __name__ == "__main__":
    root = find_root()
    exp_folder = f'{root}/CoherentQMLPython/data/experiments/simple'

    gd_history = TrainingHistory.load_pickle(f'{exp_folder}/gd_training.pickle')
    coherent_history = TrainingHistory.load_pickle(f'{exp_folder}/coherent_training.pickle')

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)

    plot_training(
        history=gd_history,
        ax=ax[0],
        show_batches=True,
        metric='loss',
        label='GD',
    )

    plot_training(
        history=coherent_history,
        ax=ax[1],
        show_batches=True,
        metric='loss',
        label='Coherent',
    )

    save_path = f'{exp_folder}/comparison.png'
    fig.savefig(save_path, dpi=300)

    print(f'Plot generated and saved in: {save_path}')
    
    # Fidelities
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), sharex=True, sharey=True)
    # optimal_state = 