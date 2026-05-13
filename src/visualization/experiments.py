from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from src.models.history import TrainingHistory
from src.models.datareuploading import DataReuploadingQNN
from experiments.final_experiment import load_dataset, ExperimentConfig
from matplotlib.colors import Normalize, LinearSegmentedColormap, LightSource


VIOLET = '#6C3C84'
LAVENDER = "#9a99ff"
PINK = '#F48FB1'
BASEFONTSIZE = 30
# --- Internal Helper for Consistent Styling ---


def _apply_standard_theme(ax: plt.Axes, title: str, xlabel: str, ylabel: str, base_fontsize: int):
    """Applies the clean-academic theme to any 2D axis."""
    plt.rcParams.update({
        "text.usetex": True,          # use LaTeX
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],   # LaTeX default font
    })
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)
    ax.set_xlabel(rf"$\mathrm{{{xlabel}}}$", fontsize=base_fontsize)
    ax.set_ylabel(rf"$\mathrm{{{ylabel}}}$", fontsize=base_fontsize)
    # Manual Title placement using text2D
    ax.text(0.5, 1.05, title,
            transform=ax.transAxes,
            fontsize=base_fontsize * 1.3,
            fontweight='bold',
            ha='center',
            va='bottom')

    # Tick size
    ax.tick_params(axis='both', labelsize=base_fontsize * 0.7)

# =========================================================
# -------------------- Series utils -----------------------
# =========================================================


def plot_training(
    history,
    ax: plt.Axes,
    metric: str = "loss",
    show_batches: bool = True,
    label: Optional[str] = None,
    color: str = VIOLET,
    base_fontsize: int = BASEFONTSIZE,
    title: Optional[str] = None,
    t: None | str = None
):
    train_vals = [step.train_loss for step in history.steps]
    test_vals = [step.test_loss for step in history.steps]
    batches = range(1, len(train_vals) + 1)

    # Plot Training
    if t is None or t == 'train':
        ax.plot(batches, train_vals, color=color, lw=4, label=f"{label or ''}", zorder=4, marker="o", ms=5)
    # ax.fill_between(batches, train_vals, color=color, alpha=0.1, zorder=2)

    # Plot Testing
    if t is None or t == 'test':
        ax.plot(batches, test_vals, color=color, lw=4, label=f"{label or ''}", zorder=3, marker="o", ms=5)

    # Apply Style
    _apply_standard_theme(ax, title or f"{metric.capitalize()} Evolution",
                          "Batch" if show_batches else "Step",
                          metric.capitalize(), base_fontsize)

    ax.legend(frameon=False, fontsize=base_fontsize, loc='upper right')
    ax.set_ylim(bottom=0)

# =========================================================
# ---------------- Fidelity visualization -----------------
# =========================================================


def plot_fidelities(history, optimal_state, ax, label=None, base_fontsize=10):
    def get_step_fidelity(step):
        # State extraction logic
        state = step.psi_evolution if step.epoch == 0 else step.psi_evolution[:, -1]
        return np.abs(np.vdot(state, optimal_state)) ** 2

    fidelities = [get_step_fidelity(s) for s in history.steps]
    batches = range(1, len(fidelities) + 1)

    # Plot Fidelity
    ax.plot(batches, fidelities, marker="o", ms=4, color=VIOLET, lw=1.5, label=label, zorder=3)
    ax.fill_between(batches, fidelities, color=VIOLET, alpha=0.1, zorder=2)

    # Apply Style
    _apply_standard_theme(ax, "Fidelity Evolution", "Batch", "Fidelity", base_fontsize)

    ax.set_ylim(0, 1.05)
    if label:
        ax.legend(frameon=False, fontsize=base_fontsize * 0.9)


# =========================================================
# ----------------- Utilities -----------------------------
# =========================================================


def find_root(start_path: str | Path = ".") -> Path:
    current = Path(start_path).resolve()

    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent

    raise FileNotFoundError("No .git folder found")


# =========================================================
# ---------------- Distribution plot ----------------------
# =========================================================
def get_cmap(z: float, vmin: float, vmax: float):
    # cmap = LinearSegmentedColormap.from_list(
    #     "violet_pink",
    #     # ["#b199c5", "#9b6bb8", "#d08ac9", "#8a02b3"],  # 6302b3
    #     ["#d62c95", "#6c1088", "#6c3bad", "#55475e", 'gray'][::-1],
    #     N=256
    # )
    cmap = LinearSegmentedColormap.from_list(
        "custom_surface",
        [
            "#340d47",  # Deep Midnight (Your original dark start)
            "#5a189a",  # Deep Violet (Strong purple presence)
            "#9d8abf",  # Muted Purple-Gray (The bridge)
            "#8b79b9",  # Your specific Gray (Bright finish)
        ], N=256
    )
    cmap = cmap.reversed()

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    colors = cmap(norm(z))
    return cmap, colors

    cmap = cm.get_cmap("magma")

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    colors = cmap(norm(z))

    return cmap, colors


def plot_distribution_2d(psi: np.ndarray, ax: plt.Axes, vmin: float | None = None, vmax: float | None = None,
                         title: str | None = None, base_fontsize: int = BASEFONTSIZE):

    # 1. Data Prep
    if psi.ndim == 1:
        d = int(np.sqrt(psi.size))
        psi = psi.reshape(d, d)
    else:
        d = psi.shape[0]

    probs = np.abs(psi) ** 2
    x, y = np.meshgrid(np.linspace(-np.pi, np.pi, len(psi), endpoint=True),
                       np.linspace(-np.pi, np.pi, len(psi), endpoint=True))
    x, y, z = x.ravel(), y.ravel(), probs.ravel()

    # Use your existing get_cmap logic here
    # Assuming get_cmap returns a list of colors for each bar
    cmap, colors = get_cmap(
        vmin=vmin if vmin is not None else probs.min(),
        vmax=vmax if vmax is not None else probs.max(),
        z=z
    )

    # 2. Draw Bars with thinner gaps for a "cleaner" texture
    ax.bar3d(x, y, np.zeros_like(z), 0.8, 0.8, z, color=colors, shade=False)

    # 3. BEAUTIFICATION & ZOOM
    ax.view_init(elev=30, azim=225)  # Changed angle for better perspective

    # ZOOM: The magic happens here
    ax.dist = 12  # Lower = Closer (Default is 10)
    ax.set_box_aspect((1, 1, 1.1))  # Flatten Z-axis to make the distribution "pop" more

    # 5. TITLES & LABELS (Manual placement since axis is off)
    font_title = base_fontsize * 1.2
    font_label = base_fontsize

    if title:
        ax.text2D(0.5, 0.9, title,
                  transform=ax.transAxes,
                  fontsize=base_fontsize * 1.3,
                  fontweight='bold',
                  ha='center',
                  va='bottom')
    # Place labels at the edges of the base
    ax.set_ylabel(r'$\theta_0$', fontsize=font_label)
    ax.set_xlabel(r'$\theta_1$', fontsize=font_label)
    points = [-np.pi, 0, np.pi]
    labels = [r'$-\pi$', r'$0$', r'$\pi$']
    ax.set_xticks(points)
    ax.set_xticklabels(labels, fontsize=font_label - 2)
    ax.set_yticks(points)
    ax.set_yticklabels(labels, fontsize=font_label - 2)

    ax.set_zticks([])

    # TUrn off axis
    # ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # 6. Z-Limit handling
    if vmax is not None:
        ax.set_zlim(0, vmax)
    else:
        ax.set_zlim(0, probs.max() * 1.1)

    # Final visual polish: Make the figure background white
    ax.set_facecolor('white')


# =========================================================
# ---------------- Loss landscape -------------------------
# =========================================================


def plot_landscape(
    qnn,
    X_train,
    y_train,
    save_folder: Path,
    resolution: int = 512,
    span: float = np.pi,
    base_fontsize: int = BASEFONTSIZE,
):
    font_title = base_fontsize * 1.5
    font_label = base_fontsize
    font_ticks = base_fontsize * 0.8

    fig = plt.figure(figsize=(10, 10), facecolor='white')  # Square figsize helps 3D fill space
    ax = fig.add_subplot(111, projection="3d")

    # Data Processing
    loss_fn = qnn.loss
    base = qnn.parameters.copy().flatten()
    grid = np.linspace(-span, span, resolution)
    P0, P1 = np.meshgrid(grid, grid)
    Z = np.zeros_like(P0)
    params = base.copy()

    for i in range(resolution):
        for j in range(resolution):
            params[0] = P0[i, j]
            params[1] = P1[i, j]
            Z[i, j] = loss_fn(parameters=params.reshape(*qnn.shape), X=X_train, y=y_train)

    # Aesthetics
    # color_sequence = ["#de49a5", "#4f0864", "#5a2a99", "#55475e", 'gray']
    # cmap = LinearSegmentedColormap.from_list('custom_cmap', color_sequence, N=256)
    cmap, _ = get_cmap(Z, vmin=Z.min(), vmax=Z.max())
    cmap = cmap.reversed()
    ls = LightSource(azdeg=180, altdeg=65)
    rgb = ls.shade(Z, cmap=cmap, blend_mode='soft', vert_exag=0.1)

    surf = ax.plot_surface(
        P0, P1, Z,
        rstride=1, cstride=1,
        facecolors=rgb,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.9
    )

    # Optimal point
    z_star = loss_fn(parameters=base.reshape(*qnn.shape), X=X_train, y=y_train)
    ax.scatter(base[0], base[1], z_star, color="#000000", s=180, edgecolors='white', linewidth=1.5, zorder=100)

    # --- THE ZOOM FIXES ---
    ax.view_init(elev=8, azim=167)
    # ax.set_axis_off()
    ax.set_zticks([])
    ax.grid(False)
    ax.set_xlabel("$\\theta_0$", fontsize=font_label, labelpad=20)
    ax.set_ylabel("$\\theta_1$", fontsize=font_label, labelpad=20)

    # ax.text(.5, 0.0, 0.0, r"$\theta_0$", fontsize=font_label)
    # ax.text(0.0, .5, 0.0, r"$\theta_1$", fontsize=font_label)

    ax.set_xticks([-span, 0, span])
    ax.set_yticks([-span, 0, span])
    ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax.tick_params(axis='both', labelsize=base_fontsize * 0.8)
    ax.invert_yaxis()

    # 1. Scaling the axes to remove empty space around the "cube"
    ax.set_box_aspect((1, 1, 0.8))  # X, Y, Z proportions. Flattening Z helps visibility.

    # 2. Adjusting the camera distance (Lower = Closer/Larger)
    # Use ax.set_proj_type('ortho') if you want no perspective distortion
    ax.dist = 9  # Default is usually 10. Lower values zoom in.

    plt.title('Loss Function Landscape', fontsize=font_title, fontweight='bold', y=0.95)

    # Colorbar positioning
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(Z)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.4, aspect=15, pad=-0.05)  # Negative pad moves it closer
    cbar.ax.tick_params(labelsize=font_ticks)
    cbar.outline.set_visible(False)

    # Final tight layout adjustment
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Force plot to edges

    save_folder.mkdir(parents=True, exist_ok=True)
    out = save_folder / "landscape.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f'Landscape plot generated and save in: {out}')


def apply_global_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.0,

        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",

        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,

        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "legend.frameon": False,
        "legend.fontsize": 10,

        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


if __name__ == "__main__":
    apply_global_style()
    exp_folder = Path(f'{find_root()}/CoherentQMLPython/data/experiments/simple')

    seed = 0
    dataset = 'simple_boundaries'
    n_train = 100
    n_test = 500
    config = ExperimentConfig(DATASET=dataset, N_TRAIN=n_train, N_TEST=n_test, SEED=seed)
    X_train, y_train, X_test, y_test = load_dataset(config=config)

    for folder in exp_folder.iterdir():

        qubits = int(folder.name.split('_')[0])
        layers = int(folder.name.split('_')[1])
        print(f'{"=" * 20} Qubits: {qubits}, Layers: {layers}, folder: {folder} {"=" * 20}')
        history = TrainingHistory.load_pickle(folder / 'coherent_training.pickle')

        # Plot training evolution
        fig, ax = plt.subplots(figsize=(10, 7))
        plot_training(history=history, ax=ax, show_batches=True, metric='loss')
        fig.savefig(folder / 'coherent_training_evolution.png', dpi=300)
        print(f'Plot generated and save in: {folder / "coherent_training_evolution.png"}')
        qnn = DataReuploadingQNN.load(folder / 'coherent_qnn.pickle')

        # Plot fidelity evolution
        optimalstate = qnn.get_optimal_state(
            X_train=X_train,
            y_train=y_train,
            grid_size=128,
            simulator_qubits=config.SIMULATOR_QUBITS,
            left=-np.pi,
            right=np.pi,
            deviation=None
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        plot_fidelities(history=history, optimal_state=optimalstate, ax=ax)
        fig.savefig(folder / 'coherent_fidelities.png', dpi=300)
        print(f'Fidelity plot generated and save in: {folder / "coherent_fidelities.png"}')
        plt.close(fig)

        # Plot probability evolution
        if qubits * layers == 2:
            fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 7), subplot_kw={"projection": "3d"})

            batches = history.batches
            if batches is None:
                raise ValueError("Batches not found in history")

            total_time = history.get(-1, -1).psi_evolution.shape[1]

            # displayed_times = [
            #     1,
            #     total_time // 3,
            #     2 * total_time // 3,
            #     total_time - 1
            # ]
            displayed_batches = [
                1,
                batches // 3,
                2 * batches // 3,
                batches - 1
            ]

            ax = ax.ravel()

            for i, b in enumerate(displayed_batches):
                # for i, b in enumerate(displayed_batches):
                # psi = history.get(1, 1).psi_evolution[:, b]
                psi = history.get(1, b).psi_evolution[:, 0]
                plot_distribution_2d(psi=psi, ax=ax[i])
            fig.savefig(folder / 'coherent_probabilities.png', dpi=300)
            print(f'Probability plot generated and save in: {folder / "coherent_probabilities.png"}')
            plt.close(fig)

        # Function landscape
            plot_landscape(
                qnn=qnn,
                X_train=X_test,
                y_train=y_test,
                save_folder=folder
            )
# =========================================================
# ------------------ Dataset visualization ----------------
# =========================================================


def plot_dataset(X, y, ax, title=None, elev=20, azim=45, base_fontsize=10):
    """
    Smarter binary classification plotter.
    Handles 1D, 2D, and 3D with consistent thematic styling.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    dims = X.shape[1]
    unique_classes = np.unique(y)

    # Mapping classes to your global color variables
    # Class 0 -> Lavender, Class 1 -> Violet (usually better for the "active" class)
    colors = [LAVENDER, VIOLET]
    palette = dict(zip(unique_classes, colors))

    # Identify if we are in 3D to apply specific cleaning
    is_3d = (dims == 3)

    if dims == 1:
        _plot_1d(X, y, palette, ax)
    elif dims == 2:
        _plot_2d(X, y, palette, ax)
    elif dims == 3:
        _plot_3d(X, y, palette, elev, azim, ax)
    else:
        raise ValueError(f"Unsupported dimensionality: {dims}")

    # Apply the factorized theme
    _apply_dataset_theme(ax, title or f"{dims}D Dataset", dims, base_fontsize, is_3d)


def _apply_dataset_theme(ax, title, dims, base_fontsize, is_3d=False):
    """Applies consistent styling to dataset plots."""
    font_title = base_fontsize * 1.3
    font_label = base_fontsize

    if is_3d:
        # 3D Specific Cleanup
        ax.grid(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_zticks([])
        ax.set_zlabel("x2", fontsize=font_label)
    else:
        # 2D/1D Specific Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if dims == 1:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])

    ax.set_xlabel("x0", fontsize=font_label)
    if dims > 1:
        ax.set_ylabel("x1", fontsize=font_label)

    # Manual Title Placement (text2D works for both 2D and 3D axes)
    ax.text(0.5, 1.05, title,
            transform=ax.transAxes,
            fontsize=font_title,
            fontweight='bold',
            ha='center', va='bottom')


# --- Internal Dispatchers ---

def _plot_1d(X, y, palette, ax):
    # Jitter to avoid point overlap on a single line
    y_jitter = np.random.normal(0, 0.02, size=len(X))
    for cls, color in palette.items():
        mask = (y == cls)
        ax.scatter(X[mask, 0], y_jitter[mask], c=color, label=f"Class {cls}",
                   alpha=0.7, s=40, edgecolor='white', linewidth=0.5)
    ax.set_ylim(-0.5, 0.5)


def _plot_2d(X, y, palette, ax):
    for cls, color in palette.items():
        mask = (y == cls)
        ax.scatter(X[mask, 0], X[mask, 1], c=color, label=f"Class {cls}",
                   alpha=0.7, s=40, edgecolor='white', linewidth=0.5)


def _plot_3d(X, y, palette, elev, azim, ax):
    # Ensure the provided ax is actually a 3D axis
    ax.view_init(elev=elev, azim=azim)
    ax.dist = 9  # Zoom in slightly

    for cls, color in palette.items():
        mask = (y == cls)
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=color,
                   label=f"Class {cls}", alpha=0.7, s=30, edgecolor='white', linewidth=0.3)
