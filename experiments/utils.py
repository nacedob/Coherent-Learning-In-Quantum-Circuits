def import_root():
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import_root()

from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
from src.data import mnist_pca_digits, multi_boundaries
from src.models.datareuploading import DataReuploadingQNN
import numpy as np
from src.models.history import TrainingHistory
from pathlib import Path
import matplotlib.pyplot as plt




def find_root(start_path: str | Path = ".") -> Path:
    current = Path(start_path).resolve()

    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent

    raise FileNotFoundError("No .git folder found in any parent directory")


def get_dataset_boundaries(n_train: int, n_test: int,
                           seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return multi_boundaries(
        x_min=-np.pi,
        x_max=np.pi,
        boundaries=[-np.pi / 3, 2 * np.pi / 3],
        n_train=n_train,
        n_test=n_test,
        seed=seed
    )
    
    


def get_dataset_mnist(n_train: int, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return mnist_pca_digits(n_train=n_train, n_test=n_test, seed=seed)



def plot_distributions(history: TrainingHistory, folder: Path) -> None:

    psis = history.psis()[0, 0, :, :]  # shape: (dims, times + 1)
    fig, axis = plt.subplots(ncols=3, nrows=2, figsize=(10, 7), sharex=True, sharey=True)
    axis = axis.flatten()

    # Plot times
    times = psis.shape[-1] - 1
    quarter = times // 4

    # Get probabilities for [0, 0.25, 0.5, 0.75, 1] * TIME
    plot_times = [0, quarter, times // 2, 3 * quarter, times]
    indices = np.arange(psis.shape[0])
    for i, t in enumerate(plot_times):
        psi = psis[:, t]
        prob = np.abs(psi)**2
        if not np.isclose(np.sum(prob), 1):
            raise ValueError(f"Probabilities do not sum to 1. Sum: {np.sum(prob)}")
        axis[i].bar(indices, prob, width=(indices[1] - indices[0]))
        axis[i].set_xlabel('Index')
        axis[i].set_ylabel('Probability')
        axis[i].set_title(f'Time {t}')

    fig.tight_layout()
    plot_path = folder / 'training_distributions.png'
    fig.savefig(plot_path, dpi=300)
    print(f'Distribution plot generated and save in: {plot_path}')
    
def plot_distributions_2d(history, folder: Path) -> None:

    psis = history.psis()[0, 0, :, :]  # (dim, time+1)
    dim, T = psis.shape

    d = int(np.sqrt(dim))
    if d * d != dim:
        raise ValueError("State is not compatible with 2-parameter decomposition.")

    psis = psis.reshape(d, d, T)

    probs_all = np.abs(psis) ** 2
    z_max = probs_all.max()

    norm = Normalize(vmin=0.0, vmax=z_max)
    cmap = cm.viridis

    fig = plt.figure(figsize=(12, 8))
    times = [0, T // 5, 2 * T // 5, 3 * T // 5, 4 * T // 5, T - 1]

    fig = plt.figure(figsize=(12, 8))

    axes = [
        fig.add_subplot(2, 3, i + 1, projection="3d")
        for i in range(len(times))
    ]


    x = np.arange(d)
    y = np.arange(d)
    X, Y = np.meshgrid(x, y)

    X = X.ravel()
    Y = Y.ravel()

    dx = dy = 0.8

    for ax, t in zip(axes, times):

        psi = psis[:, :, t]
        Z = np.abs(psi) ** 2
        Z = Z.ravel()

        colors = cmap(norm(Z))

        ax.bar3d(
            X,
            Y,
            np.zeros_like(Z),
            dx,
            dy,
            Z,
            color=colors,
            shade=True
        )

        frac = t / (T - 1) if T > 1 else 0.0

        ax.set_title(f"t = T * {frac:.2f} ({t})")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\theta_1$")
        ax.set_zlabel("Probability")

        ax.set_zlim(0, z_max)
        ax.view_init(elev=35, azim=45)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(probs_all)

    fig.colorbar(
        mappable,
        ax=axes,
        shrink=0.6,
        pad=0.1,
        label="Probability"
    )


    plot_path = folder / "training_distributions_3d.png"
    fig.savefig(plot_path, dpi=300)

    print(f"Saved to: {plot_path}")



def plot_landscape(
    found_parameters: np.ndarray,
    n_qubits: int,
    n_layers: int,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_folder: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    qnn = DataReuploadingQNN(n_qubits=n_qubits, n_layers=n_layers, seed=seed)
    qnn.parameters = found_parameters
    qnn.plot_landscape(ax=ax, X=X_train, y=y_train)
    fig.savefig(save_folder, dpi=300)
    print(f'Plot generated and save in: {save_folder}')
