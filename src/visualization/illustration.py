from icecream import ic
import os
from experiments.final_experiment import get_model, load_dataset
from .experiments import get_cmap, plot_distribution_2d
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import numpy as np

plt.rcParams.update({
        "text.usetex": True,          # use LaTeX
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],   # LaTeX default font
})
FONTSIZE = 20


def create_psi_illustrations(
    n_qubits: int = 4,
    times=(0.0, 0.35, 0.6, 1.0),
    noise_level: float = 0.6,
    seed: int = 0,
    out_dir: str = "data/plots/illustration",
    base_fontsize: int = 18
):
    """
    Generates and saves 3D bar visualizations of an evolving quantum-like state.
    """

    N = 2 ** n_qubits
    os.makedirs(out_dir, exist_ok=True)

    # ==========================================================
    # INITIAL STATE (uniform superposition)
    # ==========================================================
    def psi_initial(d):
        psi = np.ones((d, d), dtype=complex)
        psi /= np.sqrt(d * d)
        return psi

    # ==========================================================
    # TARGET STATE (noisy Gaussian mixture)
    # ==========================================================
    def psi_target(d):
        rng = np.random.default_rng(seed)

        x = np.linspace(-1, 1, d)
        X, Y = np.meshgrid(x, x)

        g1 = 0.5 * np.exp(-((X - 0.4)**2 + (Y + 0.5)**2) / 0.2)
        g2 = 0.1 * np.exp(-((X + 0.6)**2 + (Y - 0.3)**2) / 0.3)

        psi = g1 + g2

        noise = rng.normal(0.0, 1.0, size=(d, d))
        noise *= np.exp(-(X**2 + Y**2))

        psi = psi + noise_level * noise
        psi /= np.linalg.norm(psi)

        return psi.astype(complex)

    # ==========================================================
    # EVOLUTION
    # ==========================================================
    def evolve(d, t, noise_level):
        psi0 = psi_initial(d)
        psiT = psi_target(d)

        psi = (1 - t)**2 * psi0 + t**2 * psiT

        x = np.linspace(-1, 1, d)
        X, Y = np.meshgrid(x, x)

        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, size=(d, d))
        phase = np.exp(1j * (2.5 * X - 1.8 * Y) * t)
        psi = psi * phase + noise_level * noise

        psi /= np.linalg.norm(psi)
        return psi

    # ==========================================================
    # GLOBAL SCALING (for consistent visualization)
    # ==========================================================ç
    noise_levels = [0, 0.01, 0.1, 0]
    states = [evolve(N, times[i], noise_level=noise_levels[i]) for i in range(len(times))]
    all_probs = [np.abs(states)**2 for t in times]
    global_vmax = np.max(all_probs)
    global_vmin = 0.0

    # ==========================================================
    # PLOTTING
    # ==========================================================
    for i, psi in enumerate(states, start=1):

        fig = plt.figure(figsize=(8, 7), dpi=220)
        ax = fig.add_subplot(111, projection="3d")

        plot_distribution_2d(
            psi=psi,
            ax=ax,
            vmin=global_vmin,
            vmax=global_vmax,
            base_fontsize=base_fontsize
        )

        path = os.path.join(out_dir, f"psi_distribution_{i}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


def create_landscape_illustrations(
):
    """
    Generates and saves 3D loss landscape illustrations.
    """

    out_dir = "data/plots/illustration"
    os.makedirs(out_dir, exist_ok=True)
    n = 128

    # Load experiment model
    # Mock class
    class Config:
        N_QUBITS = 1
        N_LAYERS = 2
        SEED = 0
        DATASET = 'simple_boundaries'
        N_TRAIN = 100
        N_TEST = 500

    config = Config()
    qnn = get_model(config)
    X_train, y_train, X_test, y_test = load_dataset(config)


    # # ==========================================================
    # # Grid
    # # ==========================================================
    x = np.linspace(-np.pi, np.pi, n)
    y = np.linspace(-np.pi, np.pi, n)
    X, Y = np.meshgrid(x, y)

    # ==========================================================
    # Output folder
    # ==========================================================
    os.makedirs(out_dir, exist_ok=True)

    get_cmap_ = lambda Z: get_cmap(z=Z, vmax=Z.max(), vmin=Z.min())[0]

    # ==========================================================
    # Loop
    # ==========================================================
    # for i, V in enumerate(Zs, start=1):

    # Z = V(X, Y)
    base = qnn.parameters.copy().flatten()
    Z = np.zeros_like(X)
    params = base.copy()
    for batch, (x, y) in enumerate(zip(X_train, y_train)):
        def compute_point(i, j):
            local_params = params.copy()
            local_params[0] = X[i, j]
            local_params[1] = Y[i, j]

            return i, j, qnn.loss(
                parameters=local_params.reshape(*qnn.shape),
                X=X_train,
                y=y_train
            )

        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda ij: compute_point(*ij),
                [(i, j) for i in range(n) for j in range(n)]
            )

        for i, j, value in results:
            Z[i, j] = value

        cmap = get_cmap_(Z)

        fig = plt.figure(figsize=(10, 8), dpi=220)
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            X, Y, Z,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.98,
            edgecolor="none",
        )

        ax.contour(
            X, Y, Z,
            zdir='z',
            offset=Z.min() - 0.6,
            levels=25,
            cmap=cmap,
            linewidths=0.8
        )

        # ======================================================
        # Labels
        # ======================================================
        ax.set_xlabel(r"$\theta_1$", fontsize=FONTSIZE)
        ax.set_ylabel(r"$\theta_2$", fontsize=FONTSIZE)
        ax.set_zlabel("Loss", fontsize=FONTSIZE, rotation=90)

        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_zlim(Z.min() - 0.6, Z.max())

        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_yticks([0, np.pi])
        ax.set_zticks([])

        ax.set_xticklabels([r"$-\pi\quad\;$", "0", r"$\pi$"], fontsize=FONTSIZE - 2)
        ax.set_yticklabels(["0", r"$\pi$"], fontsize=FONTSIZE - 2)

        # ======================================================
        # View + cleanup
        # ======================================================
        # ax.view_init(elev=28, azim=-132)   # AQUI
        ax.view_init(elev=30, azim=225)
        ax.grid(False)

        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.zaxis.line.set_color((0, 0, 0, 0))

        # ======================================================
        # Save
        # ======================================================
        path = os.path.join(out_dir, f"landscape_full.png")
        plt.savefig(path, bbox_inches="tight", dpi=300, transparent=True)
        plt.close(fig)

        print(f"Saved figure in: {path}")
        exit()


if __name__ == "__main__":
    create_landscape_illustrations()
    # create_psi_illustrations()
