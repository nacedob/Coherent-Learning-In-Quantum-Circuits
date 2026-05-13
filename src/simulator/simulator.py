from icecream import ic
import pyfftw
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Callable, Optional, Sequence, Literal


class QuantumEvolutionSimulator:
    """Multi-Dimensional Quantum Annealing Simulator using Split-Operator FFT technique."""

    def __init__(
        self,
        qubits: Sequence[int],
        schedule_exponent_kinetic: Optional[Callable[[float, float], float]] = None,
        schedule_exponent_potential: Optional[Callable[[float, float], float]] = None,
        result_type: str = 'max',
        fft_backend: Literal['numpy', 'pyfftw'] = 'pyfftw'
    ):
        assert result_type in ['max', 'expected_value'], "result_type must be either 'max' or 'expected_value'"
        assert fft_backend in ["numpy", "pyfftw"], "fft_backend must be 'numpy' or 'pyfftw'"
        self.qubits = qubits
        self.fft_backend = fft_backend
        self.n_qubits = sum(qubits)
        self.n_dims = len(qubits)
        self._points_per_dim = [int(2**q) for q in qubits]
        self.dim = int(np.prod(self._points_per_dim))
        self.result_type = result_type

        self.solution_interpolant: Optional[interp1d] = None
        self._cached_left: Optional[float] = None
        self._cached_right: Optional[float] = None
        self._cached_target_function: Optional[Callable] = None

        # Init pyfftw
        self._fftw_enabled = (fft_backend == "pyfftw")

        if self._fftw_enabled:
            self._fftw_plan_fwd = None
            self._fftw_plan_inv = None
            self._fft_buffer = None
            self._init_fftw()

        self.schedule_exponent_kinetic = schedule_exponent_kinetic or self._default_exponent_kinetic
        self.schedule_exponent_potential = schedule_exponent_potential or self._default_exponent_potential

        self._state_history = np.array([])  # shape (dim, time_steps + 1)

    def _default_exponent_kinetic(self, t: float, T: float) -> float:
        return (1 - t / T)

    def _default_exponent_potential(self, t: float, T: float) -> float:
        return (t / T)

    def _init_fftw(self):

        shape = self._points_per_dim

        self._fft_buffer = pyfftw.empty_aligned(shape, dtype='complex128')

        self._fftw_plan_fwd = pyfftw.FFTW(
            self._fft_buffer,
            self._fft_buffer,
            axes=tuple(range(len(shape))),
            direction="FFTW_FORWARD",
            flags=["FFTW_MEASURE"]
        )

        self._fftw_plan_inv = pyfftw.FFTW(
            self._fft_buffer,
            self._fft_buffer,
            axes=tuple(range(len(shape))),
            direction="FFTW_BACKWARD",
            flags=["FFTW_MEASURE"]
        )

    # ---------------------
    # Grid utilities
    # ---------------------

    def _get_1d_grid(self, start: float, end: float, points: int) -> np.ndarray:
        """
        Generates a discrete 1D spatial coordinate grid.

        The endpoint is omitted to enforce perfect periodic boundary conditions, which is a strict mathematical
        requirement for the Fast Fourier Transform (FFT) to work correctly without aliasing.
        """
        return np.linspace(start, end, points, endpoint=False)

    def _get_nd_grids(self, left: float, right: float) -> tuple[np.ndarray, ...]:
        """
        Constructs the full N-dimensional spatial coordinate space.

        The resulting tuple of meshes is used to evaluate the continuous Potential Energy function V(x, y, ...)
        across the entire grid simultaneously.
        """
        grids_1d = [self._get_1d_grid(left, right, p) for p in self._points_per_dim]
        return np.meshgrid(*grids_1d, indexing='ij')

    def _get_kinetic_energy_grid(self, left: float, right: float) -> np.ndarray:
        """
        Constructs the N-dimensional kinetic energy operator in momentum space.

        It builds an N-dimensional momentum meshgrid and computes the total kinetic energy as the sum of the squared
        momenta across all dimensions: K = sum (k_i^2) / 2
        """
        k_grids_1d = []
        for p in self._points_per_dim:
            dx = (right - left) / p
            # Calculate momentum frequencies
            k = 2 * np.pi * np.fft.fftfreq(p, d=dx)
            k_grids_1d.append(k)

        # Create N-dim momentum meshgrid
        K_mesh = np.meshgrid(*k_grids_1d, indexing='ij')

        # Kinetic energy is sum(k_i^2 / 2)
        K_grid = sum((k_mesh**2) / 2 for k_mesh in K_mesh)
        return K_grid

    def _get_potential_grid(self, target_function: Callable, left: float, right: float) -> np.ndarray:
        spatial_grids = self._get_nd_grids(left, right)
        V = target_function(*spatial_grids)
        return V
    # ---------------------
    # Quantum evolution
    # ---------------------

    def _kinetic_step(
        self,
        psi: np.ndarray,
        phase_K: np.ndarray,
    ) -> np.ndarray:

        if self.fft_backend == "numpy":
            psi_k = np.fft.fftn(psi)
            psi_k *= phase_K
            return np.fft.ifftn(psi_k)

        # PyFFTW implementation
        self._fft_buffer[:] = psi
        self._fftw_plan_fwd()
        self._fft_buffer *= phase_K
        self._fftw_plan_inv()
        psi[:] = self._fft_buffer

        return psi

    def evolve(
        self,
        target_function: Callable,
        left: float,
        right: float,
        total_time: float,
        time_steps: int,
        initial_state: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Executes the time evolution of the quantum state using the Second-Order
        Split-Operator FFT method.

        This method bypasses the memory-intensive creation of dense Hamiltonian matrices.
        Instead, it propagates the N-dimensional wavefunction by applying the time-evolution operator in alternating
        spaces. It applies half a step of the potential energy V in position space, transforms to momentum space via FFT
        to apply a full step of the kinetic energy K, and transforms back to position space for the final half step of V
        Args:
            - `target_function` (Callable)
                The continuous N-dimensional objective function. Must accept N separate meshgrid arrays.
            - `left` (float)
                The lower spatial boundary for all dimensions.
            - `right` (float)
                The upper spatial boundary for all dimensions.
            - `total_time` (float)
                The total duration of the quantum annealing schedule (T).
            - `time_steps` (int, optional)
                The number of discrete time intervals (ΔT). Higher values reduce Trotter error and result in
                smoother evolution. Defaults to 500.
            - `initial_state` (np.ndarray | None, optional)
                The starting wavefunction. If None, the system initializes into a uniform superposition across the
                entire N-dimensional grid, which is standard for quantum annealing.
                Defaults to None.

        Returns:
            np.ndarray: the optimal solution as a np.ndarray.

            The time-evolved state history is stored internally as a 1D interpolant
            (`self.solution_interpolant`) for querying the wavefunction at any time t.
        """
        # Save arguments to cache for visualization functions
        self._cached_target_function = target_function
        self._state_history = np.empty((self.dim, time_steps + 1), dtype=np.complex64)  # shape (dim, time_steps + 1)
        self._cached_left = left
        self._cached_right = right

        # Precompute N-dimensional Position and Momentum grids
        V = self._get_potential_grid(target_function, left, right)    # Potential Energy Grid
        K = self._get_kinetic_energy_grid(left, right)                # Kinetic Energy Grid

        if initial_state is None:
            # Uniform superposition, properly shaped to N-dimensions
            psi = np.ones(self._points_per_dim, dtype=np.complex64) / np.sqrt(self.dim)
        else:
            initial_state = np.asarray(initial_state, dtype=np.complex64)
            if not (
                (initial_state.ndim == 1 and initial_state.shape[0] == self.dim) or
                (initial_state.ndim > 1 and initial_state.shape == self.qubits)
            ):
                raise ValueError(
                    f"Initial state shape {initial_state.shape} does not match expected dimension {self.dim}. "
                    f"Number of simulator qubits: {self.qubits}."
                )
            if not np.isclose(np.linalg.norm(initial_state), 1, atol=1e-3):
                raise ValueError(
                    f"Initial state must be normalized. Got norm {np.linalg.norm(initial_state)}"
                )
            initial_state = initial_state / np.linalg.norm(initial_state)
            psi = initial_state.reshape(self._points_per_dim).copy()

        dt = total_time / time_steps
        times = np.linspace(0, total_time, time_steps + 1)

        # Initialize history
        self._state_history[:, 0] = psi.flatten()

        for i, t_curr in enumerate(times[:-1]):
            if verbose:
                print(f"Evolving system. Step {i + 1}/{time_steps}", end='\r')

            # Use midpoint schedules for 2nd-order Trotter
            t_mid = t_curr + 0.5 * dt
            a_mid = self.schedule_exponent_kinetic(t_mid, total_time)
            b_mid = self.schedule_exponent_potential(t_mid, total_time)

            # Trotterized evolution
            # 1. Half-step Potential (Position Space)
            phase_V = np.exp(-1j * (0.5 * dt) * b_mid * V)
            psi *= phase_V

            # 2. Full-step Kinetic (Momentum Space via N-dimensional FFT)
            phase_K = np.exp(-1j * dt * a_mid * K)
            psi = self._kinetic_step(psi, phase_K)

            # 3. Half-step Potential (Position Space)
            psi *= phase_V

            # Save flattened state
            self._state_history[:, i + 1] = psi.flatten()

        self.solution_interpolant = interp1d(times, self._state_history, axis=1, kind='linear')
        if verbose:
            print()

        return self.get_result()

    def get_final_wavefunction(self):
        if self.solution_interpolant is None:
            raise RuntimeError("Run evolve() before querying.")
        return self.solution_interpolant(self.solution_interpolant.x[-1])

    def get_probs(self) -> np.ndarray:
        if self.solution_interpolant is None:
            raise RuntimeError("Run evolve() before querying.")
        state = self.get_final_wavefunction()
        return np.abs(state)**2

    def get_result(self) -> np.ndarray:
        """Collapses the N-dimensional wave function and returns the coordinate tuple."""
        if self._cached_left is None or self._cached_right is None:
            raise RuntimeError("Run evolve() before querying the result.")

        probs = self.get_probs()
        if self.result_type == "max":
            max_prob = np.max(probs)
            indices_max = np.where(np.isclose(probs, max_prob))[0]
            flat_idx = np.random.choice(indices_max)

            nd_indices = np.unravel_index(flat_idx, tuple(self._points_per_dim))

            theta = np.fromiter(
                (self._get_1d_grid(
                    self._cached_left,
                    self._cached_right,
                    self._points_per_dim[d]
                )[nd_indices[d]] for d in range(self.n_dims)),
                dtype=float,
                count=self.n_dims
            )
            return theta

        # Expected value
        probs_nd = probs.reshape(tuple(self._points_per_dim))

        theta = np.empty(self.n_dims)

        for d in range(self.n_dims):
            grid_d = self._get_1d_grid(
                self._cached_left,
                self._cached_right,
                self._points_per_dim[d]
            )

            # marginalize over all other axes
            axes = tuple(i for i in range(self.n_dims) if i != d)
            marginal = np.sum(probs_nd, axis=axes)

            theta[d] = np.sum(marginal * grid_d)

        return theta

    def plot_2d_distribution(
        self,
        savepath: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
        cmap: str = "magma"
    ) -> None:
        """Plots the 2D probability distribution using the cached objective function."""
        if self._cached_left is None or self._cached_right is None or self._cached_target_function is None:
            raise RuntimeError("Run evolve() before plotting.")
        if self.n_dims != 2:
            raise ValueError(f"plot_2d_distribution only supports 2D simulations. (n_dims={self.n_dims})")

        l, r = self._cached_left, self._cached_right
        x_grid = self._get_1d_grid(l, r, self._points_per_dim[0])
        y_grid = self._get_1d_grid(l, r, self._points_per_dim[1])
        extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]

        fig, ax = plt.subplots(figsize=(9, 7), dpi=120)

        # Probability density color map
        probs_flat = self.get_probs()
        probs_2d = probs_flat.reshape(tuple(self._points_per_dim))
        im = ax.imshow(
            probs_2d.T,
            extent=extent,
            origin='lower',
            cmap=cmap,
            interpolation='bicubic'
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability Density', fontsize=12, weight='bold')

        # Objective function contour lines
        hires_x = np.linspace(l, r, 200)
        hires_y = np.linspace(l, r, 200)
        X_hires, Y_hires = np.meshgrid(hires_x, hires_y, indexing='ij')
        Z_hires = self._cached_target_function(X_hires, Y_hires)

        contours = ax.contour(X_hires, Y_hires, Z_hires, levels=20, colors='white', alpha=0.4, linewidths=1.0)
        ax.clabel(contours, inline=True, fontsize=9, fmt="%.1f")

        # Plot optimal solution
        opt_coords = self.get_result()
        ax.scatter([opt_coords[0]], [opt_coords[1]], color='#00FFFF', marker='X', s=250,
                   edgecolors='black', linewidths=1.0, zorder=5,
                   label=f'Sampled Minimum: ({opt_coords[0]:.2f}, {opt_coords[1]:.2f})')

        # Figure formatting
        ax.set_xlabel('Dimension 1', fontsize=12, weight='bold')
        ax.set_ylabel('Dimension 2', fontsize=12, weight='bold')
        if title:
            ax.set_title(title, fontsize=14, weight='bold', pad=15)
        ax.legend(loc='upper right', framealpha=0.8)
        fig.tight_layout()
        if savepath is not None:
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath, dpi=300)
            print(f"Saved plot to {savepath}")
        if show:
            plt.show()


# =========================
# EXECUTION BLOCK (2D High-Res Test)
# =========================
if __name__ == "__main__":

    from time import perf_counter

    def multimodal_test_function(x, y):
        centers = [(2, 2), (-2, -2), (-2, 2), (2, -2)]
        depths = [1.5, 1.0, 0.8, 0.7]
        widths = [1.0, 0.8, 0.5, 1.2]

        potential = 0.05 * (x**2 + y**2)

        for (cx, cy), d, w in zip(centers, depths, widths):
            potential -= d * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * w**2))

        return potential

    QUBITS = (5,4) # (10, 8)
    LEFT_BOUND = -4.0
    RIGHT_BOUND = 5.0
    TOTAL_TIME = 15.0
    TIME_STEPS = 200
    RESULT_TYPE = 'max'

    start = perf_counter()

    # sim = QuantumEvolutionSimulator(
    #     qubits=QUBITS,
    #     result_type=RESULT_TYPE,
    #     fft_backend='numpy'
    # )
    # print(f'{"=" * 20}\nEvolving with numpy backend  (State vector size: {sim.dim})...')
    # sim.evolve(
    #     target_function=multimodal_test_function,
    #     left=LEFT_BOUND,
    #     right=RIGHT_BOUND,
    #     total_time=TOTAL_TIME,
    #     time_steps=TIME_STEPS
    # )
    # optimal_solution = sim.get_result()
    # value = multimodal_test_function(*optimal_solution)
    # duration = perf_counter() - start
    # print(
    #     f"Done. Time: {
    #         duration:.2f}s. Optimal solution: ({
    #         optimal_solution[0]:.2f}, {
    #             optimal_solution[1]:.2f}). Value: {
    #                 value:.2f}")

    start = perf_counter()

    sim = QuantumEvolutionSimulator(
        qubits=QUBITS,
        result_type=RESULT_TYPE,
        fft_backend='pyfftw',
        schedule_exponent_kinetic=lambda t, T: -t,
        schedule_exponent_potential=lambda t, T: t
    )
    print(f'{"=" * 20}\nEvolving with pyfftw backend  (State vector size: {sim.dim})...')
    sim.evolve(
        target_function=multimodal_test_function,
        left=LEFT_BOUND,
        right=RIGHT_BOUND,
        total_time=1e-6, # TOTAL_TIME,
        time_steps=TIME_STEPS
    )
    optimal_solution = sim.get_result()
    value = multimodal_test_function(*optimal_solution)
    duration = perf_counter() - start
    print(
        f"Done. Time: {
            duration:.2f}s. Optimal solution: ({
            optimal_solution[0]:.2f}, {
                optimal_solution[1]:.2f}). Value: {
                    value:.2f}")
    print(f"Evolving 2D system (State vector size: {sim.dim})...")
    sim.plot_2d_distribution(savepath=f'../data/output/simulator_test_2d_{RESULT_TYPE}.png')
