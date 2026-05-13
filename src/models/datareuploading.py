from __future__ import annotations
from icecream import ic
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from src.simulator import QuantumEvolutionSimulator
from typing import Optional, Callable, Literal, Iterator, Tuple
from .encoder import euler_encoder
from .gates import rz, rx, ry, apply_gate, apply_multi_gate, cnot
import pickle
from .history import TrainingHistory, StepRecord

Optimizer = Literal["sgd", "adam", "adagrad"]

# ============================================================
#  Data Reuploading Quantum Neural Network
# ============================================================


class DataReuploadingQNN:
    """
    Data Reuploading Quantum Neural Network (QNN).
    Optimized with batched matrix operations and parallelized gradients.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        n_layers: int = 1,
        parameters: Optional[np.ndarray] = None,
        encoder: Optional[Callable] = None,
        seed: Optional[int] = 42,
        loss_function: Literal['difference', 'direct'] = 'direct',
        parametric_gates: Optional[list[str]] = None
    ) -> None:

        if loss_function not in ['difference', 'direct']:
            raise ValueError("loss_function must be either 'difference' or 'direct'")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shape = (n_qubits, n_layers)
        self.encoder = encoder if encoder is not None else euler_encoder
        self.loss_function = loss_function
        self._p_gates = [eval(g) for g in parametric_gates] if parametric_gates is not None else [ry] * self.n_layers

        if seed is not None:
            np.random.seed(seed)

        if parameters is None:
            self.parameters = np.random.uniform(-np.pi, np.pi, size=self.shape)
        else:
            self.parameters = np.asarray(parameters, dtype=float).reshape(*self.shape)

        self.opt_state = dict()

    def _forward(
        self,
        X: np.ndarray,
        parameters: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return final batched statevector for input batch X."""
        theta = parameters if parameters is not None else self.parameters

        X = np.atleast_2d(X)
        batch_size = X.shape[0]
        dim = 2 ** self.n_qubits

        # Initialize batched statevector (batch_size, 2**n_qubits)
        state = np.zeros((batch_size, dim), dtype=np.complex64)
        state[:, 0] = 1.0  # Set all to |0...0>

        for layer in range(self.n_layers):

            # Data encoding
            state = self.encoder(
                state,
                X,
                layer,
                self.n_qubits
            )

            # Trainable rotations (scalar angle per qubit/layer applied to whole batch)
            for qubit in range(self.n_qubits):
                gate = self._p_gates[layer](theta[qubit, layer])
                state = apply_gate(state, gate, qubit, self.n_qubits)

            # CNOTS
            for q in range(self.n_qubits - 1):
                state = apply_multi_gate(
                    state,
                    gate=cnot(),
                    targets=[q, q + 1],
                    n_qubits=self.n_qubits
                )
        return state

    # --------------------------------------------------------

    def _predict(
        self,
        X: np.ndarray,
        parameters: Optional[np.ndarray] = None
    ) -> np.ndarray:
        state = self._forward(X, parameters)

        probabilities = np.abs(state) ** 2
        midpoint = probabilities.shape[1] // 2

        # Probability of first qubit to be |1>
        prob_1 = probabilities[:, midpoint:].sum(axis=1)

        if self.loss_function == 'difference':
            # # Predicted value is the greatest. Then 1 - 2 * prob_0 gives <0 if it's 0, >0 if it's 1
            return 2 * prob_1 - 1

        return prob_1
    # --------------------------------------------------------

    def _loss_from_predictions(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
    ) -> float:

        if self.loss_function == 'difference':
            return float(np.mean((predictions - y) ** 2))

        # predictions are probabilities of class 1
        costs = 0.5 * (1 - y) * predictions + 0.5 * (1 + y) * (1 - predictions)
        return float(np.mean(costs))

    def loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parameters: Optional[np.ndarray] = None
    ) -> float:
        """
        Implements C_y = 1 - |alpha_y|^2
        Where alpha_y is the amplitude of the correct class.
        """

        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        predictions = self._predict(X, parameters)
        return self._loss_from_predictions(y, predictions)

    # --------------------------------------------------------

    def _gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parameters: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute gradient using parameter-shift rule.
        """
        theta = self.parameters if parameters is None else parameters
        theta = theta.copy()

        P = self.n_qubits * self.n_layers
        theta_flat = theta.flatten()

        shifts = np.zeros((2 * P, P))
        for i in range(P):
            shifts[2 * i, i] = +np.pi / 2
            shifts[2 * i + 1, i] = -np.pi / 2

        theta_shifted = theta_flat + shifts
        theta_shifted = theta_shifted.reshape(2 * P, *self.shape)

        # Broadcast X over all shifted params
        X_rep = np.repeat(X[None, :, :], 2 * P, axis=0)

        losses = np.array([
            self.loss(X_rep[i], y, theta_shifted[i])
            for i in range(2 * P)
        ])

        grads = 0.5 * (losses[::2] - losses[1::2])
        grads = grads.reshape(self.shape)
        return grads

    def _apply_optimizer(self, grads: np.ndarray, lr: float, method: Optimizer):
        """Standardized update rules for different optimizers."""

        if method == 'sgd':
            self.parameters -= lr * grads

        elif method == 'adam':
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            self.opt_state["t"] += 1
            t = self.opt_state["t"]

            # Decay the first and second moment running average
            self.opt_state["m"] = beta1 * self.opt_state["m"] + (1 - beta1) * grads
            self.opt_state["v"] = beta2 * self.opt_state["v"] + (1 - beta2) * (grads**2)

            # Bias correction
            m_hat = self.opt_state["m"] / (1 - beta1**t)
            v_hat = self.opt_state["v"] / (1 - beta2**t)

            self.parameters -= lr * m_hat / (np.sqrt(v_hat) + eps)

        elif method == 'adagrad':
            eps = 1e-8
            self.opt_state["v"] += grads**2
            self.parameters -= lr * grads / (np.sqrt(self.opt_state["v"]) + eps)

        else:
            raise ValueError(f"Unknown optimizer: {method}")

    # --------------------------------------------------------
    def _accuracy_from_predictions(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
    ) -> float:

        if self.loss_function == 'difference':
            preds = -np.sign(predictions)
        else:
            preds = np.where(predictions > 0.5, 1, -1)

        return float(np.mean(preds == y))

    def accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parameters: Optional[np.ndarray] = None
    ) -> float:
        """Return classification accuracy over the set."""
        # Obtain probabilty of being |1>
        probs = self._predict(X, parameters)

        return self._accuracy_from_predictions(y, probs)

    # --------------------------------------------------------
    def _batch_generator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield mini-batches from X and y."""
        n_samples = len(X)
        if shuffle and batch_size < n_samples:
            perm = np.random.permutation(n_samples)
            X, y = X[perm], y[perm]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            yield X[start:end], y[start:end]

    # --------------------------------------------------------

    def _get_stats(
        self,
        parameters: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> tuple[float, float, Optional[float], Optional[float]]:
        predictions = self._predict(X_train, parameters)
        train_loss = self._loss_from_predictions(y_train, predictions)
        train_acc = self._accuracy_from_predictions(y_train, predictions)

        test_loss = test_acc = None
        if X_test is not None and y_test is not None:
            predictions = self._predict(X_test, parameters)
            test_loss = self._loss_from_predictions(y_test, predictions)
            test_acc = self._accuracy_from_predictions(y_test, predictions)

        return train_loss, train_acc, test_loss, test_acc

    def _print_stats(
        self,
        step_record: StepRecord,
        print_batch: bool = True,
        flush: bool = True,
    ) -> None:
        step_record.print(flush=flush, print_batch=print_batch)
    # --------------------------------------------------------

    def _early_stopping_step(
        self,
        current_loss: float,
        best_loss: float,
        best_parameters: np.ndarray,
        patience_counter: int,
        patience: Optional[int],
        tolerance: float,
        parameters: np.ndarray,
    ) -> tuple[float, np.ndarray, int, bool]:
        """
        Returns updated (best_loss, best_parameters, patience_counter, should_stop)
        """
        if patience is None:
            return best_loss, best_parameters, patience_counter, False

        if current_loss < best_loss * (1 - tolerance):
            return current_loss, parameters.copy(), 0, False

        patience_counter += 1
        should_stop = patience_counter >= patience
        return best_loss, best_parameters, patience_counter, should_stop

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 200,
        learning_rate: float = 0.1,
        optimizer: Optimizer = 'sgd',
        batch_size: Optional[int] = None,
        verbose: bool = True,
        print_every: int = 10,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        patience: Optional[int] = None,
        patience_tolerance: float = 1e-6,
        save_hist: bool = True,
    ) -> TrainingHistory:
        """Train model using gradient descent or stochastic/mini-batch GD."""

        if not set(np.unique(y_train)).issubset({-1, 1}):
            raise ValueError("y must contain only -1 and 1")

        n_samples = len(X_train)

        if batch_size is None or batch_size >= n_samples:
            batch_size = n_samples

        self.opt_state = {"m": np.zeros(self.shape), "v": np.zeros(self.shape), "t": 0}

        n_batches = int(np.ceil(len(X_train) / batch_size))
        history = TrainingHistory(epochs, batches=n_batches) if save_hist else None

        if history:
            train_loss, train_acc, test_loss, test_acc = self._get_stats(
                self.parameters, X_train, y_train, X_test, y_test
            )
            step_record = StepRecord(
                    epoch=0,
                    batch=None,
                    parameters=self.parameters.copy(),
                    train_loss=train_loss,
                    train_acc=train_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
            )
            history.append(step_record)
            if verbose:
                self._print_stats(step_record, print_batch=False, flush=False)
        best_loss = float("inf")
        best_parameters = self.parameters.copy()
        patience_counter = 0

        for epoch in range(epochs):
            for i, (X_batch, y_batch) in enumerate(self._batch_generator(X_train, y_train, batch_size, shuffle=True)):
                grads = self._gradient(X_batch, y_batch)
                self._apply_optimizer(grads, learning_rate, method=optimizer)

                if save_hist or (verbose and (epoch % print_every == 0 or epoch == epochs - 1)):
                    train_loss, train_acc, test_loss, test_acc = self._get_stats(
                        self.parameters, X_train, y_train, X_test, y_test
                    )
                    step_record = StepRecord(
                            epoch=epoch + 1,
                            batch=i + 1,
                            parameters=self.parameters.copy(),
                            train_loss=train_loss,
                            train_acc=train_acc,
                            test_loss=test_loss,
                            test_acc=test_acc,
                    )
                    if history is not None:
                        history.append(step_record)
                        
                    if batch_size == 1: 
                        self._print_stats(step_record, print_batch=True, flush=False)

            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                self._print_stats(step_record, print_batch=False, flush=False)

            # --- early stopping ---
            best_loss, best_parameters, patience_counter, stop = self._early_stopping_step(
                current_loss=train_loss, best_loss=best_loss, best_parameters=best_parameters,
                patience_counter=patience_counter, patience=patience, tolerance=patience_tolerance,
                parameters=self.parameters,
            )

            if stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                if history is not None:
                    history.epochs = epoch + 1
                    last_step = history.get(-1, -1)
                    last_step.train_loss = train_loss
                    last_step.train_acc = train_acc
                    last_step.test_loss = test_loss
                    last_step.test_acc = test_acc
                    last_step.parameters = self.parameters

                break

        self.parameters = best_parameters

        return history or TrainingHistory(epochs)

    # --------------------------------------------------------
    def _create_qhd_loss_function(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = -1
    ) -> Callable[..., np.ndarray]:
        """
        Returns a callable f(*grids) that evaluates the QNN loss across the N-dimensional meshgrids generated
        by the QHD simulator in parallel.
        """

        def landscape(*grids: np.ndarray) -> np.ndarray:
            grid_shape = grids[0].shape
            total_dims = len(grids)

            if total_dims != self.n_qubits * self.n_layers:
                raise ValueError(
                    f"Simulator grid dimensions ({total_dims}) do not match "
                    f"QNN parameters ({self.n_qubits * self.n_layers})."
                )

            flat_candidates = np.stack(grids).reshape(total_dims, -1).T
            n_candidates = flat_candidates.shape[0]

            def evaluate_candidate(i):
                params_2d = flat_candidates[i].reshape(self.n_qubits, self.n_layers)
                return self.loss(X, y, parameters=params_2d)

            # Evaluate the loss for each configuration in parallel
            losses = Parallel(n_jobs=n_jobs, batch_size='auto')(
                delayed(evaluate_candidate)(i) for i in range(n_candidates)
            )

            return np.array(losses).reshape(grid_shape)

        return landscape

    # -------------------------------------------------------
    def _coherent_batch_training(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        state: np.ndarray,
        sim: QuantumEvolutionSimulator,
        time_steps: int,
        total_time: float,
        left: float = -np.pi,
        right: float = np.pi,
    ) -> tuple[np.ndarray, np.ndarray]:
        function = self._create_qhd_loss_function(X_batch, y_batch, n_jobs=1)
        qa_solution = sim.evolve(
            initial_state=state,
            target_function=function,
            time_steps=time_steps,
            total_time=total_time,
            left=left,
            right=right,
        )
        parameters = qa_solution.reshape(self.n_qubits, self.n_layers)
        new_state = sim._state_history[:, -1].copy()

        return new_state, parameters

    def coherent_train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        simulator_qubits: int,
        time_steps: int,
        total_time: float,
        left: float = -np.pi,
        right: float = np.pi,
        init_state: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        epochs: int = 1,
        verbose: bool = True,
        print_every: int = 1,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        save_hist: bool = False,
        schedule_exponent_kinetic: Optional[Callable[[float, float], float]] = None,
        schedule_exponent_potential: Optional[Callable[[float, float], float]] = None,
        patience: Optional[int] = None,
        patience_tolerance: float = 0.0,
    ) -> TrainingHistory:

        sim = QuantumEvolutionSimulator(
            qubits=[simulator_qubits] * (self.n_layers * self.n_qubits),
            schedule_exponent_kinetic=schedule_exponent_kinetic,
            schedule_exponent_potential=schedule_exponent_potential
        )

        if batch_size is None:
            # if epochs != 1:
            #     raise ValueError("Batch size must be specified for multiple epochs.")
            batch_size = len(X_train)

        n_batches = int(np.ceil(len(X_train) / batch_size))
        history = TrainingHistory(epochs, batches=n_batches) if save_hist else None

        # Get init state
        psi = (
            init_state if init_state is not None else
            np.ones(2**(sim.n_qubits), dtype=float) / np.sqrt(2**(sim.n_qubits))
        )

        if history:
            train_loss, train_acc, test_loss, test_acc = self._get_stats(
                self.parameters, X_train, y_train, X_test, y_test
            )
            step_record = StepRecord(
                    epoch=0,
                    batch=None,
                    parameters=self.parameters.copy(),
                    train_loss=train_loss,
                    train_acc=train_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
                    psi_evolution=psi
            )
            history.append(step_record)
            if verbose:
                self._print_stats(step_record, print_batch=False, flush=False)

        # Training
        if verbose:
            print(f"Training on {simulator_qubits * self.n_layers * self.n_qubits} qubits...")

        # Stopping criteria
        best_loss = float("inf")
        best_parameters = self.parameters.copy()
        patience_counter = 0

        for epoch in range(epochs):
            for i, (X_batch, y_batch) in enumerate(self._batch_generator(X_train, y_train, batch_size)):
                psi, parameters = self._coherent_batch_training(
                    X_batch=X_batch,
                    y_batch=y_batch,
                    state=psi,
                    sim=sim,
                    time_steps=time_steps,
                    total_time=total_time,
                    left=left,
                    right=right,
                )

                if save_hist or (verbose and i % print_every == 0 or epoch == epochs - 1 and epochs > 1) or patience:
                    train_loss, train_acc, test_loss, test_acc = self._get_stats(
                        parameters=parameters, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
                    )
                    step_record = StepRecord(
                            epoch=epoch + 1,
                            batch=i + 1,
                            parameters=parameters,
                            psi_evolution=sim._state_history,
                            train_loss=train_loss,
                            train_acc=train_acc,
                            test_loss=test_loss,
                            test_acc=test_acc
                    )

                    if i % print_every == 0 or i == n_batches - 1:
                        self._print_stats(step_record=step_record, print_batch=True, flush=False)

                    if history:
                        history.append(step_record)

            # --- early stopping ---
            best_loss, best_parameters, patience_counter, stop = self._early_stopping_step(
                current_loss=train_loss, best_loss=best_loss, best_parameters=best_parameters,
                patience_counter=patience_counter, patience=patience, tolerance=patience_tolerance,
                parameters=self.parameters,
            )

            if stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                parameters = best_parameters
                if history is not None:
                    history.epochs = epoch + 1
                    last_step = history.get(-1, -1)
                    last_step.train_loss = train_loss
                    last_step.train_acc = train_acc
                    last_step.test_loss = test_loss
                    last_step.test_acc = test_acc
                    last_step.parameters = self.parameters
                break

        self.parameters = parameters

        return history or TrainingHistory(epochs)

    # --------------------------------------------------------

    def brute_force_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        left: float,
        right: float,
        grid_points: int,
    ) -> np.ndarray:
        """
        Exhaustive grid search over parameter space.
        """
        grid = np.linspace(left, right, grid_points)
        best_loss = np.inf
        best_params = np.empty(self.n_qubits * self.n_layers)
        total_params = self.n_qubits * self.n_layers

        for index_tuple in np.ndindex(*(grid_points,) * total_params):

            candidate = np.array(
                [grid[i] for i in index_tuple]
            ).reshape(self.n_qubits, self.n_layers)

            current_loss = self.loss(X_train, y_train, candidate)

            if current_loss < best_loss:
                best_loss = current_loss
                best_params = candidate.copy()

        return best_params

    def get_optimal_state(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        grid_size: int,
        simulator_qubits: int,
        left: float,
        right: float,
        deviation: Optional[float] = None  # Gaussian sigma of the packet
    ) -> np.ndarray:
        """
        Constructs a multivariate Gaussian wave packet centered at the optimal
        parameters found via brute force search.
        """
        # 1. Get optimal parameters (the center of the Gaussian)
        best_params = self.brute_force_search(X_train, y_train, left, right, grid_size)

        # 2. Setup the simulator and the 1D coordinate grid
        component_per_param = int(2**simulator_qubits)
        sim = QuantumEvolutionSimulator(
            qubits=[simulator_qubits] * (self.n_layers * self.n_qubits),
        )

        grid = sim._get_1d_grid(left, right, component_per_param)

        # 3. Build 1D Gaussian components for each parameter
        if deviation is None:
            dx = grid[1] - grid[0]
            deviation = 3 * dx

        gaussian_components = []
        for _, mu in np.ndenumerate(best_params):
            # Compute the Gaussian: exp(-(x - mu)^2 / (2 * sigma^2))
            phi_1d = np.exp(-0.5 * ((grid - mu) / deviation)**2)

            # Normalize the 1D component (L2 norm for a valid quantum state)
            phi_1d /= np.linalg.norm(phi_1d)
            gaussian_components.append(phi_1d)

        # 4. Combine components via Kronecker Product
        # The final state |Psi> = |psi_1> ⊗ |psi_2> ⊗ ... ⊗ |psi_N>
        state = gaussian_components[0]
        for i in range(1, len(gaussian_components)):
            state = np.kron(state, gaussian_components[i])

        return state

    def save(self, path: str | Path) -> None:
        path = Path(path).with_suffix(".pickle")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "DataReuploadingQNN":
        path = Path(path).with_suffix(".pickle")
        with open(path, "rb") as f:
            return pickle.load(f)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DataReuploadingQNN):
            return False
        if self.n_qubits != value.n_qubits:
            return False
        if self.n_layers != value.n_layers:
            return False
        if not np.all(self.parameters == value.parameters):
            return False
        return True


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":

    # --- Dataset for Testing the Script ---
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from src.data import mnist_pca_digits, multi_boundaries
    from src.data.synthetic import corners3d

    qubits = 2
    layers = 3
    seed = 40

    X_train, y_train, X_test, y_test = multi_boundaries(
        x_min=-np.pi,
        x_max=np.pi,
        boundaries=[-np.pi / 3, 2 * np.pi / 3],
        n_train=1000,
        n_test=300,
        seed=seed
    )

    X_train, y_train = corners3d(n_points=300, seed=seed)
    X_test, y_test = corners3d(n_points=100, seed=seed)

    # X_train, y_train, X_test, y_test =  mnist_pca_digits(n_train=1000, n_test=300, seed=seed)

    def encoder(
        state: np.ndarray,
        X: np.ndarray,
        layer: int,
        n_qubits: int
    ) -> np.ndarray:
        # Ensure X is treated as a 2D batch (batch_size, n_features)
        X = np.atleast_2d(X)
        n_features = X.shape[1]
        for q in range(n_qubits):
            angle = X.sum(axis=1) / n_features
            state = apply_gate(state=state, gate=rx(angle), qubit=q, n_qubits=n_qubits)

        return state

    qnn = DataReuploadingQNN(n_qubits=qubits, n_layers=layers, seed=seed)
    original_parameters = qnn.parameters.copy()

    # Train with Gradient Descent
    qnn.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=1000,
        optimizer='adam',
        learning_rate=0.1,
        batch_size=24,
        print_every=10,
        verbose=True,
        patience=100
    )

    acc_train = qnn.accuracy(X_train, y_train)
    acc_test = qnn.accuracy(X_test, y_test)

    print(f"{'=' * 20} GRADIENT DESCENT RESULTS: {'=' * 20}")
    print(f"Final parameters:\n{qnn.parameters}")
    print(f"Train loss={qnn.loss(X_train, y_train):.4f}, Acc={acc_train:.4f}")
    print(f"Test  loss={qnn.loss(X_test, y_test):.4f}, Acc={acc_test:.4f}")
    print("=" * 40)

    # Reset
    qnn.parameters = original_parameters

    # Train with Coherent Descent
    qnn.coherent_train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        simulator_qubits=3,
        time_steps=100,
        total_time=0.1,
        batch_size=1,
        epochs=10,
        print_every=1,
        patience=3,
        verbose=True
    )

    acc_train = qnn.accuracy(X_train, y_train)
    acc_test = qnn.accuracy(X_test, y_test)

    print(f"{'=' * 20} GRADIENT DESCENT RESULTS: {'=' * 20}")
    print(f"Final parameters:\n{qnn.parameters}")
    print(f"Train loss={qnn.loss(X_train, y_train):.4f}, Acc={acc_train:.4f}")
    print(f"Test  loss={qnn.loss(X_test, y_test):.4f}, Acc={acc_test:.4f}")
    print("=" * 40)
