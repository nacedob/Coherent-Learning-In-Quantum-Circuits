from icecream import ic
import logging
from typing import Any, Optional, Literal
import json
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, fields
from src.data import mnist_pca_digits, multi_boundaries
from src.models.datareuploading import DataReuploadingQNN, TrainingHistory
from src.models.gates import *
import numpy as np
import time
from .utils import find_root

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# --- Configuration ---


@dataclass
class ExperimentConfig:
    RECOMPUTE: bool = False
    LOAD: bool = True
    DATASET: str = "boundaries"
    SEED: int = 42
    N_TRAIN: int = 100
    N_TEST: int = 500
    PATIENCE: int = 10
    EPOCHS: int = 30
    BATCH_SIZE: Optional[int] = 1
    N_QUBITS: int = 1
    N_LAYERS: int = 2
    SIMULATOR_QUBITS: int = 4
    VERBOSE: bool = True
    TIME_STEPS: int = 100
    TOTAL_TIME: float = 1.0
    SAVE_FORMAT: Literal["pickle", "json"] = "pickle"
    EXPERIMENT_FOLDER: str = str(find_root() / "CoherentQMLPython/data/experiments/final_experiment")

    def __init__(self, **kwargs):
        valid_fields = {f.name for f in fields(self)}

        unknown = set(kwargs) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown)}\n"
                f"Available keys: {sorted(valid_fields)}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        """Returns a dictionary representation for logic/serialization."""
        return asdict(self)

    def save_to_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


def validate_config(config: ExperimentConfig) -> None:
    """Saves config if new, or validates against existing config.json."""
    folder = Path(config.EXPERIMENT_FOLDER)
    config_path = folder / "config.json"

    # Convert instance to dict and remove operational flags
    current_params = {k: v for k, v in asdict(config).items() if k not in ["RECOMPUTE", "LOAD"]}

    if config_path.exists():
        with open(config_path, "r") as f:
            saved_params = json.load(f)
            
        if current_params != saved_params and not config.RECOMPUTE:
            all_keys = sorted(set(current_params) | set(saved_params))
            differences = []

            for key in all_keys:
                current_val = current_params.get(key, "<missing>")
                saved_val = saved_params.get(key, "<missing>")

                if current_val != saved_val:
                    differences.append(
                        f" - {key}: current={current_val!r}, saved={saved_val!r}"
                    )

            diff_text = "\n".join(differences)

            error_msg = (
                f"Configuration mismatch in {folder}. Current settings do not match "
                f"saved settings.\n\n"
                f"Differences:\n{diff_text}\n\n"
                f"Set RECOMPUTE=True to overwrite or use a different "
                f"EXPERIMENT_FOLDER."
            )

            logging.error(error_msg)
            raise RuntimeError(error_msg)

        return None

    # Save the current parameters
    with open(config_path, "w") as f:
        json.dump(current_params, f, indent=4)
        logging.info(f"Configuration verified and saved to {config_path}")

# --- Data Loading ---


def load_dataset(config: ExperimentConfig):
    logging.info(f"Loading dataset: {config.DATASET}...")
    if config.DATASET == "mnist":
        return mnist_pca_digits(n_train=config.N_TRAIN, n_test=config.N_TEST, seed=config.SEED)
    if config.DATASET == "boundaries":
        return multi_boundaries(n_train=config.N_TRAIN, n_test=config.N_TEST, seed=config.SEED, n_boundaries=5,
                                x_min=-np.pi, x_max=np.pi)
    if config.DATASET == "simple_boundaries":
        eps = 0.1
        return multi_boundaries(
            n_train=config.N_TRAIN,
            n_test=config.N_TEST,
            seed=config.SEED,
            x_min=-np.pi,
            x_max=np.pi,
            noise=0.1,
            boundaries=[-np.pi / 3, - eps, + eps, 2 * np.pi / 3],
        )
    if config.DATASET == "medium_boundaries":
        return multi_boundaries(
            n_train=config.N_TRAIN,
            n_test=config.N_TEST,
            seed=config.SEED,
            x_min=-np.pi,
            x_max=np.pi,
            boundaries=[-2 * np.pi / 3, -np.pi / 6, np.pi / 2],
        )
    raise ValueError(f"Unknown dataset: {config.DATASET}")

# --- Training Runners ---


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


def get_model(config: ExperimentConfig) -> DataReuploadingQNN:
    return DataReuploadingQNN(n_qubits=config.N_QUBITS, n_layers=config.N_LAYERS, seed=config.SEED,)


def run_gd_training(config: ExperimentConfig, qnn: DataReuploadingQNN, data: tuple) -> TrainingHistory:
    X_train, y_train, X_test, y_test = data

    start_time = time.perf_counter()
    history = qnn.train(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        optimizer="sgd",
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        patience=config.PATIENCE,
        verbose=config.VERBOSE,
        print_every=10,
        save_hist=True
    )
    elapsed = time.perf_counter() - start_time
    logging.info(f"GD Training completed in {elapsed:.2f} seconds.")
    return history


def run_coherent_training(config: ExperimentConfig, qnn: DataReuploadingQNN, data: tuple) -> TrainingHistory:
    X_train, y_train, X_test, y_test = data

    start_time = time.perf_counter()
    history = qnn.coherent_train(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        epochs=1, # config.EPOCHS,
        simulator_qubits=config.SIMULATOR_QUBITS,
        time_steps=config.TIME_STEPS,
        total_time=config.TOTAL_TIME,
        patience=min(config.PATIENCE, 3),
        verbose=config.VERBOSE,
        print_every=10,
        save_hist=True,
        batch_size=config.BATCH_SIZE
    )
    elapsed = time.perf_counter() - start_time
    logging.info(f"Coherent Training completed in {elapsed:.2f} seconds.")
    return history

# --- Main Pipeline ---


def run_experiment(overrides: Optional[dict[str, Any]] = None):
    config = ExperimentConfig(**(overrides or {}))

    exp_folder = Path(config.EXPERIMENT_FOLDER)
    exp_folder.mkdir(parents=True, exist_ok=True)

    # 1. Validate Config
    validate_config(config)

    coherent_path = exp_folder / ('coherent_training' + ('.json' if config.SAVE_FORMAT == 'json' else '.pickle'))
    gd_path = exp_folder / ('gd_training' + ('.json' if config.SAVE_FORMAT == 'json' else '.pickle'))
    data = None

    # 2. Process Gradient Descent
    if not config.RECOMPUTE and gd_path.exists():
        logging.info(f"Loading existing GD history from {gd_path.name}")
        gd_history = TrainingHistory.load(
            gd_path) if config.SAVE_FORMAT == 'json' else TrainingHistory.load_pickle(gd_path)
    else:
        logging.info("Starting Gradient Descent training phase...")
        data = load_dataset(config) if data is None else data
        qnn = get_model(config)
        gd_history = run_gd_training(config, qnn, data)
        qnn.save(exp_folder / 'gd_qnn.pkl')
        gd_history.save(gd_path) if config.SAVE_FORMAT == 'json' else gd_history.save_pickle(gd_path)
        logging.info(f"Finished GD training phase. Saved history to {gd_path}.")

    # # 3. Process Coherent Training
    if not config.RECOMPUTE and coherent_path.exists():
        logging.info(f"Loading existing Coherent history from {coherent_path.name}")
        coherent_history = TrainingHistory.load(
            coherent_path) if config.SAVE_FORMAT == 'json' else TrainingHistory.load_pickle(coherent_path)
    else:
        logging.info("Starting Coherent training phase...")
        data = load_dataset(config) if data is None else data
        qnn = get_model(config)
        coherent_history = run_coherent_training(config, qnn, data)
        coherent_history.save(
            coherent_path) if config.SAVE_FORMAT == 'json' else coherent_history.save_pickle(coherent_path)
        qnn.save(exp_folder / 'coherent_qnn.pkl')
        logging.info(f"Finished Coherent training phase. Saved history to {coherent_path}.")

    # 4. Finalize
    logging.info("Generating plots...")

    logging.info("Experiment pipeline finished successfully.")

    return


def main(qubit_list: tuple[int, ...] = (1, 2), layer_list: tuple[int, ...] = (1, 2, 3)) -> None:
    for qubits in qubit_list:
        for layers in layer_list:
            if qubits * layers != 2 or qubits > 1:
                continue
            print(f'{"=" * 20} QUBITS: {qubits}, LAYERS: {layers} {"=" * 20}')
            run_experiment(
                {
                    "N_QUBITS": qubits,
                    "N_LAYERS": layers,
                    "EPOCHS": 30,
                    "RECOMPUTE": True,
                    "DATASET": "simple_boundaries",
                    "BATCH_SIZE": 1,
                    "EXPERIMENT_FOLDER": str(find_root() / f"CoherentQMLPython/data/experiments/test/{qubits}_{layers}"),
                    "TOTAL_TIME": 0.01
                }
            )
            print('\n\n')


if __name__ == "__main__":
    main()