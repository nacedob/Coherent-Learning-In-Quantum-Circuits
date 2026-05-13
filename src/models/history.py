import json
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np
from pathlib import Path

# Palette 2 (deeper + punchy highlight)
COLOR_BATCHED = "#514F91"
COLOR_CLASSICAL = "#AF4AC4"
COLOR_UNBATCHED = "#FF3EF3"


@dataclass
class StepRecord:
    epoch: int
    batch: Optional[int]
    parameters: np.ndarray
    train_loss: float
    train_acc: float
    test_loss: Optional[float] = None
    test_acc: Optional[float] = None
    psi_evolution: Optional[np.ndarray] = None

    @staticmethod
    def _encode_array(arr: np.ndarray):
        return {
            "data": arr.tolist(),
            "dtype": str(arr.dtype),
            "shape": arr.shape
        }

    @staticmethod
    def _decode_array(obj):
        return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])

    @staticmethod
    def _encode_complex(arr: np.ndarray):
        return {
            "real": arr.real.tolist(),
            "imag": arr.imag.tolist()
        }

    @staticmethod
    def _decode_complex(obj):
        return np.array(obj["real"]) + 1j * np.array(obj["imag"])

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "batch": self.batch,
            "parameters": self._encode_array(self.parameters),
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
            "psi_evolution": None if self.psi_evolution is None else self._encode_complex(self.psi_evolution),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            epoch=d["epoch"],
            batch=d["batch"],
            parameters=cls._decode_array(d["parameters"]),
            train_loss=d["train_loss"],
            train_acc=d["train_acc"],
            test_loss=d["test_loss"],
            test_acc=d["test_acc"],
            psi_evolution=None if d["psi_evolution"] is None else cls._decode_complex(d["psi_evolution"]),
        )

    def format(self, print_batch: bool = True) -> str:
        base = f"Epoch {self.epoch:3d}"
        if self.batch is not None and print_batch:
            base += f" | Batch {self.batch:3d}"

        base += (
            f" | Train loss: {self.train_loss:+.3f}"
            f" | Train acc: {self.train_acc:+.3f}"
        )

        if self.test_loss is not None and self.test_acc is not None:
            base += (
                f" | Test loss: {self.test_loss:+.3f}"
                f" | Test acc: {self.test_acc:+.3f}"
            )
        if self.parameters.ndim == 2:
            p_flatten = self.parameters.flatten()
            base += (
                f" | Parameters: {p_flatten[0]:.3f} + {p_flatten[1]:.3f}"
            )
        else:
            base += (
                f" | Param mean: {np.mean(self.parameters)}"
            )

        return base

    def print(self, flush: bool = True, print_batch: bool = True):
        print(self.format(print_batch=print_batch), end='\r' if flush else '\n')
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StepRecord):
            return False

        if self.epoch != other.epoch:
            return False

        if self.batch != other.batch:
            return False

        if not np.array_equal(self.parameters, other.parameters):
            return False

        if self.train_loss != other.train_loss:
            return False

        if self.train_acc != other.train_acc:
            return False

        if self.test_loss != other.test_loss:
            return False

        if self.test_acc != other.test_acc:
            return False

        if self.psi_evolution is None and other.psi_evolution is None:
            return True

        if (self.psi_evolution is None) != (other.psi_evolution is None):
            return False

        return np.array_equal(self.psi_evolution, other.psi_evolution)


@dataclass
class TrainingHistory:
    epochs: int
    batches: Optional[int] = None
    steps: List[StepRecord] = field(default_factory=list)

    STYLES = {
        'batched': {
            'marker': 's',
            'color': 'blue',
        },
        'unbatched': {
            'marker': 'o',
            'color': 'red',
        },
        'classical': {
            'marker': 'x',
            'color': 'green',
        },
    }

    def append(self, record: StepRecord):
        if record.epoch < 0 or record.epoch > self.epochs:
            raise ValueError("Invalid epoch index")

        if self.batches is None:
            if record.batch is not None:
                raise ValueError("Batch provided but history is epoch-level")
        else:
            if record.batch is None and record.epoch != 0:
                raise ValueError("Batch missing in batch-level history")
            if record.epoch != 0 and (record.batch < 1 or record.batch > self.batches):
                raise ValueError("Invalid batch index")

        self.steps.append(record)

    # --------------------------------------------------------
    # Access helpers
    # --------------------------------------------------------

    def get(self, epoch: int, batch: Optional[int] = None) -> StepRecord:
        if epoch == -1:
            epoch = self.epochs
        if batch == -1:
            batch = self.batches
        for s in self.steps:
            if s.epoch == epoch and (s.batch == batch or self.batches is None):
                return s

        raise ValueError(f"Epoch {epoch} and batch {batch} not found in history. Available: {[(s.epoch, s.batch) for s in self.steps]}")

    def get_epoch(self, epoch: int) -> List[StepRecord]:
        return [s for s in self.steps if s.epoch == epoch]

    # --------------------------------------------------------
    # Structured getters (ordered)
    # --------------------------------------------------------

    def train_losses(self):
        return self._ordered(lambda s: s.train_loss)

    def train_accuracies(self):
        return self._ordered(lambda s: s.train_acc)

    def test_losses(self):
        return self._ordered(lambda s: s.test_loss)

    def test_accuracies(self):
        return self._ordered(lambda s: s.test_acc)

    def parameters(self):
        return self._ordered(lambda s: s.parameters)

    def psis(self) -> np.ndarray:
        """
        shape = (epochs, batches, dim, time_steps + 1)
        """
        # Get random psi shape
        step_record = self.get(1, 1)
        if step_record.psi_evolution is None:
            raise RuntimeError("No psi evolution found in history")

        dim, times = step_record.psi_evolution.shape
        time_steps = times - 1

        psis = np.full(((self.epochs, self.batches or 1, dim, time_steps + 1)), np.nan, dtype=np.complex128)

        for e in range(1, self.epochs + 1):
            for b in range(1, self.batches + 1) if self.batches is not None else [1]:
                step_record = self.get(e, b)
                if step_record is not None and step_record.psi_evolution is not None:
                    psis[e - 1, b - 1] = step_record.psi_evolution

        return psis

    def fidelities(
        self,
        optimal_state: np.ndarray
    ) -> np.ndarray:

        if not np.isclose(np.linalg.norm(optimal_state), 1):
            raise ValueError("Optimal state is not normalized")

        # if np.count_nonzero(np.isclose(np.abs(optimal_state), 1)) != 1:
        #     raise ValueError("Optimal state is not a computational basis state")

        psi_evol = self.psis()  # shape = (epochs, batches, dim, time_steps) = (E, B, D, T)

        fidelities = np.abs(
            np.einsum("ebdt,d->ebt", np.conj(psi_evol), optimal_state)
        ) ** 2

        return fidelities

    def _ordered(self, fn, skip_none=False):
        if self.batches is None:
            out = []

            for e in range(self.epochs + 1):
                step_record = self.get(e, None)
                if step_record is not None:
                    val = fn(step_record)
                    if not (skip_none and val is None):
                        out.append(val)

            return np.array(out)

        out = np.full((self.epochs, self.batches), np.nan)

        for e in range(self.epochs + 1):
            batch_iterator = range(1, self.batches + 1) if e != 0 else [None]
            for b in batch_iterator:
                step_record = self.get(e, b)
                if step_record is not None:
                    val = fn(step_record)
                    if not (skip_none and val is None):
                        out[e - 1, (b - 1) if b is not None else 0] = val

        return out
    # --------------------------------------------------------
    # Completion check
    # --------------------------------------------------------

    def is_complete(self) -> bool:
        if self.batches is None:
            # Expect exactly one step per epoch
            return all(self.get(e, None) is not None for e in range(self.epochs + 1))
        else:
            # Expect full grid (epoch, batch)
            return all(
                self.get(e, b) is not None
                for e in range(self.epochs + 1)
                for b in range(1, self.batches + 1) if e != 0
            )

    # --------------------------------------------------------
    # Load and save methods
    # --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if not self.is_complete():
            raise RuntimeError("History is not complete, cannot save")

        path = Path(path).with_suffix(".json")

        data = {
            "epochs": self.epochs,
            "batches": self.batches,
            "steps": [s.to_dict() for s in self.steps],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    
    def save_pickle(self, path: str | Path) -> None:
        if not self.is_complete():
            raise RuntimeError("History is not complete, cannot save")

        path = Path(path).with_suffix(".pickle")

        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str | Path) -> "TrainingHistory":
        path = Path(path).with_suffix(".json")

        with open(path, "r") as f:
            data = json.load(f)

        hist = TrainingHistory(
            epochs=data["epochs"],
            batches=data["batches"],
        )

        hist.steps = [
            StepRecord.from_dict(s) for s in data["steps"]
        ]

        if not hist.is_complete():
            raise TypeError("Loaded object is incomplete")

        return hist
    
    @staticmethod
    def load_pickle(path: str | Path) -> "TrainingHistory":
        path = Path(path).with_suffix(".pickle")

        with open(path, "rb") as f:
            return pickle.load(f)
        
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrainingHistory):
            return False

        if self.epochs != other.epochs:
            return False

        if self.batches != other.batches:
            return False

        if len(self.steps) != len(other.steps):
            return False

        return all(
            s1 == s2
            for s1, s2 in zip(self.steps, other.steps)
        )

    

if __name__ == "__main__":
    def make_history(epochs, batches=None, noise=0.05, seed=0):
        rng = np.random.default_rng(seed)

        hist = TrainingHistory(epochs=epochs, batches=batches)

        for e in range(epochs + 1):
            if batches is None:
                # unbatched
                loss = np.exp(-e / epochs) + noise * rng.normal()
                acc = 1 - loss + noise * rng.normal()

                hist.append(
                    StepRecord(
                        epoch=e,
                        batch=None,
                        parameters=np.array([e]),
                        train_loss=loss,
                        train_acc=acc,
                        test_acc=acc * 0.9,
                        test_loss=loss * 0.9
                    )
                )
            else:
                # batched
                for b in range(1, batches + 1):
                    # simulate smoother convergence across batches
                    progress = e + b / batches
                    loss = np.exp(-progress / epochs) + noise * rng.normal()
                    acc = 1 - loss + noise * rng.normal()

                    hist.append(
                        StepRecord(
                            epoch=e,
                            batch=b,
                            parameters=np.array([e, b]),
                            train_loss=loss,
                            train_acc=acc,
                            test_acc=acc * 0.9,
                            test_loss=loss * 0.9
                        )
                    )

        return hist

    # -----------------------------
    # Create different histories
    # -----------------------------
    h_batched = make_history(epochs=15, batches=10, seed=1)
    h_small_batches = make_history(epochs=15, batches=3, seed=2)
    h_unbatched = make_history(epochs=15, batches=None, seed=3)

    # -----------------------------
    # Compare
    # -----------------------------
    compare_histories(
        data=[
            {'history': h_batched, 'label': 'Batched (10)', 'color': COLOR_BATCHED},
            {'history': h_small_batches, 'label': 'Batched (3)', 'color': COLOR_CLASSICAL},
            # {'history': h_unbatched, 'label': 'Unbatched', 'color': COLOR_UNBATCHED},
        ],
        metric="loss",
        show_batches=True,
        title="Loss comparison across training strategies",
    )
