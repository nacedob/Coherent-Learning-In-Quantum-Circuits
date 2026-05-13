import numpy as np
import pytest
from pathlib import Path
from src.models.history import StepRecord, TrainingHistory


# --------------------------------------------------------
# Helper: small deterministic history
# --------------------------------------------------------

def make_history():
    hist = TrainingHistory(epochs=3, batches=2)

    # epoch 0 init
    hist.steps.append(
        StepRecord(
            epoch=0,
            batch=None,
            parameters=np.array([0.0]),
            train_loss=1.0,
            train_acc=0.0,
            psi_evolution=np.random.rand(4, 4)
        )
    )

    # full grid
    for e in range(1, 4):
        for b in range(1, 3):
            hist.steps.append(
                StepRecord(
                    epoch=e,
                    batch=b,
                    parameters=np.array([e, b]),
                    train_loss=0.1 * e * b,
                    train_acc=0.2 * e * b,
                    test_loss=0.15 * e * b,
                    test_acc=0.25 * e * b,
                    psi_evolution=np.random.rand(4, 4)
                )
            )

    return hist


# --------------------------------------------------------
# Core correctness
# --------------------------------------------------------

def test_history_complete():
    hist = make_history()
    assert hist.is_complete()


def test_history_data_integrity():
    hist = make_history()

    # check retrieval consistency
    s = hist.get(2, 1)

    assert s.epoch == 2
    assert s.batch == 1
    assert np.isclose(s.train_loss, 0.2)
    assert np.isclose(s.train_acc, 0.4)


# --------------------------------------------------------
# Save / Load roundtrip (IMPORTANT TEST)
# --------------------------------------------------------

def test_history_save_load_roundtrip(tmp_path: Path):
    hist = make_history()

    file_path = tmp_path / "history.json"
    hist.save(file_path)

    assert file_path.exists()

    loaded = TrainingHistory.load(file_path)

    # structure
    assert loaded.epochs == hist.epochs
    assert loaded.batches == hist.batches
    assert len(loaded.steps) == len(hist.steps)

    # full deep comparison
    for a, b in zip(hist.steps, loaded.steps):
        assert a.epoch == b.epoch
        assert a.batch == b.batch
        assert np.allclose(a.parameters, b.parameters)
        assert np.isclose(a.train_loss, b.train_loss)
        assert np.isclose(a.train_acc, b.train_acc)

        if a.test_loss is not None:
            assert np.isclose(a.test_loss, b.test_loss)

        if a.psi_evolution is not None:
            assert np.allclose(a.psi_evolution, b.psi_evolution)
            
    # Check with equal method
    assert hist.__eq__(loaded)
