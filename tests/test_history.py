import numpy as np
import pytest
from src.models.history import StepRecord, TrainingHistory

# ============================================================
# StepRecord
# ============================================================


def test_steprecord_format_basic():
    r = StepRecord(
        epoch=1,
        batch=2,
        parameters=np.array([1.0]),
        train_loss=0.5,
        train_acc=0.8,
    )

    s = r.format()
    assert "Epoch" in s
    assert "Batch" in s
    assert "Train loss" in s
    assert "Train acc" in s


def test_steprecord_optional_test_fields():
    r = StepRecord(
        epoch=1,
        batch=None,
        parameters=np.array([1.0]),
        train_loss=0.5,
        train_acc=0.8,
        test_loss=0.4,
        test_acc=0.9,
    )

    s = r.format()
    assert "Test loss" in s
    assert "Test acc" in s


# ============================================================
# TrainingHistory append validation
# ============================================================

def build_synthetic_history():
    epochs = 3
    batches = 2

    h = TrainingHistory(epochs=epochs, batches=batches)

    train_losses = np.random.random((epochs, batches))
    train_accs = np.random.random((epochs, batches))

    # --------------------------------------------------------
    # insert in SHUFFLED order (important for testing robustness)
    # --------------------------------------------------------
    order = [
        (2, 2),
        (1, 1),
        (3, 1),
        (1, 2),
        (2, 1),
        (3, 2),
    ]

    for e, b in order:
        h.append(
            StepRecord(
                epoch=e,
                batch=b,
                parameters=np.array([0.0]),
                train_loss=train_losses[e - 1, b - 1],
                train_acc=train_accs[e - 1, b - 1],
            )
        )

    return h, train_losses, train_accs


def test_losses_shape_and_values():
    h, losses, _ = build_synthetic_history()

    out = h.train_çlosses()

    # shape check
    assert out.shape == (3, 2)

    # value check (structured, NOT flattened)
    for e in range(3):
        for b in range(2):
            assert np.isclose(out[e, b], losses[e, b])


def test_accuracies_shape_and_values():
    h, _, accs = build_synthetic_history()

    out = h.train_accuracies()

    assert out.shape == (3, 2)

    for e in range(3):
        for b in range(2):
            assert np.isclose(out[e, b], accs[e, b])


def test_history_append_epoch_only():
    h = TrainingHistory(epochs=2, batches=None)

    r = StepRecord(
        epoch=1,
        batch=None,
        parameters=np.array([0.0]),
        train_loss=0.1,
        train_acc=0.9,
    )

    h.append(r)
    assert len(h.steps) == 1


def test_history_append_invalid_epoch():
    h = TrainingHistory(epochs=2, batches=None)

    r = StepRecord(
        epoch=3,
        batch=None,
        parameters=np.array([0.0]),
        train_loss=0.1,
        train_acc=0.9,
    )

    with pytest.raises(ValueError):
        h.append(r)


def test_history_batch_mismatch_error():
    h = TrainingHistory(epochs=2, batches=3)

    r = StepRecord(
        epoch=1,
        batch=None,  # invalid
        parameters=np.array([0.0]),
        train_loss=0.1,
        train_acc=0.9,
    )

    with pytest.raises(ValueError):
        h.append(r)


def test_get_epoch():
    h = TrainingHistory(epochs=2)

    r1 = StepRecord(1, None, np.array([0.0]), 0.1, 0.9)
    r2 = StepRecord(2, None, np.array([0.0]), 0.2, 0.8)

    h.append(r1)
    h.append(r2)

    assert h.get_epoch(1)[0].train_loss == 0.1
    assert h.get_epoch(2)[0].train_loss == 0.2


def test_get_specific_step():
    h = TrainingHistory(epochs=2, batches=2)

    r = StepRecord(1, 2, np.array([0.0]), 0.3, 0.7)
    h.append(r)

    found = h.get(1, 2)
    assert found is not None
    assert found.train_loss == 0.3


def test_get_missing_returns_none():
    h = TrainingHistory(epochs=2)

    assert h.get(1, None) is None


def test_losses_ordering():
    h = TrainingHistory(epochs=2)

    h.append(StepRecord(1, None, np.array([0]), 0.1, 0.9))
    h.append(StepRecord(2, None, np.array([0]), 0.2, 0.8))

    losses = h.train_losses()
    assert np.all(np.isclose(losses, np.array([0.1, 0.2])))


def test_accuracies_ordering():
    h = TrainingHistory(epochs=2)

    h.append(StepRecord(1, None, np.array([0]), 0.9, 0.9))
    h.append(StepRecord(2, None, np.array([0]), 0.8, 0.8))

    acc = h.train_accuracies()
    assert np.all(np.isclose(acc, np.array([0.9, 0.8])))


def test_psis_filters_none():
    h = TrainingHistory(epochs=2)

    h.append(StepRecord(1, None, np.array([0]), 0.1, 0.9, psi=None))
    h.append(StepRecord(2, None, np.array([0]), 0.2, 0.8, psi=np.array([1])))

    psis = h.psis()
    assert len(psis) == 1
