import numpy as np
import pytest
from src.models.history import StepRecord 


def test_steprecord_basic_fields():
    step = StepRecord(
        epoch=2,
        batch=1,
        parameters=np.array([1.0, 2.0, 3.0]),
        train_loss=0.42,
        train_acc=0.88,
        test_loss=0.5,
        test_acc=0.85,
    )

    assert step.epoch == 2
    assert step.batch == 1
    assert step.train_loss == 0.42
    assert step.train_acc == 0.88


def test_steprecord_array_roundtrip():
    arr = np.random.randn(5, 3)

    step = StepRecord(
        epoch=1,
        batch=None,
        parameters=arr,
        train_loss=0.1,
        train_acc=0.9,
    )

    d = step.to_dict()
    restored = StepRecord.from_dict(d)

    assert np.allclose(restored.parameters, arr)


def test_steprecord_complex_psi_roundtrip():
    psi = np.array([1+1j, 0.5-0.2j, -1j])

    step = StepRecord(
        epoch=1,
        batch=1,
        parameters=np.array([0.0]),
        train_loss=0.2,
        train_acc=0.7,
        psi_evolution=psi,
    )

    d = step.to_dict()
    restored = StepRecord.from_dict(d)

    assert np.allclose(restored.psi_evolution, psi)
    
def test_format():
    psi = np.array([1+1j, 0.5-0.2j, -1j])
    step = StepRecord(
        epoch=1,
        batch=1,
        parameters=np.array([0.0]),
        train_loss=0.2,
        train_acc=0.7,
        psi_evolution=psi,
    )

    s = step.format()
    assert "Epoch" in s
    assert "Batch" in s
    assert "Train loss" in s
    assert "Train acc" in s
    
def test_print():
    psi = np.array([1+1j, 0.5-0.2j, -1j])
    step = StepRecord(
        epoch=1,
        batch=1,
        parameters=np.array([0.0]),
        train_loss=0.2,
        train_acc=0.7,
        psi_evolution=psi,
    )

    step.print()
