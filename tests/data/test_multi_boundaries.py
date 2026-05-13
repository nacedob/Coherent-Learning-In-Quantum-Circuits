from src.data import multi_boundaries
import numpy as np
import pytest


def test_shapes():
    X_train, y_train, X_test, y_test = multi_boundaries(
        n_train=100,
        n_test=50,
        seed=0
    )

    assert X_train.shape == (100, 1)
    assert y_train.shape == (100,)
    assert X_test.shape == (50, 1)
    assert y_test.shape == (50,)


def test_labels_are_pm_one():
    _, y_train, _, y_test = multi_boundaries(seed=0)
    y = np.concatenate([y_train, y_test])

    assert set(np.unique(y)) <= {-1, 1}


def test_reproducibility():
    out1 = multi_boundaries(seed=123)
    out2 = multi_boundaries(seed=123)

    for a, b in zip(out1, out2):
        assert np.array_equal(a, b)


def test_different_seeds_produce_different_data():
    out1 = multi_boundaries(seed=1)
    out2 = multi_boundaries(seed=2)

    # Not strictly guaranteed, but extremely likely
    assert not np.array_equal(out1[0], out2[0])


def test_class_balance():
    X_train, y_train, X_test, y_test = multi_boundaries(
        n_train=2000,
        n_test=2000,
        n_boundaries=5,
        noise=0.0,
        seed=42
    )

    y = np.concatenate([y_train, y_test])

    prop_pos = np.mean(y == 1)
    prop_neg = np.mean(y == -1)

    assert 0.45 <= prop_pos <= 0.55
    assert 0.45 <= prop_neg <= 0.55


def test_noise_effect():
    _, y_train_clean, _, _ = multi_boundaries(noise=0.0, seed=0)
    _, y_train_noisy, _, _ = multi_boundaries(noise=0.5, seed=0)

    # With noise, labels should differ
    assert not np.array_equal(y_train_clean, y_train_noisy)


def test_boundaries_argument():
    boundaries = [-2.0, 0.0, 3.0]

    X_train, y_train, _, _ = multi_boundaries(
        x_min=-5,
        x_max=5,
        boundaries=boundaries,
        n_train=1000,
        n_test=0,
        seed=0
    )

    X = X_train.flatten()

    # Manually recompute regions
    region_idx = np.sum(X[:, None] > np.array(boundaries)[None, :], axis=1)
    y_expected = np.where(region_idx % 2 == 0, 1, -1)

    assert np.array_equal(y_train, y_expected)


def test_no_overlap_train_test():
    X_train, _, X_test, _ = multi_boundaries(seed=0)

    # Very unlikely any overlap if sampling is correct
    assert not np.intersect1d(X_train.flatten(), X_test.flatten()).size > 0


def test_range_respected():
    x_min, x_max = -3, 7
    X_train, _, X_test, _ = multi_boundaries(
        x_min=x_min,
        x_max=x_max,
        seed=0
    )

    X = np.concatenate([X_train, X_test]).flatten()
    assert np.all(X >= x_min)
    assert np.all(X <= x_max)
