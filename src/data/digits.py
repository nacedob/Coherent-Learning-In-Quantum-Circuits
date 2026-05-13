import numpy as np
from typing import Optional, Tuple, Iterable
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

CACHE_PATH = Path("data/mnist_digits_cached.npz")


def _load_mnist_cached() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST from disk cache if available.
    Otherwise download once and store locally.
    """

    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH)
        return data["X"], data["y"]

    # First execution → download
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False
    )

    X = X.astype(np.float32) / 255.0
    y = y.astype(int)

    # Save for future runs
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, X=X, y=y)

    return X, y


def mnist_pca_digits(
    n_components: int = 3,
    n_train: int = 1000,
    digits: Iterable[int] = (0, 8),
    n_test: int = 500,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)

    # 1. Load cached dataset
    X, y = _load_mnist_cached()

    # 2. Optional digit filtering
    assert len(digits) == 2, f"Expected 2 digits, got {len(digits)}"
    mask = np.isin(y, list(digits))
    X, y = X[mask], y[mask]

    # 3. PCA
    pca = PCA(n_components=n_components, random_state=seed)
    X = pca.fit_transform(X)

    # 4. Sampling
    n_total = n_train + n_test
    idx = rng.choice(len(X), size=n_total, replace=False)

    X = X[idx]
    y = y[idx]
    
    # 5. Convert output labels to +-1
    labels = np.unique(y)
    y = np.where(y == labels[0], -1, 1)

    return (
        X[:n_train],
        y[:n_train],
        X[n_train:],
        y[n_train:]
    )


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = mnist_pca_digits(
        n_components=3,
        digits=(1, 2),
        n_train=2000,
        n_test=1000,
        seed=42
    )

    print(X_train.shape)
