import numpy as np
from typing import Optional, Iterable, Tuple

def multi_boundaries(
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    boundaries: Optional[Iterable[float]] = None,
    n_boundaries: Optional[int] = None,
    n_train: int = 100,
    n_test: int = 50,
    noise: float = 0.0,  # Added noise for realism
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    rng = np.random.default_rng(seed)
    
    # 1. Setup Range
    if x_min is None: x_min = rng.uniform(-10, 0)
    if x_max is None: x_max = rng.uniform(5, 15)

    # 2. Setup Boundaries
    if boundaries is None:
        if n_boundaries is None: 
            n_boundaries = rng.integers(1, 5)
        n_regions = n_boundaries + 1

        # Random region widths that sum to total length
        widths = rng.dirichlet(np.ones(n_regions))

        # Optional: enforce rough balance between classes
        # (even vs odd regions)
        even_sum = widths[::2].sum()
        odd_sum = widths[1::2].sum()
        widths[::2] *= 0.5 / even_sum
        widths[1::2] *= 0.5 / odd_sum

        # Rescale to actual interval
        widths *= (x_max - x_min)

        # Convert widths → boundaries
        boundaries = x_min + np.cumsum(widths[:-1])
    else:
        boundaries = np.sort(np.array(list(boundaries), dtype=float))

    # 3. Generate Data
    n_points = n_train + n_test
    # Using uniform instead of linspace makes the density more natural
    X = rng.uniform(x_min, x_max, n_points) 
    
    # Determine regions (vectorized)
    region_idx = np.sum(X[:, None] > boundaries[None, :], axis=1)
    y = np.where(region_idx % 2 == 0, 1, -1)

    # 4. Add Noise (Optional)
    if noise > 0:
        # Flip labels with probability 'noise'
        flip_mask = rng.random(n_points) < noise
        y[flip_mask] *= -1

    # 5. Correct Shuffling (Keeping X and y paired)
    indices = np.arange(n_points)
    rng.shuffle(indices)
    X, y = X[indices], y[indices]

    return X[:n_train].reshape(-1, 1), y[:n_train], X[n_train:].reshape(-1, 1), y[n_train:]

# --- Visualization Check ---
if __name__ == "__main__":
    from visualization import plot_1d_classification
    # Example: 3 boundaries (creates 4 alternating strips)
    X_train, y_train, X_test, y_test = multi_boundaries(
        boundaries=[-1, 0, 1], 
        n_train=400, 
        noise=0,
        seed=42
    )
    
    plot_1d_classification(X_train, y_train)