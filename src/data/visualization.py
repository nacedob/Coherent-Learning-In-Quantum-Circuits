import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_dataset(X, y, ax, title=None, elev=20, azim=45):
    """
    Automatically detects dimensions (1D, 2D, or 3D) and plots binary classification data.
    """
    # Ensure X is a 2D array (samples, features)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    dims = X.shape[1]
    unique_classes = np.unique(y)
    colors = ['firebrick', 'dodgerblue']
    palette = dict(zip(unique_classes, colors))

    if dims == 1:
        _plot_1d(X, y, title or "1D Classification", palette, ax)
    elif dims == 2:
        _plot_2d(X, y, title or "2D Classification", palette, ax)
    elif dims == 3:
        _plot_3d(X, y, title or "3D Classification", palette, elev, azim, ax)
    else:
        raise ValueError(f"Unsupported dimensionality: {dims}. Only 1D, 2D, and 3D are supported.")

# --- Internal Dispatchers ---

def _plot_1d(X, y, title, palette, ax):
    plt.figure(figsize=(10, 3))
    # Add jitter to y-axis to see overlapping points
    y_jitter = np.random.normal(0, 0.01, size=len(X))
    sns.scatterplot(x=X[:, 0], y=y_jitter, hue=y, palette=palette, alpha=0.6, s=60, edgecolor='w', ax=ax)
    ax.set_yticks([])
    ax.set_xlabel("Feature 1")
    ax.set_ylim(-1, 1)
    plt.title(title)
    sns.despine(left=True)

def _plot_2d(X, y, title, palette, ax):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=palette, alpha=0.7, s=60, edgecolor='w', ax=ax)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    sns.despine()
    plt.show()

def _plot_3d(X, y, title, palette, elev, azim, ax):
    fig = plt.figure(figsize=(10, 7))
    if not hasattr(ax, 'view_init'):
        raise ValueError("3D plot requires matplotlib 3.2.0 or newer.")
    
    for label, color in palette.items():
        mask = (y == label)
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], 
                   label=f"Class {label}", c=color, alpha=0.6, s=40, edgecolors='w')
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    import sys
    import os
    
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data.synthetic import corners3d
    from src.data import multi_boundaries
    
    # X, y = corners3d(n_points=10000)
    X,y, _, _ = multi_boundaries(n_train=10000, n_boundaries=5)
    plot_dataset(X, y)