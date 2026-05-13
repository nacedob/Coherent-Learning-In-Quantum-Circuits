import numpy as np
import random

# Constants
DEFAULT_N_POINTS = 100
DEFAULT_SPREAD = 1.0

# --- Private Helpers ---

def _generate_random_points(n_points: int, spread: float, dim: int = 3, seed: int = None):
    rng = np.random.default_rng(seed)
    return rng.uniform(-spread, spread, (n_points, dim))


def _get_labels(n_points, spread, seed, condition_fn):
    """
    Internal helper to generate points and apply a labeling condition.
    Returns labels as 1 or -1.
    """
    points = _generate_random_points(n_points, spread, 3, seed=seed)
    # Apply condition and transform {0, 1} -> {-1, 1}
    binary_labels = np.array([condition_fn(p) for p in points], dtype=int)
    labels = 2 * binary_labels - 1
    return points, labels

# --- Geometric Samplers ---

def torus(n_points=DEFAULT_N_POINTS, inner_radius=0.25, outer_radius=0.75,
          spread=DEFAULT_SPREAD, seed=None):
    assert inner_radius < outer_radius
    def is_inside(p):
        dist_from_center = np.sqrt(p[0]**2 + p[1]**2)
        return (dist_from_center - outer_radius)**2 + p[2]**2 < inner_radius**2
    return _get_labels(n_points, spread, seed, is_inside)


def sphere(n_points=DEFAULT_N_POINTS, radius=1.0, center=(0, 0, 0),
           spread=DEFAULT_SPREAD, seed=None):
    c = np.array(center)
    def is_inside(p):
        return np.sum((p - c)**2) < radius**2
    return _get_labels(n_points, spread, seed, is_inside)


def shell(n_points=DEFAULT_N_POINTS, inner_radius=0.5, outer_radius=1.0,
          center=(0, 0, 0), spread=DEFAULT_SPREAD, seed=None):
    c = np.array(center)
    def is_inside(p):
        dist_sq = np.sum((p - c)**2)
        return inner_radius**2 < dist_sq < outer_radius**2
    return _get_labels(n_points, spread, seed, is_inside)


def cube(n_points=DEFAULT_N_POINTS, side_length=1.0, 
         spread=DEFAULT_SPREAD, seed=None):
    half_side = side_length / 2
    def is_inside(p):
        return np.all(np.abs(p) <= half_side)
    return _get_labels(n_points, spread, seed, is_inside)


def multi_spheres(n_points=DEFAULT_N_POINTS, centers=((0, -0.4, 0.2), (0.1, 0.2, 0)),
                  radii=(0.3, 0.5), spread=DEFAULT_SPREAD, seed=None):
    centers_arr = [np.array(c) for c in centers]
    def is_inside(p):
        return any(np.linalg.norm(p - c) < r for c, r in zip(centers_arr, radii))
    return _get_labels(n_points, spread, seed, is_inside)


def cylinder(n_points=DEFAULT_N_POINTS, radius=0.8, spread=DEFAULT_SPREAD, seed=None):
    def is_inside(p):
        return (p[0]**2 + p[1]**2) < radius**2
    return _get_labels(n_points, spread, seed, is_inside)

# --- Complex/Manifold Samplers ---

def helix(n_points, radius=1.0, z_speed=None, ang_speed=4.0, noise=True, seed=None):
    rng = np.random.default_rng(seed)
    z_speed = z_speed or (ang_speed / np.pi)
    half_n = n_points // 2

    theta = np.sqrt(rng.random(half_n)) * 2 / z_speed

    def create_helix(t, phase=0):
        x = np.cos(ang_speed * t + phase) * radius
        y = np.sin(ang_speed * t + phase) * radius
        z = -1 + t * z_speed
        data = np.stack([x, y, z], axis=1)
        if noise:
            data += rng.uniform(-0.05, 0.05, data.shape)
        return data

    data_a = create_helix(theta, phase=0)
    data_b = create_helix(theta, phase=np.pi)

    points = np.concatenate([data_a, data_b], axis=0)
    # Changed labels to 1 and -1
    labels = np.concatenate([np.ones(half_n), -1 * np.ones(half_n)], axis=0)

    idx = np.random.permutation(len(points))
    return points[idx], labels[idx].astype(int)


def sinus3d(n_points, amplitude=1.0, freq=np.pi, offset_phase=0.0, 
            offset_sin=0.0, spread=1.0, seed=None, direction=1):
    
    points = _generate_random_points(n_points, spread, 3, seed=seed)
    x, y, z = points.T

    if direction == 1:
        boundary = -amplitude * np.sin(freq * x + offset_phase) + offset_sin
    elif direction == 2:
        boundary = amplitude * np.sin(freq * y + offset_phase) + offset_sin
    else: 
        boundary = -amplitude * np.cos(freq * z + offset_phase) + offset_sin

    # Boolean logic to {-1, 1}
    labels = np.where(z > boundary, 1, -1)
    return points, labels


def corners3d(n_points=DEFAULT_N_POINTS, spread=DEFAULT_SPREAD, seed=None):
    centers = [[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
    radii = [0.75] * 8
    
    return multi_spheres(
        n_points=n_points, 
        centers=centers, 
        radii=radii, 
        spread=spread, 
        seed=seed
    )