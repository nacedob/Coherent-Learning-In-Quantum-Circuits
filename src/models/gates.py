import numpy as np

# ============================================================
#  Single-qubit rotation matrices
# ============================================================


def rx(theta: float | np.ndarray) -> np.ndarray:
    """Vectorized Rx gate. Supports scalar or 1D array of angles."""
    theta = np.asarray(theta)
    half_theta = theta / 2.0
    c = np.cos(half_theta)
    s = -1j * np.sin(half_theta)

    if theta.ndim == 0:
        return np.array([[c, s], [s, c]])

    # Batched case: shape (batch_size, 2, 2)
    out = np.zeros((theta.shape[0], 2, 2), dtype=complex)
    out[:, 0, 0] = c
    out[:, 0, 1] = s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    return out


def ry(theta: float | np.ndarray) -> np.ndarray:
    """Vectorized Ry gate. Supports scalar or 1D array of angles."""
    theta = np.asarray(theta)
    half_theta = theta / 2.0
    c = np.cos(half_theta)
    s = np.sin(half_theta)

    if theta.ndim == 0:
        return np.array([[c, -s], [s, c]])

    # Batched case: shape (batch_size, 2, 2)
    out = np.zeros((theta.shape[0], 2, 2), dtype=complex)
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    return out


def rz(theta: float | np.ndarray) -> np.ndarray:
    """Vectorized Rz gate. Supports scalar or 1D array of angles."""
    theta = np.asarray(theta)
    half_theta = theta / 2.0

    # Rz diagonal elements
    phase_minus = np.exp(-1j * half_theta)
    phase_plus = np.exp(1j * half_theta)

    if theta.ndim == 0:
        return np.array([
            [phase_minus, 0.0],
            [0.0, phase_plus]
        ], dtype=complex)

    # Batched case: shape (batch_size, 2, 2)
    out = np.zeros((theta.shape[0], 2, 2), dtype=complex)
    out[:, 0, 0] = phase_minus
    out[:, 0, 1] = 0.0
    out[:, 1, 0] = 0.0
    out[:, 1, 1] = phase_plus
    return out

_cache_reshape = dict()
def apply_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
    n_qubits: int
) -> np.ndarray:
    """
    Applies a 1-qubit gate to a batched statevector.
    state shape: (batch_size, 2**n_qubits)
    gate shape: (2, 2) OR (batch_size, 2, 2)
    """
    batch_size = state.shape[0]

    # Reshape to isolate the target qubit.
    # Assumes standard MSB ordering (Qubit 0 is the most significant bit)
    left_dim = 2 ** qubit
    right_dim = 2 ** (n_qubits - qubit - 1)
    
    # ---- cached reshape dimensions ----
    if _cache_reshape.get((n_qubits, qubit)) is None:
        left_dim = 2 ** qubit
        right_dim = 2 ** (n_qubits - qubit - 1)
    if _cache_reshape.get((qubit)) is None:
        _cache_reshape[(n_qubits, qubit)] = (left_dim, right_dim)
    else:
        left_dim, right_dim = _cache_reshape[qubit]

    
    reshaped_state = state.reshape((batch_size, left_dim, 2, right_dim))

    if gate.ndim == 2:
        # Same gate applied to the whole batch
        new_state = np.einsum('ij, bkjl -> bkil', gate, reshaped_state)
    else:
        # Different gate for each item in the batch
        new_state = np.einsum('bij, bkjl -> bkil', gate, reshaped_state)

    return new_state.reshape((batch_size, -1))


# ============================================================
#  Efficient multi-qubit gate application
# ============================================================
def cnot() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
    )

_cache_multiqubits = dict()
def apply_multi_gate(
    state: np.ndarray,      # (batch, 2**n)
    gate: np.ndarray,       # (2**k, 2**k)
    targets: list[int],
    n_qubits: int
) -> np.ndarray:
    
    batch_size = state.shape[0]
    k = len(targets)

    # Reshape into tensor form
    tensor = state.reshape(batch_size, *([2] * n_qubits))

    targets = tuple(targets)

    if _cache_multiqubits is not None and targets in _cache_multiqubits:
        axes, inv_axes = _cache_multiqubits[targets]
    else:
        axes = targets + tuple(i for i in range(n_qubits) if i not in targets)
        inv_axes = np.argsort(axes)
        if _cache_multiqubits is not None:
            _cache_multiqubits[targets] = (axes, inv_axes)

    # Move target qubits to front
    tensor = np.moveaxis(tensor, targets, range(k))

    # Merge target subsystem
    tensor = tensor.reshape(batch_size, 2**k, -1)

    # Apply gate
    tensor = np.einsum("ij,bjk->bik", gate, tensor, optimize=True)

    # Restore shape
    tensor = tensor.reshape(batch_size, *([2] * n_qubits))

    # Restore ordering
    tensor = np.moveaxis(tensor, range(k), targets)

    return tensor.reshape(batch_size, -1)