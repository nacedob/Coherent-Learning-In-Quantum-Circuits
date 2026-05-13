from .gates import rz, ry, apply_gate
import numpy as np


def euler_encoder(
    state: np.ndarray,
    X: np.ndarray,
    layer: int,
    n_qubits: int
) -> np.ndarray:
    # Ensure X is treated as a 2D batch (batch_size, n_features)
    X = np.atleast_2d(X)
    n_features = X.shape[1]

    for q in range(n_qubits):
        for i in range(n_features):
            feature = X[:, i]
            if i % 3 in [0, 2]:
                state = apply_gate(state=state, gate=rz(feature), qubit=q, n_qubits=n_qubits)
            else:
                state = apply_gate(state=state, gate=ry(feature), qubit=q, n_qubits=n_qubits)
    return state
