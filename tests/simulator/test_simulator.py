import time
import numpy as np
import pytest
from src.simulator import QuantumEvolutionSimulator


def multimodal_test_function(x, y):
    return 0.1 * (x**2 + y**2) - np.exp(-(x**2 + y**2))


@pytest.fixture
def sim():
    return QuantumEvolutionSimulator(qubits=(4, 4), fft_backend="numpy")


# -------------------------
# BASIC PHYSICAL CONSISTENCY
# -------------------------
def test_wavefunction_physics_consistency(sim):
    psi0 = np.ones(sim.dim, dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)

    sim.evolve(
        target_function=multimodal_test_function,
        left=-2,
        right=2,
        total_time=1.0,
        time_steps=10,
        initial_state=psi0,
    )

    final = sim.get_final_wavefunction()

    # 1. Norm conservation (unitarity check)
    norm = np.linalg.norm(final)
    assert np.isclose(norm, 1.0, atol=1e-4)

    # 2. Probability normalization
    probs = np.abs(final) ** 2
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)

    # 3. Shape consistency
    assert final.shape == (sim.dim, )


# -------------------------
# BACKEND CONSISTENCY
# -------------------------
def run_sim(backend):
    sim = QuantumEvolutionSimulator(
        qubits=(4, 4),
        fft_backend=backend,
        result_type="max",
    )

    sim.evolve(
        target_function=multimodal_test_function,
        left=-2,
        right=2,
        total_time=2.0,
        time_steps=20,
    )

    return sim.get_result()


def test_numpy_vs_pyfftw_consistency():
    np.random.seed(0)

    res_numpy = run_sim("numpy")
    res_fftw = run_sim("pyfftw")

    # stochastic collapse → allow tolerance
    assert np.allclose(res_numpy, res_fftw, atol=1e-1)


def test_pyfftw_faster_than_numpy():
    def run(backend):
        sim = QuantumEvolutionSimulator(
            qubits=(8, 8),
            fft_backend=backend,
            result_type="max",
        )

        t0 = time.perf_counter()
        sim.evolve(
            target_function=multimodal_test_function,
            left=-2,
            right=2,
            total_time=3.0,
            time_steps=30,
        )
        return time.perf_counter() - t0

    numpy_time = run("numpy")
    fftw_time = run("pyfftw")

    # relaxed threshold (hardware variability)
    assert fftw_time < numpy_time * 0.99


# -------------------------
# POTENTIAL FUNCTION CHECKS (IMPORTANT ADDITION)
# -------------------------
def test_potential_function_validity():
    sim = QuantumEvolutionSimulator(qubits=(3, 3))

    grids = sim._get_nd_grids(-2, 2)
    V = multimodal_test_function(*grids)

    # must be finite
    assert np.all(np.isfinite(V))

    # must be real-valued (no accidental complex drift)
    assert np.isrealobj(V)

    # sanity: should have multiple minima structure
    assert np.std(V) > 0
