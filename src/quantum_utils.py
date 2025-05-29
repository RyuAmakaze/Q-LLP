import numpy as np
import torch

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except Exception:  # pragma: no cover - qiskit may not be installed
    QuantumCircuit = None
    Statevector = None


def data_to_circuit(angles, params=None):
    """Return a QuantumCircuit encoding ``angles`` via Y rotations.

    Parameters
    ----------
    angles : Sequence[float] or torch.Tensor
        Rotation angles for RY gates on each qubit.
    params : Sequence[float] or torch.Tensor, optional
        Additional rotation angles to add to the data encoding.

    Notes
    -----
    If qiskit is not installed, this function raises ``ImportError``.
    """
    if QuantumCircuit is None:
        raise ImportError("qiskit is required for circuit construction")

    angles = np.array(angles, dtype=float)
    n_qubits = angles.shape[0]
    if params is not None:
        angles = angles + np.array(params, dtype=float)
    qc = QuantumCircuit(n_qubits)
    for i, theta in enumerate(angles):
        qc.ry(float(theta), i)
    return qc


def circuit_state_probs(circuit):
    """Simulate ``circuit`` and return measurement probabilities."""
    if Statevector is None:
        raise ImportError("qiskit is required for circuit simulation")

    state = Statevector.from_instruction(circuit)
    probs = state.probabilities()
    return torch.tensor(probs, dtype=torch.float32)
