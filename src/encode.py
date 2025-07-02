import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import (
    ZZFeatureMap,
    ZFeatureMap,
    PauliFeatureMap,
)

from quantum_utils import multi_rz

try:  # Qiskit <2.0 uses IsingXYGate, >=2.0 renamed it
    from qiskit.circuit.library import IsingXYGate
except Exception:  # pragma: no cover - handle version differences
    try:
        from qiskit.circuit.library import XXPlusYYGate as IsingXYGate
    except Exception:  # pragma: no cover - final fallback
        from qiskit.circuit.library import XYGate as IsingXYGate

def adaptive_feature_map(num_qubits: int,
                         lambda_factors=None,
                         delta_weights=None,
                         gamma: float = 0.5) -> QuantumCircuit:
    """Return the adaptive single-axis feature map circuit."""

    expected_length = 5 * 16
    if lambda_factors is None:
        lambda_factors = [0.3] * 5
    if delta_weights is None:
        delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]

    x = ParameterVector("x", expected_length)
    qc = QuantumCircuit(num_qubits)

    for l in range(5):
        base = 16 * l

        for j in range(num_qubits):
            qc.ry(np.pi * x[base + j], j)

        for j in range(num_qubits):
            idx_a = base + 10 + (j % 6)
            idx_b = base + 10 + ((j + 1) % 6)
            angle = np.pi * (0.5 * x[idx_a] + 0.5 * x[idx_b] + 0.1 * (x[idx_a] - x[idx_b]))
            qc.crx(angle, j, (j + 1) % num_qubits)

        for j in range(num_qubits):
            idx1 = base + 10 + (j % 6)
            idx2 = base + 10 + ((j + 2) % 6)
            idx3 = base + 10 + ((j + 4) % 6)
            avg_triple = (x[idx1] + x[idx2] + x[idx3]) / 3
            qc.cry(np.pi * avg_triple, j, (j + 2) % num_qubits)

        for j in range(num_qubits):
            idx_a = base + 10 + (j % 6)
            idx_b = base + 10 + ((j + 1) % 6)
            pair_avg = 0.5 * (x[idx_a] + x[idx_b])
            angle = np.pi * lambda_factors[l] * pair_avg
            qc.cu(angle, 0.0, 0.0, 0.0, j, (j + 3) % num_qubits)

        for j in range(num_qubits // 2):
            qc.append(IsingXYGate(np.pi * gamma), [j, j + num_qubits // 2])

    global_sum = 0
    for l in range(5):
        global_sum = global_sum + delta_weights[l] * x[16 * l + 10]
    global_angle = np.pi * global_sum
    multi_rz(qc, list(range(num_qubits)), global_angle)
    return qc


def npqc_feature_map(num_qubits: int) -> QuantumCircuit:
    """Simple RX + CZ feature map."""
    x = ParameterVector("x", num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i, p in enumerate(x):
        qc.rx(p, i)
    for i in range(num_qubits - 1):
        qc.cz(i, i + 1)
    return qc


def yzcx_feature_map(num_qubits: int) -> QuantumCircuit:
    """RY/RZ layers followed by CX chain."""
    x = ParameterVector("x", num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i, p in enumerate(x):
        qc.ry(p, i)
        qc.rz(p, i)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


AVAILABLE_MAPS = {
    "zz": ZZFeatureMap,
    "z": ZFeatureMap,
    "pauli": lambda num_qubits: PauliFeatureMap(feature_dimension=num_qubits, paulis=["X", "Z"]),
    "adaptive": adaptive_feature_map,
    "npqc": npqc_feature_map,
    "yzcx": yzcx_feature_map,
}


def get_feature_map(name: str, num_qubits: int):
    """Return a feature map circuit by name."""
    key = name.lower()
    if key not in AVAILABLE_MAPS:
        raise ValueError(f"Unknown feature map: {name}")
    fm = AVAILABLE_MAPS[key]
    if callable(fm):
        return fm(num_qubits)
    return fm(feature_dimension=num_qubits)

