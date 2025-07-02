import numpy as np
import config
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import (
    ZZFeatureMap,
    ZFeatureMap,
    PauliFeatureMap,
)

from quantum_utils import multi_rz, amplitude_encoding

try:  # Qiskit <2.0 uses IsingXYGate, >=2.0 renamed it
    from qiskit.circuit.library import IsingXYGate
except Exception:  # pragma: no cover - handle version differences
    try:
        from qiskit.circuit.library import XXPlusYYGate as IsingXYGate
    except Exception:  # pragma: no cover - final fallback
        from qiskit.circuit.library import XYGate as IsingXYGate

def adaptive_feature_map(
    num_qubits: int,
    *,
    features_per_layer: int | None = None,
    lambdas=None,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> QuantumCircuit:
    """Return the adaptive single-axis feature map circuit.

    This implementation mirrors ``quantum_utils.adaptive_entangling_circuit``
    but uses symbolic parameters for compatibility with VQC.
    """

    if features_per_layer is None:
        features_per_layer = config.FEATURES_PER_LAYER

    x = ParameterVector("x", features_per_layer)

    if lambdas is None:
        lambdas = np.ones(num_qubits)
    else:
        lambdas = np.asarray(lambdas, dtype=float)

    qc = QuantumCircuit(num_qubits)

    # Stage 0: local encoding
    for j in range(num_qubits):
        qc.ry(np.pi * x[j % features_per_layer], j)

    # Stage 1: immediate neighbor entanglement
    for j in range(num_qubits):
        x_a = x[j % features_per_layer]
        x_b = x[(j + 1) % features_per_layer]
        angle = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
        qc.crx(angle, j, (j + 1) % num_qubits)

    # Stage 2: next-nearest neighbor correlations
    for j in range(num_qubits):
        vals = [x[j % features_per_layer], x[(j + 1) % features_per_layer], x[(j + 2) % features_per_layer]]
        avg = sum(vals) / 3
        qc.cry(np.pi * avg, j, (j + 2) % num_qubits)

    # Stage 3: adaptive CRot with layer-dependent scaling
    for j in range(num_qubits):
        x_a = x[j % features_per_layer]
        x_b = x[(j + 1) % features_per_layer]
        scale = lambdas[j % len(lambdas)]
        angle = np.pi * scale * 0.5 * (x_a + x_b)
        qc.cu(angle, 0.0, 0.0, 0.0, j, (j + 3) % num_qubits)

    # Stage 4: long-range entanglement via IsingXY-like coupling
    half = num_qubits // 2
    for j in range(half):
        qc.append(IsingXYGate(np.pi * gamma), [j, j + half])

    # Stage 5: global multi-qubit rotation
    global_angle = np.pi * delta * x[min(features_per_layer - 1, len(x) - 1)]
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


def amplitude_feature_map(num_qubits: int) -> QuantumCircuit:
    """Prepare an equal superposition state on ``num_qubits``."""
    vec = np.ones(2 ** num_qubits)
    return amplitude_encoding(vec, n_qubits=num_qubits)


AVAILABLE_MAPS = {
    "zz": ZZFeatureMap,
    "z": ZFeatureMap,
    "pauli": lambda num_qubits: PauliFeatureMap(feature_dimension=num_qubits, paulis=["X", "Z"]),
    "adaptive": adaptive_feature_map,
    "npqc": npqc_feature_map,
    "yzcx": yzcx_feature_map,
    "amplitude": amplitude_feature_map,
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

