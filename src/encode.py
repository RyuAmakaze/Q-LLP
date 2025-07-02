from qiskit.circuit.library import (
    ZZFeatureMap,
    ZFeatureMap,
    PauliFeatureMap,
)

AVAILABLE_MAPS = {
    "zz": ZZFeatureMap,
    "z": ZFeatureMap,
    "pauli": lambda num_qubits: PauliFeatureMap(feature_dimension=num_qubits, paulis=['X','Z']),
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

