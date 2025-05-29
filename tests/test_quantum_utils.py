import sys
import os
import pytest

torch = pytest.importorskip("torch")
qiskit = pytest.importorskip("qiskit")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_utils import data_to_circuit
from qiskit import QuantumCircuit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_data_to_circuit_cuda_tensor():
    angles = torch.tensor([0.0, 1.0], device="cuda")
    circuit = data_to_circuit(angles)
    assert isinstance(circuit, QuantumCircuit)

