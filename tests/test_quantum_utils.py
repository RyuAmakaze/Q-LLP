import sys
import os
import pytest

torch = pytest.importorskip("torch")
qiskit = pytest.importorskip("qiskit")
pytest.importorskip("matplotlib")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_utils import data_to_circuit
from qiskit import QuantumCircuit

from quantum_utils import save_circuit_png, save_model_circuit
from model import QuantumLLPModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_data_to_circuit_cuda_tensor():
    angles = torch.tensor([0.0, 1.0], device="cuda")
    circuit = data_to_circuit(angles)
    assert isinstance(circuit, QuantumCircuit)


def test_save_circuit_png(tmp_path):
    angles = [0.0, 0.5]
    circuit = data_to_circuit(angles)
    out_file = tmp_path / "circuit.png"
    save_circuit_png(circuit, out_file)
    assert out_file.exists()


def test_save_model_circuit(tmp_path):
    model = QuantumLLPModel(n_qubits=2)
    out_file = tmp_path / "model_circuit.png"
    save_model_circuit(model, out_file)
    assert out_file.exists()
