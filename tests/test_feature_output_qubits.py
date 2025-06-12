import sys
import os
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qiskit")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from model import QuantumLLPModel


def test_circuit_feature_qargs_output_shape():
    config.MEASURE_SHOTS = None
    model = QuantumLLPModel(n_qubits=4, use_circuit=True)
    x_batch = torch.rand(2, 4)
    probs = model(x_batch)
    assert probs.shape == (2, config.NUM_CLASSES)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
