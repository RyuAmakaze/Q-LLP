import sys
import os
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qiskit")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from model import QuantumLLPModel


def test_forward_with_adaptive_encoding():
    config.MEASURE_SHOTS = None
    model = QuantumLLPModel(n_qubits=4, adaptive=True)
    x_batch = torch.rand(2, config.FEATURES_PER_LAYER)
    probs = model(x_batch)
    assert probs.shape[0] == 2
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
