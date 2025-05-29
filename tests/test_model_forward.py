import sys
import os
import pytest

torch = pytest.importorskip("torch")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from model import QuantumLLPModel


def test_forward_outputs_sum_to_one():
    n_qubits = config.NUM_QUBITS + 1
    model = QuantumLLPModel(n_qubits=n_qubits)
    x_batch = torch.rand(2, n_qubits)
    probs = model(x_batch)
    assert probs.shape == (2, config.NUM_CLASSES)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
