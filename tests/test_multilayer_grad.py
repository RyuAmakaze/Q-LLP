import sys
import os
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qiskit")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from model import QuantumLLPModel


def test_multilayer_entangling_backward():
    config.MEASURE_SHOTS = None
    model = QuantumLLPModel(n_qubits=2, num_layers=2, entangling=True)
    x_batch = torch.rand(3, 2)
    pred = model(x_batch)
    loss = pred.mean()
    loss.backward()

    assert model.params.grad is not None
    assert model.params.grad.shape == (2, 2)
