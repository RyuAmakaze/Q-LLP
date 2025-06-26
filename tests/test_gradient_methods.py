import sys
import os
import pytest
np = pytest.importorskip("numpy")

# Skip if PyTorch or qiskit are not installed
torch = pytest.importorskip("torch")
pytest.importorskip("qiskit")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from quantum_utils import parameter_shift_gradients, finite_difference_gradients
from model import QuantumLLPModel


def test_finite_diff_matches_parameter_shift():
    n_qubits = 2
    x = torch.rand(n_qubits)
    params = torch.randn(n_qubits)
    ps_probs, ps_grads = parameter_shift_gradients(np.pi * x, params)
    fd_probs, fd_grads = finite_difference_gradients(np.pi * x, params)
    assert torch.allclose(ps_probs, fd_probs, atol=1e-6)
    assert torch.allclose(ps_grads, fd_grads, atol=1e-2)


def test_backward_with_finite_diff():
    model = QuantumLLPModel(n_qubits=2, num_layers=2, entangling=True, gradient_method="finite_diff")
    x_batch = torch.rand(3, 2)
    pred = model(x_batch)
    loss = pred.mean()
    loss.backward()
    assert model.params.grad is not None
    assert model.params.grad.shape == (2, 2)
