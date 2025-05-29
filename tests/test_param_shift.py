import sys
import os
import pytest
np = pytest.importorskip("numpy")

# Skip if PyTorch is not installed
torch = pytest.importorskip("torch")

# Skip if qiskit is not installed
pytest.importorskip("qiskit")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from quantum_utils import parameter_shift_gradients
from model import QuantumLLPModel


def test_parameter_shift_matches_autograd():
    n_qubits = 2
    model = QuantumLLPModel(n_qubits=n_qubits)
    x = torch.rand(n_qubits)
    params = torch.randn(n_qubits, requires_grad=True)

    probs = model._state_probs(np.pi * x + params)

    def f(p):
        return model._state_probs(np.pi * x + p)

    jac = torch.autograd.functional.jacobian(f, params)

    ps_probs, grads = parameter_shift_gradients(np.pi * x, params.detach())

    assert torch.allclose(probs, ps_probs, atol=1e-6)
    assert torch.allclose(grads.t(), jac, atol=1e-5)
