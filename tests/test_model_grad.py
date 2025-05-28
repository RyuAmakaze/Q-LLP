import sys
import os
import pytest
torch = pytest.importorskip("torch")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import QuantumLLPModel


def test_single_backward_step():
    model = QuantumLLPModel(n_qubits=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    x_batch = torch.rand(4, 2)
    target = torch.full((4,), 0.25)

    pred = model(x_batch)
    bag_pred = pred.mean(dim=0)
    loss = loss_fn(bag_pred, target)
    loss.backward()

    assert model.params.grad is not None
    assert torch.any(model.params.grad.abs() > 0)

    optimizer.step()
