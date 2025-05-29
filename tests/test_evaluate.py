import sys
import os
import pytest

torch = pytest.importorskip("torch")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from trainer import evaluate_model

class DummyModel(torch.nn.Module):
    def __init__(self, preds):
        super().__init__()
        self.preds = preds
    def forward(self, x):
        return self.preds[: x.shape[0]]

def test_evaluate_model_accuracy():
    inputs = torch.zeros(4, 2)
    labels = torch.tensor([0, 1, 1, 0])
    preds = torch.tensor([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.4, 0.6],
        [0.6, 0.4],
    ], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    model = DummyModel(preds)
    metrics = evaluate_model(model, loader, num_classes=2, device="cpu")
    assert abs(metrics["accuracy"] - 0.75) < 1e-6
