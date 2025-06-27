import sys
import os
import pytest

torch = pytest.importorskip("torch")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_utils import preload_dataset
import config


def test_preload_dataset_pca_reduces_dim():
    data = torch.randn(10, config.ENCODING_DIM)
    labels = torch.zeros(10, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(data, labels)
    ds = preload_dataset(dataset, batch_size=4, pca_dim=5)
    x0, _ = ds[0]
    assert x0.shape[0] == 5
    assert len(ds) == 10


def test_preload_dataset_saves_features(tmp_path):
    data = torch.randn(4, config.ENCODING_DIM)
    labels = torch.arange(4)
    dataset = torch.utils.data.TensorDataset(data, labels)
    before = tmp_path / "before.pt"
    after = tmp_path / "after.pt"
    _ = preload_dataset(
        dataset,
        batch_size=2,
        pca_dim=2,
        save_before=str(before),
        save_after=str(after),
    )
    before_data = torch.load(before)
    after_data = torch.load(after)
    assert torch.equal(before_data["features"], data)
    assert torch.equal(before_data["labels"], labels)
    assert after_data["features"].shape[1] == 2
