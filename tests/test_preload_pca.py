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
