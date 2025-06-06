import sys
import os
import pytest

torch = pytest.importorskip("torch")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_utils import get_dataset_class, get_transform, compute_proportions
from torchvision import datasets
import config as config


def test_get_dataset_class_cifar10():
    assert get_dataset_class("CIFAR10") is datasets.CIFAR10


def test_get_dataset_class_cifar100():
    assert get_dataset_class("CIFAR100") is datasets.CIFAR100


def test_transform_output_size():
    transform = get_transform()
    x = torch.randn(3, 32, 32)
    out = transform(x)
    print("Output shape:", out.shape)
    print("Expected shape:", config.ENCODING_DIM)
    assert out.shape[0] == config.ENCODING_DIM


def test_compute_proportions():
    labels = torch.tensor([0, 1, 1, 2, 3, 3])
    props = compute_proportions(labels, 4)
    assert torch.isclose(props.sum(), torch.tensor(1.0))
    assert props.shape[0] == 4

