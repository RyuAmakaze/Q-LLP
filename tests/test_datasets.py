import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_utils import get_dataset_class, get_transform
from torchvision import datasets
import torch
import config as config


def test_get_dataset_class_cifar10():
    assert get_dataset_class("CIFAR10") is datasets.CIFAR10


def test_get_dataset_class_cifar100():
    assert get_dataset_class("CIFAR100") is datasets.CIFAR100


def test_transform_output_size():
    transform = get_transform()
    x = torch.randn(3, 32, 32)
    out = transform(x)
    assert out.shape[0] == config.ENCODING_DIM

