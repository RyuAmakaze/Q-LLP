import types
import torch

try:
    from torchvision import datasets as tv_datasets, transforms as tv_transforms
except Exception:  # pragma: no cover - torchvision may not be installed
    class DummyCompose:
        def __init__(self, funcs):
            self.funcs = funcs
        def __call__(self, x):
            for f in self.funcs:
                x = f(x)
            return x

    class DummyLambda:
        def __init__(self, func):
            self.func = func
        def __call__(self, x):
            return self.func(x)

    class DummyToTensor:
        def __call__(self, x):
            return torch.tensor(x)

    tv_datasets = types.SimpleNamespace(MNIST=object, CIFAR10=object, CIFAR100=object)
    tv_transforms = types.SimpleNamespace(Compose=DummyCompose, Lambda=DummyLambda, ToTensor=DummyToTensor)

from config import ENCODING_DIM


def get_dataset_class(name: str):
    mapping = {
        "MNIST": tv_datasets.MNIST,
        "CIFAR10": tv_datasets.CIFAR10,
        "CIFAR100": tv_datasets.CIFAR100,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dataset: {name}")
    return mapping[name]


def _maybe_to_tensor(x):
    """Convert input to a tensor if it isn't already one."""
    if isinstance(x, torch.Tensor):
        return x
    return tv_transforms.ToTensor()(x)


def get_transform():
    """Return a transform that flattens and truncates images."""
    return tv_transforms.Compose([
        tv_transforms.Lambda(_maybe_to_tensor),
        tv_transforms.Lambda(lambda x: x.view(-1)),
        tv_transforms.Lambda(lambda x: x[:ENCODING_DIM]),
    ])
