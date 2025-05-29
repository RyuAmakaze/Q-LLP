import types
import torch
import random
import math
from typing import Sequence, List
from torch.utils.data import Sampler

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


def filter_indices_by_class(dataset, num_classes):
    """Return indices of samples whose label is < num_classes."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = dataset.labels
    return [i for i, t in enumerate(targets) if int(t) < num_classes]


def compute_proportions(labels, num_classes):
    """Compute normalized label counts for a batch."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts / counts.sum()


class FixedBatchSampler(Sampler[List[int]]):
    """Yield predefined lists of indices as batches."""

    def __init__(self, batches: Sequence[Sequence[int]]):
        self.batches = [list(b) for b in batches]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def create_fixed_proportion_batches(dataset, teacher_probs_list, bag_size, num_classes):
    """Return a FixedBatchSampler where each batch matches the given proportions."""
    dataset_indices = list(range(len(dataset)))

    # Walk to the root dataset to access labels
    base_dataset = dataset
    while hasattr(base_dataset, "indices"):
        base_dataset = base_dataset.dataset

    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        targets = getattr(base_dataset, "labels", None)
    if targets is None and isinstance(base_dataset, torch.utils.data.TensorDataset):
        if len(base_dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset must contain at least two tensors to provide labels"
            )
        targets = base_dataset.tensors[1]
    if targets is None:
        raise ValueError(
            "Could not locate labels. Provide 'targets', 'labels', or use a TensorDataset with labels"
        )

    class_to_indices = {i: [] for i in range(num_classes)}
    for idx in dataset_indices:
        root_idx = idx
        ds = dataset
        # Resolve the index through potentially nested Subset objects
        while hasattr(ds, "indices"):
            root_idx = ds.indices[root_idx]
            ds = ds.dataset
        label = int(targets[root_idx])
        if label < num_classes:
            # store dataset-relative index
            class_to_indices[label].append(idx)

    for idx_list in class_to_indices.values():
        random.shuffle(idx_list)

    batches = []
    for probs in teacher_probs_list:
        raw = [p * bag_size for p in probs]
        counts = [math.floor(c) for c in raw]
        remaining = bag_size - sum(counts)
        fractions = [r - math.floor(r) for r in raw]
        for cls in sorted(range(num_classes), key=lambda i: fractions[i], reverse=True)[:remaining]:
            counts[cls] += 1

        batch = []
        for cls, count in enumerate(counts):
            batch.extend(class_to_indices[cls][:count])
            class_to_indices[cls] = class_to_indices[cls][count:]
        batches.append(batch)

    return FixedBatchSampler(batches)


def create_random_bags(dataset, bag_size, num_classes, shuffle=True):
    """Create random bags and return a sampler and teacher label proportions."""
    dataset_indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(dataset_indices)

    # Walk to the root dataset to access labels
    base_dataset = dataset
    while hasattr(base_dataset, "indices"):
        base_dataset = base_dataset.dataset

    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        targets = getattr(base_dataset, "labels", None)
    if targets is None and isinstance(base_dataset, torch.utils.data.TensorDataset):
        if len(base_dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset must contain at least two tensors to provide labels"
            )
        targets = base_dataset.tensors[1]
    if targets is None:
        raise ValueError(
            "Could not locate labels. Provide 'targets', 'labels', or use a TensorDataset with labels"
        )

    batches = []
    teacher_props = []
    # ignore last incomplete batch
    full_len = len(dataset_indices) - len(dataset_indices) % bag_size
    for start in range(0, full_len, bag_size):
        batch_indices = dataset_indices[start : start + bag_size]
        batches.append(batch_indices)

        labels = []
        for idx in batch_indices:
            root_idx = idx
            ds = dataset
            while hasattr(ds, "indices"):
                root_idx = ds.indices[root_idx]
                ds = ds.dataset
            label = int(targets[root_idx])
            if label < num_classes:
                labels.append(label)
        teacher_props.append(compute_proportions(torch.tensor(labels), num_classes))

    sampler = FixedBatchSampler(batches)
    teacher_tensor = torch.stack(teacher_props)
    return sampler, teacher_tensor
