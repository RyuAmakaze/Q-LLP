import sys
import os
import pytest

torch = pytest.importorskip("torch")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_utils import create_fixed_proportion_batches, compute_proportions
from torch.utils.data import TensorDataset, Subset, random_split


def test_sampler_on_nested_subset():
    # base dataset with alternating labels
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    data = torch.zeros(len(labels), 1)
    base = TensorDataset(data, labels)

    # first subset keeps all indices but wrapped in Subset
    first_subset = Subset(base, list(range(len(labels))))

    # create nested subset using random_split
    sub_a, _ = random_split(first_subset, [4, len(first_subset) - 4],
                            generator=torch.Generator().manual_seed(0))

    # derive teacher proportions from the selected subset
    subset_labels = torch.tensor([sub_a[i][1] for i in range(len(sub_a))])
    teacher = compute_proportions(subset_labels, 2)

    sampler = create_fixed_proportion_batches(sub_a, [teacher], len(sub_a), 2)
    batches = list(iter(sampler))
    assert len(batches) == 1
    batch = batches[0]
    # indices should be relative to sub_a
    assert all(0 <= idx < len(sub_a) for idx in batch)

    batch_labels = torch.tensor([sub_a[i][1] for i in batch])
    props = compute_proportions(batch_labels, 2)
    assert torch.allclose(props, teacher)
