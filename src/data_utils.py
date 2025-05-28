from torchvision import datasets, transforms
from config import ENCODING_DIM


def get_dataset_class(name: str):
    mapping = {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dataset: {name}")
    return mapping[name]


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
        transforms.Lambda(lambda x: x[:ENCODING_DIM])
    ])
