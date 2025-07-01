import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info import Statevector

from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from data_utils import get_transform, preload_dataset
from quantum_utils import amplitude_encoding

# --- Configuration ---
DATA_ROOT = "./data"
NUM_CLASSES = 4
SUBSET_SIZE = 100
TEST_SUBSET_SIZE = 30
BAG_SIZE = 1
NUM_QUBITS = 9
NUM_LAYERS = 5
EPOCHS = 3
LR = 0.1

def amplitude_to_real(v, n_qubits: int = NUM_QUBITS) -> torch.Tensor:
    """Encode ``v`` as amplitudes and return the real part of the statevector."""
    qc = amplitude_encoding(v, n_qubits=n_qubits)
    sv = Statevector.from_instruction(qc)
    return torch.tensor(sv.data.real[:n_qubits])

def parse_args():
    parser = argparse.ArgumentParser(description="Simple VQC LLP example")
    parser.add_argument(
        "--use-dino",
        action="store_true",
        help="use get_transform(use_dino=True) for feature extraction",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="preload dataset and apply PCA to NUM_QUBITS dimensions",
    )
    parser.add_argument(
        "--amplitude",
        action="store_true",
        help="encode data using quantum_utils.amplitude_encoding",
    )
    return parser.parse_args()

args = parse_args()

def main():
    print("args", args)
    print("DEVICE", DEVICE)
    
    # VQC feature map and ansatz
    feature_map = ZZFeatureMap(feature_dimension=NUM_QUBITS)
    ansatz = TwoLocal(NUM_QUBITS, ["ry", "rz"], "cx", reps=NUM_LAYERS)

    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=None, output_shape=NUM_CLASSES)
    nn = getattr(vqc, "neural_network", None)
    if nn is None:
        nn = getattr(vqc, "_neural_network")
    model = TorchConnector(nn).to(DEVICE)

    if args.use_dino:
        transform = get_transform(use_dino=True)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
        )

    if args.amplitude:
        transform = transforms.Compose([transform, transforms.Lambda(amplitude_to_real)])
    else:
        transform = transforms.Compose(
            [transform, transforms.Lambda(lambda x: x[:NUM_QUBITS])]
        )

    dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    indices = [i for i, t in enumerate(dataset.targets) if t < NUM_CLASSES][:SUBSET_SIZE]
    train_subset = Subset(dataset, indices)

    val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    val_indices = [i for i, t in enumerate(val_dataset.targets) if t < NUM_CLASSES][:TEST_SUBSET_SIZE]
    val_subset = Subset(val_dataset, val_indices)

    if args.preload:
        train_subset = preload_dataset(train_subset, batch_size=BAG_SIZE, pca_dim=NUM_QUBITS)
        val_subset = preload_dataset(val_subset, batch_size=BAG_SIZE, pca_dim=NUM_QUBITS)

    train_loader = DataLoader(train_subset, batch_size=BAG_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BAG_SIZE)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()


    for epoch in range(EPOCHS):
        model.train()
        for x, y in tqdm(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            batch_size = x.size(0)
            labels = torch.nn.functional.one_hot(y, NUM_CLASSES).float()
            teacher = labels.mean(dim=0)

            optim.zero_grad()
            preds = model(x)
            bag_pred = preds.mean(dim=0)
            loss = loss_fn(bag_pred, teacher)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(val_loader):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                preds = model(x)
                pred_class = preds.argmax(dim=1)
                correct += (pred_class == y).sum().item()
                total += y.size(0)
            print(f'Epoch {epoch+1} Acc {correct/total:.3f}')

if __name__ == "__main__":
    main()
