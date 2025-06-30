import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

# --- Configuration ---
DATA_ROOT = './data'
NUM_CLASSES = 4
SUBSET_SIZE = 100
TEST_SUBSET_SIZE = 30
BAG_SIZE = 5
NUM_QUBITS = 6
NUM_LAYERS = 2
EPOCHS = 3
LR = 0.1

# VQC feature map and ansatz
feature_map = ZZFeatureMap(feature_dimension=NUM_QUBITS)
ansatz = TwoLocal(NUM_QUBITS, ['ry', 'rz'], 'cx', reps=NUM_LAYERS)

vqc = VQC(feature_map=feature_map, ansatz=ansatz,
          optimizer=None, num_classes=NUM_CLASSES)
nn = getattr(vqc, "neural_network", None)
if nn is None:
    nn = getattr(vqc, "_neural_network")
model = TorchConnector(nn)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)[:NUM_QUBITS])
])

dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
indices = [i for i, t in enumerate(dataset.targets) if t < NUM_CLASSES][:SUBSET_SIZE]
train_subset = Subset(dataset, indices)

val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
val_indices = [i for i, t in enumerate(val_dataset.targets) if t < NUM_CLASSES][:TEST_SUBSET_SIZE]
val_subset = Subset(val_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=BAG_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BAG_SIZE)

optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
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
        for x, y in val_loader:
            preds = model(x)
            pred_class = preds.argmax(dim=1)
            correct += (pred_class == y).sum().item()
            total += y.size(0)
        print(f'Epoch {epoch+1} Acc {correct/total:.3f}')
