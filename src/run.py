import torch
from torch.utils.data import DataLoader, Subset, random_split

from model import QuantumLLPModel
from trainer import train_model
from data_utils import get_dataset_class, get_transform
from config import (
    DATA_ROOT,
    SUBSET_SIZE,
    BATCH_SIZE,
    SHUFFLE_DATA,
    DATASET,
    VAL_SPLIT,
    NUM_QUBITS,
    RUN_EPOCHS,
    RUN_LR,
    TEACHER_PROBS_EVEN,
    TEACHER_PROBS_ODD,
    NUM_CLASSES,
    DEVICE,
)

# Print basic information
print(f"Using dataset: {DATASET}")
print(f"Number of classes: {NUM_CLASSES}")

# 1. Prepare datasets
transform = get_transform()
DatasetClass = get_dataset_class(DATASET)
train_full = DatasetClass(root=DATA_ROOT, train=True, download=True, transform=transform)
test_dataset = DatasetClass(root=DATA_ROOT, train=False, download=True, transform=transform)

subset_indices = list(range(SUBSET_SIZE))
subset = Subset(train_full, subset_indices)
val_size = int(len(subset) * VAL_SPLIT)
train_size = len(subset) - val_size
train_subset, val_subset = random_split(subset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. Teacher class distributions (alternating even/odd)
teacher_probs_train = torch.tensor([
    TEACHER_PROBS_EVEN if i % 2 == 0 else TEACHER_PROBS_ODD
    for i in range(len(train_loader))
], device=DEVICE)
teacher_probs_val = torch.tensor([
    TEACHER_PROBS_EVEN if i % 2 == 0 else TEACHER_PROBS_ODD
    for i in range(len(val_loader))
], device=DEVICE)

# 3. Train model
model = QuantumLLPModel(n_qubits=NUM_QUBITS).to(DEVICE)
train_model(
    model,
    train_loader,
    val_loader,
    teacher_probs_train,
    teacher_probs_val,
    epochs=RUN_EPOCHS,
    lr=RUN_LR,
    device=DEVICE,
)

# 4. Save model
torch.save(model.state_dict(), "trained_quantum_llp.pt")
print("Model saved to trained_quantum_llp.pt")

# 5. Inference on a few test batches
model.eval()
with torch.no_grad():
    for i, (x_batch, _) in enumerate(test_loader):
        x_batch = x_batch.to(DEVICE)
        pred_probs = model(x_batch)
        bag_pred = pred_probs.mean(dim=0)
        print(f"Test batch {i+1} predicted class proportions: {bag_pred.cpu().numpy()}")
        if i >= 1:  # limit output for brevity
            break

