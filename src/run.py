# train.py
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from model import QuantumLLPModel
from trainer import train_model
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
    DEVICE,
)
from data_utils import get_dataset_class, get_transform

# 1. データセットの準備
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


# 2. 教師のクラス分布（仮にbagごとに均等とする）
teacher_probs_train = torch.tensor([
    TEACHER_PROBS_EVEN if i % 2 == 0 else TEACHER_PROBS_ODD
    for i in range(len(train_loader))
], device=DEVICE)
teacher_probs_val = torch.tensor([
    TEACHER_PROBS_EVEN if i % 2 == 0 else TEACHER_PROBS_ODD
    for i in range(len(val_loader))
], device=DEVICE)

# 3. モデルの定義と訓練
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

# 4. モデル保存
torch.save(model.state_dict(), "trained_quantum_llp.pt")
print("Model saved to trained_quantum_llp.pt")
