# train.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import QuantumLLPModel
from trainer import train_model
from config import (
    DATA_ROOT,
    SUBSET_SIZE,
    BATCH_SIZE,
    ENCODING_DIM,
    SHUFFLE_DATA,
    NUM_QUBITS,
    RUN_EPOCHS,
    RUN_LR,
    TEACHER_PROBS_EVEN,
    TEACHER_PROBS_ODD,
)

# 1. データセットの準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),  # flatten
    transforms.Lambda(lambda x: x[:ENCODING_DIM])  # 次元削減
])

dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
data_indices = list(range(SUBSET_SIZE))
data_subset = Subset(dataset, data_indices)
dataloader = DataLoader(data_subset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)

# 2. 教師のクラス分布（仮にbagごとに均等とする）
teacher_probs = torch.tensor([
    TEACHER_PROBS_EVEN if i % 2 == 0 else TEACHER_PROBS_ODD
    for i in range(len(dataloader))
])

# 3. モデルの定義と訓練
model = QuantumLLPModel(n_qubits=NUM_QUBITS)
train_model(model, dataloader, teacher_probs, epochs=RUN_EPOCHS, lr=RUN_LR)

# 4. モデル保存
torch.save(model.state_dict(), "trained_quantum_llp.pt")
print("Model saved to trained_quantum_llp.pt")