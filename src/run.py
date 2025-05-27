# train.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import QuantumLLPModel
from trainer import train_model

# 1. データセットの準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),  # flatten
    transforms.Lambda(lambda x: x[:4])        # 次元削減（例：最初の4次元）
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
data_indices = list(range(100))  # 簡易に100個を選択
data_subset = Subset(dataset, data_indices)
dataloader = DataLoader(data_subset, batch_size=10, shuffle=False)  # 10個ずつbagに

# 2. 教師のクラス分布（仮にbagごとに均等とする）
teacher_probs = torch.tensor([[0.2, 0.8] if i % 2 == 0 else [0.8, 0.2] for i in range(len(dataloader))])

# 3. モデルの定義と訓練
model = QuantumLLPModel(n_qubits=2)
train_model(model, dataloader, teacher_probs, epochs=5, lr=0.1)

# 4. モデル保存
torch.save(model.state_dict(), "trained_quantum_llp.pt")
print("Model saved to trained_quantum_llp.pt")