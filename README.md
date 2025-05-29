# Q-LLP
Quantum Learning from Label Proportion

## 実行方法
1. 任意で仮想環境を作成します。
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. 依存パッケージをインストールします。
   ```bash
   pip install torch torchvision qiskit pytest
   ```
3. 学習を実行します。
   ```bash
   python src/run.py
   ```
   学習が完了すると `trained_quantum_llp.pt` が作成されます。
   CUDA が利用可能な環境では自動的に GPU を使用して計算します。

## Docker での実行
1. Docker イメージをビルドします。
   ```bash
   docker build -t q-llp -f Dockerfile/Dockerfile .
   ```
2. コンテナを起動して学習を実行します。
   ```bash
   docker run --rm q-llp
   ```

## テスト
`pytest` を実行してユニットテストを確認できます。
```bash
pytest
```

## 量子回路のシミュレーション

`QuantumLLPModel` はデフォルトでは解析的に測定確率を計算しますが、
`use_circuit=True` を指定すると、実際に `qiskit` の量子回路を構築して
シミュレーションすることもできます。

```python
from model import QuantumLLPModel
import torch

model = QuantumLLPModel(n_qubits=4, use_circuit=True)
features = torch.rand(1, 4)
probs = model(features)
print(probs)
```

このモードは勾配計算ができないため学習用途ではなく、回路の挙動を確
認したい場合に使用してください。
