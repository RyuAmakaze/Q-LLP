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
  データ読み込みのワーカー数は `config.NUM_WORKERS` で調整できます。
  GPU 利用時にワーカーを有効にするため、スクリプト冒頭で
  `torch.multiprocessing.set_start_method("spawn")` を呼び出しています。
  特徴量を事前計算してメモリに展開するには `config.PRELOAD_DATASET` を `True` に設定します。

## Docker での実行
1. Docker イメージをビルドします。
   ```bash
   docker build -t q-llp -f Dockerfile/Dockerfile .
   ```
2. 作業ディレクトリをコンテナにマウントして学習を実行します。GPU を利用する場合は `--gpus all` を指定します。
   ```bash
   sudo docker run --rm --gpus all -v $(pwd):/app -w /app q-llp python src/run.py
   ```

## テスト
`pytest` を実行してユニットテストを確認できます。
```bash
pytest
```

## 回路図の保存
`scripts/save_circuit_png.py` を実行することで、サンプル回路を描画した
PNG ファイルを保存できます。
```bash
python scripts/save_circuit_png.py circuit.png
```

学習後のモデルから回路図を保存するには `src/run.py` を
`--save-circuit` オプション付きで実行します。
```bash
python src/run.py --save-circuit trained_circuit.png
```
