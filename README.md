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
 デフォルトでは利用可能な CPU コア数をそのまま使用しますが、必要に応じて
 `config.NUM_WORKERS` を変更して制限することもできます。
 またこの値は PyTorch のスレッド数にも反映されるため、
 `BAG_SIZE` を 1 より大きくすると CPU 上でも並列推論が可能になります。
GPU 利用時にワーカーを有効にするため、スクリプト冒頭で
`torch.multiprocessing.set_start_method("spawn")` を呼び出しています。
特徴量を事前計算してメモリに展開するには `config.PRELOAD_DATASET` を `True` に設定します。

学習済みモデルから量子回路図を描画するには `plot_circuit.py` を使用します。(configはrun.py実行時と同じに)
以下のように実行すると `circuit.png` に図が保存されます。
```bash
python src/plot_circuit.py trained_quantum_llp.pt -o circuit.png
```
`-o` を省略すると、図がテキストとして標準出力に表示されます。

## Docker での実行
1. Docker イメージをビルドします。
   ```bash
   docker build -t q-llp -f Dockerfile/Dockerfile .
   ```
2. 作業ディレクトリをコンテナにマウントして学習を実行します。GPU を利用する場合は `--gpus all` を指定します。
   ```bash
   docker run --rm --shm-size=2g --gpus all -v $(pwd):/app -w /app q-llp python src/run.py
   ```

Dockerに入るだけ
```bash
docker run --rm --gpus all -v $(pwd):/app -w /app -it q-llp bash
```


## テスト
`pytest` を実行してユニットテストを確認できます。
```bash
pytest
```

Multi-layer or entangling quantum circuits are now differentiated using the
parameter-shift rule.

### Output qubit options

When `config.USE_FEATURE_OUTPUT` is `True` and `NUM_OUTPUT_QUBITS` is `0`,
the highest feature qubits are reused for class prediction. The number of
bits required is determined from `config.NUM_CLASSES`.  Ensure that the
resulting value does not exceed `config.NUM_QUBITS`.
If `NUM_OUTPUT_QUBITS` is set to a positive value, dedicated qubits are
allocated instead. Circuit simulation is still enabled automatically when
entangling layers or multiple parameterized layers are used. When only a
single non-entangling layer is present, class probabilities for the output
qubits are computed analytically for improved performance.  In the dedicated
case the training labels are converted to probability vectors of length
`2 ** NUM_OUTPUT_QUBITS` so that the loss computation matches the model
output.
