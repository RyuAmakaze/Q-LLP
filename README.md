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
   pip install torch torchvision qiskit qiskit-machine-learning pytest
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
PCA 圧縮前後の特徴量を保存したい場合は `config.PRELOAD_SAVE_BEFORE` および
`config.PRELOAD_SAVE_AFTER` にファイルパスを指定してください。

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

Multi-layer or entangling quantum circuits are differentiable using either
the parameter-shift rule or a finite-difference approximation. The method can
be selected via `config.GRADIENT_METHOD`.

### Dedicated output qubits

Setting `config.NUM_OUTPUT_QUBITS` to a value greater than zero adds
additional qubits that are measured for class prediction. Circuit
simulation is still enabled automatically when entangling layers or
multiple parameterized layers are used. When only a single non-entangling
layer is present, class probabilities for the output qubits are computed
analytically for improved performance.
When using dedicated output qubits, the training labels are converted
to probability vectors of length `2 ** NUM_OUTPUT_QUBITS` so that the
loss computation matches the model output.

### Amplitude encoding

Setting `config.USE_AMPLITUDE_ENCODING = True` enables amplitude-based
state preparation for input features. The feature vector is normalised and
padded or truncated to the available qubits before being used to
initialise the quantum state via `QuantumCircuit.initialize`.

### VQC を用いた LLP のサンプル

`qiskit-machine-learning` がインストールされていれば `src/vqc_llp_example.py`
で VQC を使った簡単な LLP 学習を試すことができます。以下のように実行します。

```bash
python src/vqc_llp_example.py
```

CIFAR10 の一部を用いて学習し、テスト精度が表示されます。
CUDA が利用可能な環境では自動的に GPU を使用して計算します。

`--save-circuit <path>` を指定すると、VQC の回路図を PNG 形式で保存でき
ます。`--print-circuit` を付けるとテキストとして表示されます。
`--save-model <path>` を指定すると学習後の重みを保存できます。

#### DINO 特徴量と振幅エンコーディングを使用する

`vqc_llp_example.py` では以下のフラグを指定することで DINO 特徴量の利用や
PCA 圧縮、振幅エンコーディングを試すことができます。

```bash
python src/vqc_llp_example.py --use-dino --preload --amplitude
```

`--use-dino` は `get_transform(use_dino=True)` を、`--preload` は
`preload_dataset(..., pca_dim=NUM_QUBITS)` を適用します。
`--amplitude` を指定すると各サンプルを `quantum_utils.amplitude_encoding`
で量子状態に変換してから学習を行います。
`--feature-map` には `zz` や `pauli` のほか `adaptive`、`npqc`、`yzcx`、`amplitude`
といった名前を指定して異なるフィーチャーマップを試すこともできます。
