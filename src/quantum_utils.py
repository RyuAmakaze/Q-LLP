import numpy as np
import torch

import config

from typing import List, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    CRXGate,
    CRYGate,
    CU3Gate,
    RXXGate,
)
try:  # Qiskit <2.0 uses IsingXYGate, >=2.0 renamed it
    from qiskit.circuit.library import IsingXYGate
except Exception:  # pragma: no cover - handle version differences
    try:
        from qiskit.circuit.library import XXPlusYYGate as IsingXYGate
    except Exception:
        from qiskit.circuit.library import XYGate as IsingXYGate


def _to_numpy(x):
    if torch.is_tensor(x):
        try:
            return x.detach().cpu().numpy()
        except Exception:  # pragma: no cover - handle missing numpy
            return np.asarray(x.detach().cpu().tolist(), dtype=float)
    return np.asarray(x, dtype=float)

def multi_rz(qc: QuantumCircuit, qubits: list[int], theta: float):
    # CNOT チェーン
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
    # 最後の量子ビットに RZ をかける
    qc.rz(theta, qubits[-1])
    # CNOT チェーンを戻す
    for i in reversed(range(len(qubits) - 1)):
        qc.cx(qubits[i], qubits[i + 1])

def amplitude_encoding(x, n_qubits=config.NUM_QUBITS):
    """Return a circuit preparing ``x`` as amplitudes on ``n_qubits``.

    Parameters
    ----------
    x : Sequence[float] or torch.Tensor
        Real-valued feature vector to encode. The vector is normalised to
        unit length and padded or truncated to match ``2 ** n_qubits``.
    n_qubits : int, optional
        Number of qubits available for the state preparation.
    """

    vec = _to_numpy(x).astype(float).flatten()
    target_len = 2 ** n_qubits
    if vec.size < target_len:
        vec = np.pad(vec, (0, target_len - vec.size))
    elif vec.size > target_len:
        vec = vec[:target_len]

    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("amplitude_encoding: input vector has zero norm")
    vec = vec / norm

    qc = QuantumCircuit(n_qubits)
    qc.initialize(vec.tolist(), list(range(n_qubits)))
    return qc

def amplitude_data_to_circuit(x, params=None, entangling=False, n_output_qubits=0):
    """Return a circuit with amplitude-encoded ``x`` followed by parameter layers."""

    if params is not None:
        n_qubits = params.shape[-1]
    else:
        n_qubits = config.NUM_QUBITS

    qc = amplitude_encoding(x, n_qubits)

    if params is not None:
        params = _to_numpy(params)
        params = np.atleast_2d(params)
        feature_qubits = n_qubits - int(n_output_qubits)
        for layer_idx, layer in enumerate(params):
            for q, theta in enumerate(layer):
                if layer_idx == 0 and q >= feature_qubits:
                    continue
                qc.rz(float(theta), q)
            if entangling and n_qubits > 1:
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)

    return qc

def data_to_circuit(angles, params=None, entangling=False, n_output_qubits=0):
    """Return a QuantumCircuit encoding ``angles`` via Y rotations.

    Parameters
    ----------
    angles : Sequence[float] or torch.Tensor
        Rotation angles for RY gates on each qubit.
    params : Sequence[float] or torch.Tensor, optional
        Rotation angles for each additional layer.  If ``params`` is
        one-dimensional and ``entangling`` is ``False`` the values are
        added directly to ``angles`` for backwards compatibility.  If
        ``params`` is two-dimensional it is interpreted as
        ``(num_layers, n_qubits)`` with each layer applied sequentially
        using ``RZ`` rotations.
    entangling : bool, optional
        If ``True`` a chain of ``CX`` gates is inserted after each
        parameterized layer to introduce entanglement.
    n_output_qubits : int, optional
        Number of dedicated output qubits placed at the end of the register.
        ``RY`` and the first layer of ``RZ`` rotations are skipped for these
        qubits.

    Notes
    -----
    If qiskit is not installed, this function raises ``ImportError``.
    """

    angles = _to_numpy(angles)
    n_qubits = angles.shape[0]
    feature_qubits = n_qubits - int(n_output_qubits)

    # Backwards compatible path: single parameter vector without entanglement
    if params is not None and not entangling and np.ndim(params) == 1:
        params = _to_numpy(params)
        angles = angles + np.array(params, dtype=float)
        qc = QuantumCircuit(n_qubits)
        for i, theta in enumerate(angles[:feature_qubits]):
            qc.ry(float(theta), i)
        return qc

    qc = QuantumCircuit(n_qubits)
    for i, theta in enumerate(angles[:feature_qubits]):
        qc.ry(float(theta), i)

    if params is not None:
        params = _to_numpy(params)
        params = np.atleast_2d(params)
        for layer_idx, layer in enumerate(params):
            for q, theta in enumerate(layer):
                if layer_idx == 0 and q >= feature_qubits:
                    continue
                qc.rz(float(theta), q)
            if entangling and n_qubits > 1:
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)
    return qc

def circuit_state_probs(
    circuit: QuantumCircuit,
    qargs: Optional[List[int]] = None,
    shots: Optional[int] = None,
):
    """Simulate ``circuit`` and return measurement probabilities.

    When CUDA and a GPU-enabled ``AerSimulator`` are available the
    simulation is executed on the GPU for improved performance.  If the
    GPU simulator is unavailable the function falls back to the default
    :class:`~qiskit.quantum_info.Statevector` implementation.
    """

    if circuit.num_qubits > 24:
        raise ValueError(
            f"circuit_state_probs: {circuit.num_qubits} qubits exceeds the 24 qubit limit of Statevector simulation"
        )

    if qargs is None:
        qargs = list(range(circuit.num_qubits))

    if shots is not None:
        from qiskit_aer import AerSimulator
        from qiskit import ClassicalRegister

        circ = circuit.copy()
        if circ.num_clbits < len(qargs):
            circ.add_register(ClassicalRegister(len(qargs) - circ.num_clbits))
        circ.measure(qargs, range(len(qargs)))
        sim = AerSimulator(device='GPU')
        circ = transpile(circ, backend=sim)
        result = sim.run(circ, shots=shots).result()
        counts = result.get_counts()
        probs = np.zeros(2 ** len(qargs), dtype=float)
        for bitstr, count in counts.items():
            idx = int(bitstr[::-1], 2)
            probs[idx] = count / shots
    else:
        probs = None

        if torch.cuda.is_available():
            try:
                from qiskit_aer import AerSimulator

                sim = AerSimulator(method="statevector", device="GPU")
                circ = circuit.copy()
                circ.save_statevector()
                circ = transpile(circ, backend=sim)
                result = sim.run(circ).result()
                state = result.get_statevector()
                probs = state.probabilities(qargs=qargs[::-1])
            except Exception:
                probs = None

        if probs is None:
            state = Statevector.from_instruction(circuit)
            probs = state.probabilities(qargs=qargs[::-1])

    return torch.tensor(probs, dtype=torch.float32)

def adaptive_data_to_circuit(
    x,
    params=None,
    *,
    n_qubits=config.NUM_QUBITS,
    features_per_layer=config.FEATURES_PER_LAYER,
    entangling=False,
    n_output_qubits: int = 0,
    lambdas=None,
    gamma=1.0,
    delta=1.0,
):
    """Return circuit with adaptive entangling encoding followed by parameter layers."""

    qc = adaptive_entangling_circuit(
        x,
        n_qubits=n_qubits,
        features_per_layer=features_per_layer,
        lambdas=lambdas,
        gamma=gamma,
        delta=delta,
        n_output_qubits=n_output_qubits,
    )

    if params is not None:
        params = _to_numpy(params)
        params = np.atleast_2d(params)
        feature_qubits = n_qubits - int(n_output_qubits)
        for layer_idx, layer in enumerate(params):
            for q, theta in enumerate(layer):
                if layer_idx == 0 and q >= feature_qubits:
                    continue
                qc.rz(float(theta), q)
            if entangling and n_qubits > 1:
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)

    return qc

def parameter_shift_gradients(
    angles,
    params,
    shift=np.pi / 2,
    entangling=False,
    qargs: Optional[List[int]] = None,
    shots: Optional[int] = None,
    n_output_qubits: int = 0,
    *,
    adaptive: bool = False,
    amplitude: bool = False,
    features_per_layer: int = config.FEATURES_PER_LAYER,
    lambdas=None,
    gamma: float = 1.0,
    delta: float = 1.0,
):
    """Return probabilities and gradients via the parameter-shift rule.

    Parameters
    ----------
    angles : Sequence[float] or torch.Tensor
        Data-encoding rotation angles for each qubit.
    params : Sequence[float] or torch.Tensor
        Additional learned rotation angles. Can be ``(n_qubits,)`` or
        ``(num_layers, n_qubits)``.
    shift : float, optional
        Shift amount for the parameter-shift rule (default: ``π/2``).
    entangling : bool, optional
        If ``True`` entangling ``CX`` gates are inserted between layers.
    n_output_qubits : int, optional
        Number of dedicated output qubits to exclude from the first layer
        rotations.
    amplitude : bool, optional
        If ``True`` the data is encoded using amplitude initialization rather
        than rotation angles.
    """


    angles = _to_numpy(angles)

    params = _to_numpy(params)

    # Backwards compatible path: single layer without entanglement
    if params.ndim == 1 and not entangling:
        if adaptive:
            base_circuit = adaptive_data_to_circuit(
                angles,
                params,
                entangling=False,
                n_qubits=params.shape[-1],
                n_output_qubits=n_output_qubits,
                features_per_layer=features_per_layer,
                lambdas=lambdas,
                gamma=gamma,
                delta=delta,
            )
        elif amplitude:
            base_circuit = amplitude_data_to_circuit(
                angles,
                params,
                entangling=False,
                n_output_qubits=n_output_qubits,
            )
        else:
            base_circuit = data_to_circuit(
                angles,
                params,
                entangling=False,
                n_output_qubits=n_output_qubits,
            )
        base_probs = circuit_state_probs(base_circuit, qargs=qargs, shots=shots)

        grads = []
        for i in range(len(params)):
            shift_vec = np.zeros_like(params)
            shift_vec[i] = shift
            if adaptive:
                plus_circ = adaptive_data_to_circuit(
                    angles,
                    params + shift_vec,
                    entangling=False,
                    n_qubits=params.shape[-1],
                    n_output_qubits=n_output_qubits,
                    features_per_layer=features_per_layer,
                    lambdas=lambdas,
                    gamma=gamma,
                    delta=delta,
                )
                minus_circ = adaptive_data_to_circuit(
                    angles,
                    params - shift_vec,
                    entangling=False,
                    n_qubits=params.shape[-1],
                    n_output_qubits=n_output_qubits,
                    features_per_layer=features_per_layer,
                    lambdas=lambdas,
                    gamma=gamma,
                    delta=delta,
                )
            else:
                if amplitude:
                    plus_circ = amplitude_data_to_circuit(
                        angles,
                        params + shift_vec,
                        entangling=False,
                        n_output_qubits=n_output_qubits,
                    )
                    minus_circ = amplitude_data_to_circuit(
                        angles,
                        params - shift_vec,
                        entangling=False,
                        n_output_qubits=n_output_qubits,
                    )
                else:
                    plus_circ = data_to_circuit(
                        angles,
                        params + shift_vec,
                        entangling=False,
                        n_output_qubits=n_output_qubits,
                    )
                    minus_circ = data_to_circuit(
                        angles,
                        params - shift_vec,
                        entangling=False,
                        n_output_qubits=n_output_qubits,
                    )
            plus_probs = circuit_state_probs(plus_circ, qargs=qargs, shots=shots)
            minus_probs = circuit_state_probs(minus_circ, qargs=qargs, shots=shots)
            grad = 0.5 * (plus_probs - minus_probs)
            grads.append(grad)
        grads = torch.stack(grads, dim=0)
        return base_probs, grads

    # Multi-layer or entangling case
    params = np.atleast_2d(params)
    num_layers, n_qubits = params.shape

    if adaptive:
        base_circuit = adaptive_data_to_circuit(
            angles,
            params,
            entangling=entangling,
            n_qubits=n_qubits,
            n_output_qubits=n_output_qubits,
            features_per_layer=features_per_layer,
            lambdas=lambdas,
            gamma=gamma,
            delta=delta,
        )
    elif amplitude:
        base_circuit = amplitude_data_to_circuit(
            angles,
            params,
            entangling=entangling,
            n_output_qubits=n_output_qubits,
        )
    else:
        base_circuit = data_to_circuit(
            angles,
            params,
            entangling=entangling,
            n_output_qubits=n_output_qubits,
        )
    base_probs = circuit_state_probs(base_circuit, qargs=qargs, shots=shots)

    grads = torch.zeros(num_layers, n_qubits, base_probs.numel(), dtype=base_probs.dtype)

    for layer in range(num_layers):
        for q in range(n_qubits):
            shift_mat = np.zeros_like(params)
            shift_mat[layer, q] = shift
            if adaptive:
                plus_circ = adaptive_data_to_circuit(
                    angles,
                    params + shift_mat,
                    entangling=entangling,
                    n_qubits=n_qubits,
                    n_output_qubits=n_output_qubits,
                    features_per_layer=features_per_layer,
                    lambdas=lambdas,
                    gamma=gamma,
                    delta=delta,
                )
                minus_circ = adaptive_data_to_circuit(
                    angles,
                    params - shift_mat,
                    entangling=entangling,
                    n_qubits=n_qubits,
                    n_output_qubits=n_output_qubits,
                    features_per_layer=features_per_layer,
                    lambdas=lambdas,
                    gamma=gamma,
                    delta=delta,
                )
            elif amplitude:
                plus_circ = amplitude_data_to_circuit(
                    angles,
                    params + shift_mat,
                    entangling=entangling,
                    n_output_qubits=n_output_qubits,
                )
                minus_circ = amplitude_data_to_circuit(
                    angles,
                    params - shift_mat,
                    entangling=entangling,
                    n_output_qubits=n_output_qubits,
                )
            else:
                plus_circ = data_to_circuit(
                    angles,
                    params + shift_mat,
                    entangling=entangling,
                    n_output_qubits=n_output_qubits,
                )
                minus_circ = data_to_circuit(
                    angles,
                    params - shift_mat,
                    entangling=entangling,
                    n_output_qubits=n_output_qubits,
                )
            plus_probs = circuit_state_probs(plus_circ, qargs=qargs, shots=shots)
            minus_probs = circuit_state_probs(minus_circ, qargs=qargs, shots=shots)
            grad = 0.5 * (plus_probs - minus_probs)
            grads[layer, q] = grad

    return base_probs, grads


def adaptive_entangling_circuit(
    x,
    *,
    n_qubits=config.NUM_QUBITS,
    features_per_layer=config.FEATURES_PER_LAYER,
    lambdas=None,
    gamma=1.0,
    delta=1.0,
    n_output_qubits: int = 0,
):
    """Return a multi-stage entangling circuit for ``x``.

    This helper implements the six-stage design discussed in the
    project notes. The number of qubits and required features can be
    configured via :mod:`config`.

    Parameters
    ----------
    x : Sequence[float] or torch.Tensor
        Input features for a single layer.  At least
        ``features_per_layer`` elements are required.
    n_qubits : int, optional
        Number of qubits used in the circuit.
    features_per_layer : int, optional
        Number of features consumed from ``x``.  By default this is
        :data:`config.FEATURES_PER_LAYER`.
    lambdas : Sequence[float], optional
        Per-qubit scaling factors for stage 3. If ``None`` all ones are
        used.
    gamma : float, optional
        Scaling factor for the long-range entangling gate (stage 4).
    delta : float, optional
        Scaling factor for the global ``MultiRZ`` gate (stage 5).
    """


    x = _to_numpy(x)

    if len(x) < features_per_layer:
        raise ValueError(
            f"adaptive_entangling_circuit requires at least {features_per_layer} features"
        )

    if lambdas is None:
        lambdas = np.ones(n_qubits)
    else:
        lambdas = np.asarray(lambdas, dtype=float)

    feature_qubits = n_qubits - int(n_output_qubits)

    qc = QuantumCircuit(n_qubits)

    # Stage 0: local encoding
    for j in range(feature_qubits):
        angle = np.pi * x[j]
        qc.ry(float(angle), j)

    # Stage 1: immediate neighbor entanglement
    for j in range(feature_qubits):
        x_a = x[j]
        x_b = x[(j + 1) % feature_qubits]
        angle = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
        qc.append(CRXGate(angle), [j, (j + 1) % feature_qubits])

    # Stage 2: next-nearest neighbor correlations
    for j in range(feature_qubits):
        vals = [x[j], x[(j + 1) % feature_qubits], x[(j + 2) % feature_qubits]]
        angle = np.pi * float(np.mean(vals))
        qc.append(CRYGate(angle), [j, (j + 2) % feature_qubits])

    # Stage 3: adaptive CRot with layer-dependent scaling
    for j in range(feature_qubits):
        x_a = x[j]
        x_b = x[(j + 1) % feature_qubits]
        scale = lambdas[j % len(lambdas)]
        angle = np.pi * scale * 0.5 * (x_a + x_b)
        qc.append(CU3Gate(angle, 0.0, 0.0), [j, (j + 3) % feature_qubits])

    # Stage 4: long-range entanglement via IsingXY-like coupling
    half = feature_qubits // 2
    for j in range(half):
        qc.append(IsingXYGate(np.pi * gamma), [j, j + half])

    # Stage 5: global multi-qubit rotation
    global_angle = np.pi * delta * x[min(features_per_layer - 1, len(x) - 1)]
    multi_rz(qc, list(range(feature_qubits)), global_angle)

    return qc
