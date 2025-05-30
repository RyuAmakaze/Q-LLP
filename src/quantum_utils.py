import numpy as np
import torch

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except Exception:  # pragma: no cover - qiskit may not be installed
    QuantumCircuit = None
    Statevector = None


def data_to_circuit(angles, params=None, entangling=False):
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

    Notes
    -----
    If qiskit is not installed, this function raises ``ImportError``.
    """
    if QuantumCircuit is None:
        raise ImportError("qiskit is required for circuit construction")

    if torch.is_tensor(angles):
        angles = angles.detach().cpu().numpy()
    else:
        angles = np.array(angles, dtype=float)
    n_qubits = angles.shape[0]

    # Backwards compatible path: single parameter vector without entanglement
    if params is not None and not entangling and np.ndim(params) == 1:
        if torch.is_tensor(params):
            params = params.detach().cpu().numpy()
        angles = angles + np.array(params, dtype=float)
        qc = QuantumCircuit(n_qubits)
        for i, theta in enumerate(angles):
            qc.ry(float(theta), i)
        return qc

    qc = QuantumCircuit(n_qubits)
    for i, theta in enumerate(angles):
        qc.ry(float(theta), i)

    if params is not None:
        if torch.is_tensor(params):
            params = params.detach().cpu().numpy()
        params = np.array(params, dtype=float)
        params = np.atleast_2d(params)
        for layer in params:
            for q, theta in enumerate(layer):
                qc.rz(float(theta), q)
            if entangling and n_qubits > 1:
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)
    return qc


def circuit_state_probs(circuit):
    """Simulate ``circuit`` and return measurement probabilities."""
    if Statevector is None:
        raise ImportError("qiskit is required for circuit simulation")

    state = Statevector.from_instruction(circuit)
    probs = state.probabilities()
    return torch.tensor(probs, dtype=torch.float32)

def parameter_shift_gradients(angles, params, shift=np.pi/2):
    """Return probabilities and gradients via the parameter-shift rule.

    Parameters
    ----------
    angles : Sequence[float] or torch.Tensor
        Data-encoding rotation angles for each qubit.
    params : Sequence[float] or torch.Tensor
        Additional learned rotation angles.
    shift : float, optional
        Shift amount for the parameter-shift rule (default: ``π/2``).
    """
    if QuantumCircuit is None:
        raise ImportError("qiskit is required for circuit simulation")

    if torch.is_tensor(angles):
        angles = angles.detach().cpu().numpy()
    else:
        angles = np.array(angles, dtype=float)

    if torch.is_tensor(params):
        params = params.detach().cpu().numpy()
    else:
        params = np.array(params, dtype=float)

    base_circuit = data_to_circuit(angles, params, entangling=False)
    base_probs = circuit_state_probs(base_circuit)

    grads = []
    for i in range(len(params)):
        shift_vec = np.zeros_like(params)
        shift_vec[i] = shift
        plus_circ = data_to_circuit(angles, params + shift_vec, entangling=False)
        minus_circ = data_to_circuit(angles, params - shift_vec, entangling=False)
        plus_probs = circuit_state_probs(plus_circ)
        minus_probs = circuit_state_probs(minus_circ)
        grad = 0.5 * (plus_probs - minus_probs)
        grads.append(grad)
    grads = torch.stack(grads, dim=0)
    return base_probs, grads
