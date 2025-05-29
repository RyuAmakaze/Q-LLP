import torch
import torch.nn as nn
import numpy as np
from config import NUM_CLASSES
from quantum_utils import data_to_circuit, circuit_state_probs, QuantumCircuit


def kronecker_product(probs_list):
    """Compute the Kronecker product of a list of probability vectors."""
    result = probs_list[0]
    for p in probs_list[1:]:
        result = torch.einsum("i,j->ij", result, p).reshape(-1)
    return result

class QuantumLLPModel(nn.Module):
    def __init__(self, n_qubits, use_circuit=False):
        """Quantum LLP model.

        Parameters
        ----------
        n_qubits : int
            Number of qubits / features encoded.
        use_circuit : bool, optional
            If ``True`` measurement probabilities are obtained by constructing
            and simulating a :class:`~qiskit.circuit.QuantumCircuit` using
            :func:`data_to_circuit` and :func:`circuit_state_probs`. This path is
            **not differentiable** and mainly provided for inspection or
            debugging.  The default analytic calculation remains the default
            behaviour when ``False``.
        """

        super().__init__()
        self.n_qubits = n_qubits
        self.use_circuit = use_circuit
        self.params = nn.Parameter(torch.randn(n_qubits, dtype=torch.float32))

    def _state_probs(self, angles):
        """Return probabilities of measuring each basis state for given angles."""
        # angles: tensor shape (n_qubits,)
        # Always compute probabilities using differentiable PyTorch operations.
        # Even when qiskit is available we avoid converting tensors to numpy,
        # otherwise gradients are detached and ``loss.backward()`` will fail.
        p0 = torch.cos(angles / 2) ** 2
        p1 = torch.sin(angles / 2) ** 2
        probs_list = [torch.stack([p0[i], p1[i]]) for i in range(self.n_qubits)]
        probs = kronecker_product(probs_list)
        return probs.to(angles.device)

    def forward(self, x_batch):
        x_batch = x_batch.to(self.params.device)
        probs_batch = []
        for x in x_batch:
            if x.shape[0] != self.n_qubits:
                x = x[: self.n_qubits]
            angles = np.pi * x + self.params
            if self.use_circuit:
                if QuantumCircuit is None:
                    raise ImportError("qiskit is required for circuit simulation")
                circuit = data_to_circuit(np.pi * x.cpu(), self.params.detach().cpu())
                probs = circuit_state_probs(circuit)
                probs = probs[:NUM_CLASSES]
                probs = probs.to(self.params.device)
            else:
                probs = self._state_probs(angles)[:NUM_CLASSES]
            probs = probs.to(self.params.device)
            probs = probs / probs.sum()
            probs_batch.append(probs)
        return torch.stack(probs_batch)
