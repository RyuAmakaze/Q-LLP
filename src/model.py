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
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
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
            probs = self._state_probs(angles)[:NUM_CLASSES]
            probs = probs.to(self.params.device)
            probs = probs / probs.sum()
            probs_batch.append(probs)
        return torch.stack(probs_batch)
