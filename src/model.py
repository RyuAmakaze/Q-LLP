import torch
import torch.nn as nn
import numpy as np
from config import NUM_CLASSES
from quantum_utils import (
    data_to_circuit,
    circuit_state_probs,
    parameter_shift_gradients,
    QuantumCircuit,
)


def kronecker_product(probs_list):
    """Compute the Kronecker product of a list of probability vectors."""
    result = probs_list[0]
    for p in probs_list[1:]:
        result = torch.einsum("i,j->ij", result, p).reshape(-1)
    return result


class CircuitProbFunction(torch.autograd.Function):
    """Autograd function for differentiable circuit simulation."""

    @staticmethod
    def forward(ctx, params, x, entangling=False):
        ctx.entangling = entangling
        ctx.save_for_backward(params, x)

        circuit = data_to_circuit(np.pi * x.cpu(), params.cpu(), entangling=entangling)
        probs = circuit_state_probs(circuit)
        return probs.to(params.device)

    @staticmethod
    def backward(ctx, grad_output):
        params, x = ctx.saved_tensors
        angles = np.pi * x.cpu()
        _, grads = parameter_shift_gradients(angles, params.cpu(), entangling=ctx.entangling)
        grads = grads.to(grad_output.device)
        grad_params = torch.einsum("p,lqp->lq", grad_output, grads)
        return grad_params.to(params.device), None, None

class QuantumLLPModel(nn.Module):
    def __init__(self, n_qubits, num_layers=1, use_circuit=False, entangling=False):
        """Quantum LLP model supporting optional deep entangling circuits.

        Parameters
        ----------
        n_qubits : int
            Number of qubits / features encoded.
        num_layers : int, optional
            Number of parameterized layers applied after the data encoding.
        use_circuit : bool, optional
            If ``True`` measurement probabilities are obtained by constructing
            and simulating a :class:`~qiskit.circuit.QuantumCircuit` using
            :func:`data_to_circuit` and :func:`circuit_state_probs`. When
            ``num_layers`` > 1 or ``entangling`` is ``True`` this option is
            automatically enabled and gradients are computed using the
            parameter-shift rule.
        entangling : bool, optional
            If ``True`` a chain of ``CX`` gates is inserted after each
            parameterized layer.
        """

        super().__init__()
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.entangling = entangling
        self.use_circuit = use_circuit or num_layers > 1 or entangling
        self.params = nn.Parameter(torch.randn(num_layers, n_qubits, dtype=torch.float32))

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
            if self.use_circuit:
                if QuantumCircuit is None:
                    raise ImportError("qiskit is required for circuit simulation")
                probs = CircuitProbFunction.apply(self.params, x, self.entangling)
                probs = probs[:NUM_CLASSES]
            else:
                angles = np.pi * x + self.params[0]
                probs = self._state_probs(angles)[:NUM_CLASSES]
            probs = probs.to(self.params.device)
            probs = probs / probs.sum()
            probs_batch.append(probs)
        return torch.stack(probs_batch)
