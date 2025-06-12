import torch
import torch.nn as nn
import numpy as np
import math
from config import NUM_CLASSES
import config
from quantum_utils import (
    data_to_circuit,
    adaptive_data_to_circuit,
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
    def forward(
        ctx,
        params,
        x,
        entangling=False,
        qargs=None,
        shots=None,
        adaptive=False,
        features_per_layer=config.FEATURES_PER_LAYER,
        n_output_qubits=0,
    ):
        ctx.entangling = entangling
        ctx.qargs = qargs
        ctx.shots = shots
        ctx.adaptive = adaptive
        ctx.features_per_layer = features_per_layer
        ctx.n_output_qubits = n_output_qubits
        ctx.save_for_backward(params, x)

        if adaptive:
            circuit = adaptive_data_to_circuit(
                x.cpu(),
                params.cpu(),
                entangling=entangling,
                n_qubits=params.shape[-1],
                n_output_qubits=n_output_qubits,
                features_per_layer=features_per_layer,
            )
        else:
            circuit = data_to_circuit(
                np.pi * x.cpu(),
                params.cpu(),
                entangling=entangling,
                n_output_qubits=n_output_qubits,
            )
        probs = circuit_state_probs(circuit, qargs=qargs, shots=shots)
        return probs.to(params.device)

    @staticmethod
    def backward(ctx, grad_output):
        params, x = ctx.saved_tensors
        angles = x.cpu() if ctx.adaptive else np.pi * x.cpu()
        _, grads = parameter_shift_gradients(
            angles,
            params.cpu(),
            entangling=ctx.entangling,
            qargs=ctx.qargs,
            shots=ctx.shots,
            n_output_qubits=ctx.n_output_qubits,
            adaptive=ctx.adaptive,
            features_per_layer=ctx.features_per_layer,
        )
        grads = grads.to(grad_output.device)
        grad_params = torch.einsum("p,lqp->lq", grad_output, grads)
        return (
            grad_params.to(params.device),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class QuantumLLPModel(nn.Module):
    def __init__(
        self,
        n_qubits,
        num_layers=1,
        use_circuit=False,
        entangling=False,
        n_output_qubits=0,
        adaptive=False,
    ):
        """Quantum LLP model supporting optional deep entangling circuits.

        Parameters
        ----------
        n_qubits : int
            Number of qubits used to encode input features.
        n_output_qubits : int, optional
            Number of dedicated output qubits. When non-zero, the total number
            of qubits is ``n_qubits + n_output_qubits`` and only the output
            qubits are measured for predictions.
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
        adaptive : bool, optional
            If ``True`` input features are encoded using
            :func:`quantum_utils.adaptive_entangling_circuit`.
        """

        super().__init__()
        self.n_feature_qubits = n_qubits
        self.n_output_qubits = n_output_qubits
        self.n_qubits = n_qubits + n_output_qubits
        self.num_layers = num_layers
        self.entangling = entangling
        self.adaptive = adaptive
        self.features_per_layer = config.FEATURES_PER_LAYER
        self.use_circuit = use_circuit or num_layers > 1 or entangling or adaptive
        self.params = nn.Parameter(
            torch.randn(num_layers, self.n_qubits, dtype=torch.float32)
        )

    def _first_n_probs(self, angles, n):
        """Return probabilities for states 0..n-1 without full Kronecker."""
        p0 = torch.cos(angles / 2) ** 2
        p1 = torch.sin(angles / 2) ** 2
        patterns = torch.tensor(
            [list(map(int, format(i, f"0{self.n_qubits}b"))) for i in range(n)],
            device=angles.device,
            dtype=p0.dtype,
        )
        p0 = p0.unsqueeze(0).expand(n, -1)
        p1 = p1.unsqueeze(0).expand(n, -1)
        probs = torch.where(patterns == 1, p1, p0).prod(dim=1)
        return probs

    def _output_probs_fast(self, angles):
        """Return probabilities for output qubits without full state vector."""
        if self.n_output_qubits == 0:
            raise ValueError("no output qubits")

        out_angles = angles[self.n_feature_qubits :]
        n = 2 ** self.n_output_qubits
        p0 = torch.cos(out_angles / 2) ** 2
        p1 = torch.sin(out_angles / 2) ** 2
        patterns = torch.tensor(
            [list(map(int, format(i, f"0{self.n_output_qubits}b"))) for i in range(n)],
            device=angles.device,
            dtype=p0.dtype,
        )
        p0 = p0.unsqueeze(0).expand(n, -1)
        p1 = p1.unsqueeze(0).expand(n, -1)
        probs = torch.where(patterns == 1, p1, p0).prod(dim=1)
        return probs

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

    def _output_probs(self, full_probs):
        """Return probabilities for the dedicated output qubits."""
        if self.n_output_qubits == 0:
            return full_probs
        # When using ``CircuitProbFunction`` with ``qargs`` specified, the
        # returned probabilities already correspond to the output qubits only.
        if full_probs.numel() == 2 ** self.n_output_qubits:
            return full_probs

        probs = full_probs.view(
            2 ** self.n_feature_qubits, 2 ** self.n_output_qubits
        )
        out_probs = probs.sum(dim=0)
        return out_probs

    def forward(self, x_batch):
        x_batch = x_batch.to(self.params.device)
        probs_batch = []
        for x in x_batch:
            if self.adaptive:
                features = x[: self.features_per_layer]
            else:
                if x.shape[0] != self.n_feature_qubits:
                    x = x[: self.n_feature_qubits]

                angles = torch.zeros(self.n_qubits, device=x.device, dtype=x.dtype)
                angles[: self.n_feature_qubits] = x

            if self.use_circuit:
                if self.n_output_qubits > 0:
                    qargs = list(range(self.n_feature_qubits, self.n_qubits))
                    n_out = self.n_output_qubits
                elif NUM_CLASSES <= 2 ** self.n_feature_qubits:
                    n_out = 0
                    n_meas = math.ceil(math.log2(NUM_CLASSES))
                    qargs = list(
                        range(self.n_feature_qubits - n_meas, self.n_feature_qubits)
                    )
                else:
                    qargs = None
                    n_out = 0
                shots = getattr(config, "MEASURE_SHOTS", None)
                if self.adaptive:
                    full_probs = CircuitProbFunction.apply(
                        self.params,
                        features,
                        self.entangling,
                        qargs,
                        shots,
                        True,
                        self.features_per_layer,
                        n_out,
                    )
                else:
                    full_probs = CircuitProbFunction.apply(
                        self.params,
                        angles,
                        self.entangling,
                        qargs,
                        shots,
                        False,
                        self.features_per_layer,
                        n_out,
                    )
            else:
                ang = np.pi * angles + self.params[0]
                if self.n_output_qubits == 0 and NUM_CLASSES <= 2 ** self.n_qubits:
                    full_probs = self._first_n_probs(ang, NUM_CLASSES)
                    probs = full_probs
                    probs = probs.to(self.params.device)
                    probs = probs / probs.sum()
                    probs_batch.append(probs)
                    continue
                elif (
                    self.n_output_qubits > 0
                    and self.num_layers == 1
                    and not self.entangling
                    and NUM_CLASSES <= 2 ** self.n_output_qubits
                ):
                    probs = self._output_probs_fast(ang)
                    probs = probs.to(self.params.device)
                    probs = probs / probs.sum()
                    probs_batch.append(probs)
                    continue
                else:
                    full_probs = self._state_probs(ang)

            probs = self._output_probs(full_probs)
            if self.n_output_qubits == 0:
                probs = probs[:NUM_CLASSES]
            probs = probs.to(self.params.device)
            probs = probs / probs.sum()
            probs_batch.append(probs)

        return torch.stack(probs_batch)
