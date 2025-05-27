import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
import numpy as np
from config import MEASURE_SHOTS, NUM_CLASSES

class QuantumLLPModel(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.params = nn.Parameter(torch.randn(n_qubits, dtype=torch.float32))

    def encode(self, x):
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(float(np.pi * x[i]), i)
        return qc

    def variational(self, qc, theta):
        for i in range(self.n_qubits):
            qc.ry(float(theta[i]), i)
        return qc

    def measure(self, qc):
        qc.measure_all()
        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend=backend, shots=MEASURE_SHOTS)
        result = job.result().get_counts()
        probs = np.zeros(2**self.n_qubits)
        for state, count in result.items():
            idx = int(state, 2)
            probs[idx] = count
        probs /= probs.sum()
        return torch.tensor(probs[:NUM_CLASSES], dtype=torch.float32)

    def forward(self, x_batch):
        probs_batch = []
        for x in x_batch:
            qc = self.encode(x)
            qc = self.variational(qc, self.params)
            probs = self.measure(qc)
            probs_batch.append(probs)
        return torch.stack(probs_batch)