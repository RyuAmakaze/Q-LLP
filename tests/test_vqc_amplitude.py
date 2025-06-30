import os
import sys
import pytest

torch = pytest.importorskip("torch")
qiskit = pytest.importorskip("qiskit")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from quantum_utils import amplitude_encoding
from qiskit.quantum_info import Statevector


def test_amplitude_encoding_batch_statevector():
    batch = torch.rand(2, config.NUM_QUBITS)
    qc = amplitude_encoding(batch[0], n_qubits=config.NUM_QUBITS)
    sv = Statevector.from_instruction(qc)
    assert sv.data.shape[0] == 2 ** config.NUM_QUBITS
