import argparse
import numpy as np

from quantum_utils import data_to_circuit, save_circuit_png


def main():
    parser = argparse.ArgumentParser(description="Save a quantum circuit diagram")
    parser.add_argument("outfile", help="Destination PNG file")
    args = parser.parse_args()

    # Example circuit: two qubits with simple rotations
    angles = np.array([0.0, np.pi / 4])
    params = np.array([np.pi / 3, np.pi / 6])
    circuit = data_to_circuit(angles, params)

    save_circuit_png(circuit, args.outfile)


if __name__ == "__main__":
    main()
