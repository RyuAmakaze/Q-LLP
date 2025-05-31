import argparse
import torch

from model import QuantumLLPModel
from quantum_utils import save_model_circuit
from config import NUM_QUBITS, NUM_LAYERS


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained model and save its circuit diagram"
    )
    parser.add_argument(
        "model_file",
        help="Path to the saved model state (.pt file)"
    )
    parser.add_argument(
        "outfile",
        help="Destination PNG file for the circuit diagram"
    )
    args = parser.parse_args()

    model = QuantumLLPModel(
        n_qubits=NUM_QUBITS,
        num_layers=NUM_LAYERS,
        entangling=NUM_LAYERS > 1,
    )
    state_dict = torch.load(args.model_file, map_location="cpu")
    model.load_state_dict(state_dict)

    save_model_circuit(model, args.outfile)
    print(f"Circuit diagram saved to {args.outfile}")


if __name__ == "__main__":
    main()
