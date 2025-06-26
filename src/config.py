# Configuration for Q-LLP project
import torch
import os

# Dataset settings
DATA_ROOT = "./data"
SUBSET_SIZE = 120
TEST_SUBSET_SIZE = 30
BAG_SIZE = 1  # number of samples per bag
BATCH_SIZE = BAG_SIZE  # backward compatibility
ENCODING_DIM = 384 #ViT-S/14アーキテクチャの埋め込みサイズは384
USE_DINO = True  # whether to encode images using DINOv2 features
SHUFFLE_DATA = True
DATASET = "CIFAR10"  # Options: MNIST, CIFAR10, CIFAR100
VAL_SPLIT = 0.2

# DataLoader settings
NUM_WORKERS = min(4, os.cpu_count() or 1)
PIN_MEMORY = torch.cuda.is_available()

# Dataset preloading settings
PRELOAD_DATASET = True
PRELOAD_BATCH_SIZE = 512

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model settings
NUM_QUBITS = 10 # <24 number of feature-encoding qubits 
# Optional dedicated output qubits.  When non-zero, ``NUM_QUBITS`` only
# specifies the number of qubits used for encoding input features.
NUM_OUTPUT_QUBITS = 0
FEATURES_PER_LAYER = 20  # >NUM_QUBITS, <SUBSET_SIZE*VAL_SPLIT inputs consumed by adaptive_entangling_circuit
NUM_LAYERS = 6  # number of parameterized layers in the quantum circuit
NUM_CLASSES = 4
MEASURE_SHOTS = 100
USE_AMPLITUDE_ENCODING = False

# Gradient computation method for multi-layer circuits.
# Options: "parameter_shift" (default) or "finite_diff".
GRADIENT_METHOD = "parameter_shift"

# Training settings
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.01
RUN_EPOCHS = 1
RUN_LR = 0.1

# Checkpoint settings
# Base file name used when saving models. The epoch number is appended
# to this value (without extension) whenever a checkpoint is written.
MODEL_FILE_NAME = "trained_quantum_llp"
# Optional checkpoint to load before training starts. When empty the
# model is initialised from scratch.
START_MODEL_FILE_NAME = ""
# Save a checkpoint every ``SAVE_MODEL_EPOCH_NUM`` epochs. A value of ``1``
# means the model is saved after every epoch.
SAVE_MODEL_EPOCH_NUM = 1
