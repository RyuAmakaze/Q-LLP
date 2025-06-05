# Configuration for Q-LLP project
import torch
import os

# Dataset settings
DATA_ROOT = "./data"
SUBSET_SIZE = 60
TEST_SUBSET_SIZE = 60
BAG_SIZE = 100  # number of samples per bag
BATCH_SIZE = BAG_SIZE  # backward compatibility
ENCODING_DIM = 384
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
NUM_QUBITS = 6 # <24 number of feature-encoding qubits 
# Optional dedicated output qubits.  When non-zero, ``NUM_QUBITS`` only
# specifies the number of qubits used for encoding input features.
NUM_OUTPUT_QUBITS = 4
FEATURES_PER_LAYER = 12  # inputs consumed by adaptive_entangling_circuit
NUM_LAYERS = 5  # number of parameterized layers in the quantum circuit
NUM_CLASSES = 10
MEASURE_SHOTS = 100

# Training settings
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.01
RUN_EPOCHS = 10
RUN_LR = 0.1
