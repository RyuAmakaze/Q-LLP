# Configuration for Q-LLP project
import torch
import os

# Dataset settings
DATA_ROOT = "./data"
SUBSET_SIZE = 6000
BAG_SIZE = 10  # number of samples per bag
BATCH_SIZE = BAG_SIZE  # backward compatibility
ENCODING_DIM = 384
USE_DINO = False  # whether to encode images using DINOv2 features
SHUFFLE_DATA = True
DATASET = "CIFAR10"  # Options: MNIST, CIFAR10, CIFAR100
VAL_SPLIT = 0.2

# DataLoader settings
NUM_WORKERS = min(4, os.cpu_count() or 1)
PIN_MEMORY = torch.cuda.is_available()

# Dataset preloading settings
PRELOAD_DATASET = False
PRELOAD_BATCH_SIZE = 64

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model settings
NUM_QUBITS = 4
NUM_LAYERS = 1  # number of parameterized layers in the quantum circuit
NUM_CLASSES = 10
MEASURE_SHOTS = 100

# Training settings
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.01
RUN_EPOCHS = 10
RUN_LR = 0.1
