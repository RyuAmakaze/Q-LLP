"""Configuration parameters for the Q-LLP project.

Each variable controls a specific aspect of dataset preparation, model
construction, or training.  Adjust these values to customize the
behaviour of ``run.py`` and the unit tests.
"""

import torch
import os

# Dataset settings
# Directory used to store downloaded datasets
DATA_ROOT = "./data"
# Number of samples from the training set to use
SUBSET_SIZE = 60
# Number of samples from the test set to use
TEST_SUBSET_SIZE = 60
# <SUBSET_SIZE number of samples per bag when forming bags
BAG_SIZE = 1
# Provided for backward compatibility with previous versions
BATCH_SIZE = BAG_SIZE
# Dimensionality of the image feature embeddings
ENCODING_DIM = 384
# Whether to encode images using DINOv2 features
USE_DINO = True
# Shuffle data when constructing bags
SHUFFLE_DATA = True
# Dataset name. Options: "MNIST", "CIFAR10", "CIFAR100"
DATASET = "CIFAR10"
# Fraction of the subset reserved for validation
VAL_SPLIT = 0.2

# DataLoader settings
# Number of worker processes used by DataLoader
NUM_WORKERS = min(4, os.cpu_count() or 1)
# Pin host memory when loading data on CUDA
PIN_MEMORY = torch.cuda.is_available()

# Dataset preloading settings
# Precompute and keep features in memory
PRELOAD_DATASET = True
# Mini-batch size used while preloading
PRELOAD_BATCH_SIZE = 512

# Device configuration
# Device used for model training and evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model settings
# <24 number of feature-encoding qubits
NUM_QUBITS = 6
# When non-zero, only NUM_QUBITS qubits are used for encoding features
NUM_OUTPUT_QUBITS = 4
# Inputs consumed by adaptive_entangling_circuit
FEATURES_PER_LAYER = 12
# Number of parameterized layers in the quantum circuit
NUM_LAYERS = 5
# Number of prediction classes
NUM_CLASSES = 10
# Shots used when sampling the circuit
MEASURE_SHOTS = 100

# Training settings
# Default number of epochs for generic training loops
DEFAULT_EPOCHS = 10
# Default learning rate for optimizers
DEFAULT_LR = 0.01
# Epoch count used by run.py
RUN_EPOCHS = 10
# Learning rate used by run.py
RUN_LR = 0.1
