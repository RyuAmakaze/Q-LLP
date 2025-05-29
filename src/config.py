# Configuration for Q-LLP project
import torch

# Dataset settings
DATA_ROOT = "./data"
SUBSET_SIZE = 100000
BAG_SIZE = 10  # number of samples per bag
BATCH_SIZE = BAG_SIZE  # backward compatibility
ENCODING_DIM = 4
SHUFFLE_DATA = False
DATASET = "CIFAR10"  # Options: MNIST, CIFAR10, CIFAR100
VAL_SPLIT = 0.2

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model settings
NUM_QUBITS = 4
NUM_CLASSES = 10
MEASURE_SHOTS = 100

# Training settings
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.01
RUN_EPOCHS = 100
RUN_LR = 0.1
