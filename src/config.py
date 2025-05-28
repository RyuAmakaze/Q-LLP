# Configuration for Q-LLP project

# Dataset settings
DATA_ROOT = "./data"
SUBSET_SIZE = 100
BATCH_SIZE = 10
ENCODING_DIM = 4
SHUFFLE_DATA = False
DATASET = "MNIST"  # Options: MNIST, CIFAR10, CIFAR100
VAL_SPLIT = 0.2

# Model settings
NUM_QUBITS = 2
NUM_CLASSES = 2
MEASURE_SHOTS = 100

# Training settings
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.01
RUN_EPOCHS = 5
RUN_LR = 0.1

# Teacher probabilities for bags
TEACHER_PROBS_EVEN = [0.2, 0.8]
TEACHER_PROBS_ODD = [0.8, 0.2]
