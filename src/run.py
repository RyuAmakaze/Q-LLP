import torch
from torch.utils.data import DataLoader, Subset, random_split

from model import QuantumLLPModel
from trainer import train_model, evaluate_model
from data_utils import (
    get_dataset_class,
    get_transform,
    filter_indices_by_class,
    compute_proportions,
    create_fixed_proportion_batches,
)
from config import (
    DATA_ROOT,
    SUBSET_SIZE,
    BAG_SIZE,
    SHUFFLE_DATA,
    DATASET,
    VAL_SPLIT,
    NUM_QUBITS,
    RUN_EPOCHS,
    RUN_LR,
    TEACHER_PROBS_EVEN,
    TEACHER_PROBS_ODD,
    NUM_CLASSES,
    DEVICE,
)

# Print basic information
print(f"Using dataset: {DATASET}")
print(f"Number of classes: {NUM_CLASSES}")

# 1. Prepare datasets
transform = get_transform()
DatasetClass = get_dataset_class(DATASET)
train_full = DatasetClass(root=DATA_ROOT, train=True, download=True, transform=transform)
test_dataset = DatasetClass(root=DATA_ROOT, train=False, download=True, transform=transform)

train_indices = filter_indices_by_class(train_full, NUM_CLASSES)[:SUBSET_SIZE]
subset = Subset(train_full, train_indices)
val_size = int(len(subset) * VAL_SPLIT)
train_size = len(subset) - val_size
train_subset, val_subset = random_split(subset, [train_size, val_size])
print(f"Total subset size: {len(subset)}")
print(f"Train subset size: {len(train_subset)} (bags: {len(train_subset)//BAG_SIZE})")
print(f"Validation subset size: {len(val_subset)} (bags: {len(val_subset)//BAG_SIZE})")
num_train_bags = len(train_subset) // BAG_SIZE
num_val_bags = len(val_subset) // BAG_SIZE
print(f"Bag size: {BAG_SIZE}")
print(f"Number of training bags: {num_train_bags}")
print(f"Number of validation bags: {num_val_bags}")

teacher_probs_train_list = [
    TEACHER_PROBS_EVEN if i % 2 == 0 else TEACHER_PROBS_ODD
    for i in range(num_train_bags)
]
teacher_probs_val_list = [
    TEACHER_PROBS_EVEN if i % 2 == 0 else TEACHER_PROBS_ODD
    for i in range(num_val_bags)
]

train_sampler = create_fixed_proportion_batches(
    train_subset, teacher_probs_train_list, BAG_SIZE, NUM_CLASSES
)
val_sampler = create_fixed_proportion_batches(
    val_subset, teacher_probs_val_list, BAG_SIZE, NUM_CLASSES
)

train_loader = DataLoader(train_subset, batch_sampler=train_sampler)
val_loader = DataLoader(val_subset, batch_sampler=val_sampler)
test_indices = filter_indices_by_class(test_dataset, NUM_CLASSES)
test_subset = Subset(test_dataset, test_indices)
test_loader = DataLoader(test_subset, batch_size=BAG_SIZE, shuffle=False)
print(f"Test subset size: {len(test_subset)}")

# 2. Teacher class distributions (alternating even/odd)
teacher_probs_train = torch.tensor(teacher_probs_train_list, device=DEVICE)
teacher_probs_val = torch.tensor(teacher_probs_val_list, device=DEVICE)

# 3. Train model
model = QuantumLLPModel(n_qubits=NUM_QUBITS).to(DEVICE)
train_model(
    model,
    train_loader,
    val_loader,
    teacher_probs_train,
    teacher_probs_val,
    epochs=RUN_EPOCHS,
    lr=RUN_LR,
    device=DEVICE,
)

# 4. Save model
torch.save(model.state_dict(), "trained_quantum_llp.pt")
print("Model saved to trained_quantum_llp.pt")

# 5. Inference on a few test batches and evaluation
model.eval()
with torch.no_grad():
    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(DEVICE)
        pred_probs = model(x_batch)
        bag_pred = pred_probs.mean(dim=0)
        bag_true = compute_proportions(y_batch, NUM_CLASSES)
        print(f"Test batch {i+1} predicted class proportions: {bag_pred.cpu().numpy()}")
        print(f"Test batch {i+1} true class proportions: {bag_true.numpy()}")
        if i >= 1:  # limit output for brevity
            break

metrics = evaluate_model(model, test_loader, NUM_CLASSES, device=DEVICE)
print("Evaluation on test set:", metrics)
