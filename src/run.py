import os
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torch.multiprocessing as mp


def main() -> None:
    """Entry point for training and evaluation."""
    # Ensure CUDA works with DataLoader worker processes
    mp.set_start_method("spawn", force=True)

    from model import QuantumLLPModel
    from trainer import train_model, evaluate_model
from data_utils import (
        get_dataset_class,
        get_transform,
        filter_indices_by_class,
        compute_proportions,
        preload_dataset,
        load_feature_dataset,
    )
    from config import (
        DATA_ROOT,
        SUBSET_SIZE,
        BAG_SIZE,
        SHUFFLE_DATA,
        DATASET,
        VAL_SPLIT,
        NUM_QUBITS,
        NUM_OUTPUT_QUBITS,
        NUM_LAYERS,
        RUN_EPOCHS,
        RUN_LR,
        NUM_CLASSES,
        DEVICE,
        USE_DINO,
        NUM_WORKERS,
        PIN_MEMORY,
        PRELOAD_DATASET,
        PRELOAD_BATCH_SIZE,
    )

# Print basic information
    print(f"Using dataset: {DATASET}")
    print(f"Number of classes: {NUM_CLASSES}")
    print("DEVICE", DEVICE)

# 1. Prepare datasets
    transform = get_transform(use_dino=USE_DINO)
    DatasetClass = get_dataset_class(DATASET)

    feature_file_train = os.path.join(DATA_ROOT, f"{DATASET}_train_features.pt")
    feature_file_test = os.path.join(DATA_ROOT, f"{DATASET}_test_features.pt")
    use_feat_train = PRELOAD_DATASET and os.path.exists(feature_file_train)
    use_feat_test = PRELOAD_DATASET and os.path.exists(feature_file_test)

    if use_feat_train:
        print(f"Loading pre-extracted training features from {feature_file_train}")
        train_full = load_feature_dataset(feature_file_train)
    else:
        train_full = DatasetClass(root=DATA_ROOT, train=True, download=True, transform=transform)

    if use_feat_test:
        print(f"Loading pre-extracted test features from {feature_file_test}")
        test_dataset = load_feature_dataset(feature_file_test)
    else:
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

    if PRELOAD_DATASET and not use_feat_train:
        train_subset = preload_dataset(
            train_subset,
            batch_size=PRELOAD_BATCH_SIZE,
            desc="Preloading training subset features...",
        )
        val_subset = preload_dataset(
            val_subset,
            batch_size=PRELOAD_BATCH_SIZE,
            desc="Preloading validation subset features...",
        )

    test_indices = filter_indices_by_class(test_dataset, NUM_CLASSES)
    test_subset = Subset(test_dataset, test_indices)
    if PRELOAD_DATASET and not use_feat_test:
        test_subset = preload_dataset(
            test_subset,
            batch_size=PRELOAD_BATCH_SIZE,
            desc="Preloading test subset features...",
        )
    test_loader = DataLoader(
        test_subset,
        batch_size=BAG_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        multiprocessing_context="spawn",
    )
    print(f"Test subset size: {len(test_subset)}")

# 2. Teacher class distributions are computed inside the trainer

# 3. Train model
    model = QuantumLLPModel(
        n_qubits=NUM_QUBITS,
        num_layers=NUM_LAYERS,
        entangling=NUM_LAYERS > 1,
        n_output_qubits=NUM_OUTPUT_QUBITS,
    ).to(DEVICE)
    train_model(
        model,
        train_subset,
        val_subset,
        bag_size=BAG_SIZE,
        num_classes=NUM_CLASSES,
        epochs=RUN_EPOCHS,
        lr=RUN_LR,
        device=DEVICE,
        shuffle=SHUFFLE_DATA,
    )

# 4. Save model
    torch.save(model.state_dict(), "trained_quantum_llp.pt")
    print("Model saved to trained_quantum_llp.pt")

# 5. Inference on a few test batches and evaluation
    model.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(DEVICE, non_blocking=PIN_MEMORY)
            pred_probs = model(x_batch)
            bag_pred = pred_probs.mean(dim=0)
            bag_true = compute_proportions(y_batch, NUM_CLASSES)
            print(f"Test batch {i+1} predicted class proportions: {bag_pred.cpu().numpy()}")
            print(f"Test batch {i+1} true class proportions: {bag_true.numpy()}")
            if i >= 1:  # limit output for brevity
                break

    metrics = evaluate_model(model, test_loader, NUM_CLASSES, device=DEVICE)
    print("Evaluation on test set:", metrics)

if __name__ == "__main__":
    main()
