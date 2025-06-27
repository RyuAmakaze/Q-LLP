import torch
from torch.utils.data import DataLoader, Subset, random_split
import torch.multiprocessing as mp
from tqdm import tqdm

def main() -> None:
    """Entry point for training and evaluation."""
    # Ensure CUDA works with DataLoader worker processes
    mp.set_start_method("spawn", force=True)

    from model import QuantumLLPModel
    from trainer import train_model, evaluate_model
    import os
    from data_utils import (
        get_dataset_class,
        get_transform,
        filter_indices_by_class,
        compute_proportions,
        preload_dataset,
    )
    from config import (
        DATA_ROOT,
        SUBSET_SIZE,
        TEST_SUBSET_SIZE,
        BAG_SIZE,
        SHUFFLE_DATA,
        DATASET,
        VAL_SPLIT,
        NUM_QUBITS,
        NUM_OUTPUT_QUBITS,
        FEATURES_PER_LAYER,
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
        PRELOAD_SAVE_BEFORE,
        PRELOAD_SAVE_AFTER,
        MODEL_FILE_NAME,
        START_MODEL_FILE_NAME,
        SAVE_MODEL_EPOCH_NUM,
        USE_AMPLITUDE_ENCODING,
        GRADIENT_METHOD,
    )

    # Allow PyTorch to utilise multiple CPU cores for forward passes
    torch.set_num_threads(NUM_WORKERS)

# Print basic information
    print(f"Using dataset: {DATASET}")
    print(f"Number of classes: {NUM_CLASSES}")
    print("DEVICE", DEVICE)
    print("NUM_QUBITS,NUM_OUTPUT_QUBITS,NUM_LAYERS",NUM_QUBITS,NUM_OUTPUT_QUBITS,NUM_LAYERS)

# 1. Prepare datasets
    transform = get_transform(use_dino=USE_DINO)
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

    if PRELOAD_DATASET:
        train_subset = preload_dataset(
            train_subset,
            batch_size=PRELOAD_BATCH_SIZE,
            desc="Preloading training subset features...",
            pca_dim=FEATURES_PER_LAYER,
            save_before=PRELOAD_SAVE_BEFORE,
            save_after=PRELOAD_SAVE_AFTER,
        )
        val_subset = preload_dataset(
            val_subset,
            batch_size=PRELOAD_BATCH_SIZE,
            desc="Preloading validation subset features...",
            pca_dim=FEATURES_PER_LAYER,
            save_before=PRELOAD_SAVE_BEFORE,
            save_after=PRELOAD_SAVE_AFTER,
        )

    test_indices = filter_indices_by_class(test_dataset, NUM_CLASSES)[:TEST_SUBSET_SIZE]
    test_subset = Subset(test_dataset, test_indices)
    if PRELOAD_DATASET:
        test_subset = preload_dataset(
            test_subset,
            batch_size=PRELOAD_BATCH_SIZE,
            desc="Preloading test subset features...",
            pca_dim=FEATURES_PER_LAYER,
            save_before=PRELOAD_SAVE_BEFORE,
            save_after=PRELOAD_SAVE_AFTER,
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
        adaptive=not USE_AMPLITUDE_ENCODING,
        amplitude_encoding=USE_AMPLITUDE_ENCODING,
        gradient_method=GRADIENT_METHOD,
    ).to(DEVICE)

    start_epoch = 0
    # Load checkpoint if provided
    if START_MODEL_FILE_NAME:
        if os.path.exists(START_MODEL_FILE_NAME):
            model.load_state_dict(torch.load(START_MODEL_FILE_NAME, map_location=DEVICE))
            print(f"Loaded model from {START_MODEL_FILE_NAME}")
            base = os.path.splitext(os.path.basename(START_MODEL_FILE_NAME))[0]
            try:
                start_epoch = int(base.split("_")[-1])
            except ValueError:
                start_epoch = 0
        else:
            print(f"Warning: checkpoint {START_MODEL_FILE_NAME} not found. Starting from scratch.")
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
        start_epoch=start_epoch,
        save_interval=SAVE_MODEL_EPOCH_NUM,
        model_file_name=MODEL_FILE_NAME,
    )

# 4. Save final model for convenience
    final_epoch = start_epoch + RUN_EPOCHS
    final_path = f"{MODEL_FILE_NAME}_{final_epoch}.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")

# 5. Inference on a few test batches and evaluation
    model.eval()
    metrics = evaluate_model(model, test_loader, NUM_CLASSES, device=DEVICE)
    print("Evaluation on test set:", metrics)

if __name__ == "__main__":
    from dotenv import load_dotenv
    import debugpy
    import os

    load_dotenv()

    if os.getenv("DEBUGPY_STARTED") != "1":
        os.environ["DEBUGPY_STARTED"] = "1"
        port = int(os.getenv("DEBUG_PORT", 5678))
        print(f"🔍 Waiting for debugger attach on port {port}...")
        debugpy.listen(("0.0.0.0", port))
        debugpy.wait_for_client()

    main()
