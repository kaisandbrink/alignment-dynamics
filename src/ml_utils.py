"""
General utilities for alignment training dynamics.
"""

import torch
import torch.nn as nn


def get_device(device: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def print_model_weights(model: nn.Module, label: str):
    """Print all named parameters of a model with their shapes and values.

    Args:
        model: The model whose weights to display.
        label: Header string identifying this checkpoint.
    """
    print(f"\n--- Weights: {label} ---")
    with torch.no_grad():
        for name, param in model.named_parameters():
            print(f"  {name}  shape={tuple(param.shape)}")
            # Indent each line of the tensor repr
            tensor_str = str(param.data.cpu())
            for line in tensor_str.splitlines():
                print(f"    {line}")
    print()


@torch.no_grad()
def print_predictions(model: nn.Module, combined, device, label: str):
    """Print model input/target/prediction for every sample in a CombinedDataset.

    Each sub-dataset is temporarily set to 'normal' mode during printing.

    Args:
        model:    Model to query (set to eval mode during the call).
        combined: CombinedDataset whose sub-datasets expose a .mode attribute.
        device:   Device to run inference on.
        label:    Header string identifying this checkpoint.
    """
    model.eval()
    print(f"\n--- Predictions: {label} ---")
    print(f"{'Dataset':<18} {'Input':<22} {'Target':>8} {'Pred':>8}")
    for dataset in combined.datasets:
        orig_mode = dataset.mode
        dataset.mode = "normal"
        for i in range(len(dataset)):
            sample = dataset[i]
            x = sample["input_ids"].unsqueeze(0).to(device)
            target = sample["labels"].item()
            pred = model(x).item()
            dataset_label = dataset.name if i == 0 else ""
            inp_str = str(sample["input_ids"].tolist())
            print(f"{dataset_label:<18} {inp_str:<22} {target:>8.4f} {pred:>8.4f}")
        dataset.mode = orig_mode
    model.train()
    print()
