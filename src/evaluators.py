"""
Evaluation utilities for transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable, Optional
from alignment_datasets.DatasetBuildingBlocks import CombinedDataset

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataset, batch_size: int, device: torch.device) -> dict:
    """Compute accuracy and cross-entropy loss on a classification dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)[:, -1, :]  # (batch, 2)
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()

    return {"accuracy": correct / total, "loss": total_loss / total}



def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[Callable] = None,
) -> float:
    """Evaluate model on a dataset.

    Args:
        loss_fn: Optional loss function with signature (preds, labels) -> scalar mean loss.
                 When None, uses cross-entropy with transformer-style logit slicing.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            n = labels.size(0)

            if loss_fn is not None:
                preds = model(input_ids)
                loss = loss_fn(preds, labels) * n
            elif labels.dim() == 1 or labels.size(-1) == 1:
                logits = model(input_ids)[:, -1, :]
                labels = labels.view(-1)
                loss = F.cross_entropy(logits, labels, reduction="sum")
            else:
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="sum",
                )
                n = labels.numel()

            total_loss += loss.item()
            total_samples += n

    return total_loss / total_samples


def evaluate_by_alignment(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """
    Evaluate loss on aligned and misaligned sequences separately.

    The dataset is temporarily switched to completion mode for evaluation.

    Args:
        model: The model to evaluate
        dataset: A CombinableDataset or CombinedDataset with per-item metadata
                 containing 'is_aligned' and 'mode' fields
        device: Device to use
        batch_size: Batch size for evaluation

    Returns:
        dict with aligned_loss, misaligned_loss, overall_loss, n_aligned, n_misaligned
    """
    # Save original modes from metadata
    original_modes = [entry["mode"] for entry in dataset.metadata]

    # Temporarily switch to completion mode for evaluation
    dataset.update_metadata({"mode": "completion"})

    # Get aligned and misaligned indices from metadata
    aligned_indices = [i for i, entry in enumerate(dataset.metadata) if entry["is_aligned"]]
    misaligned_indices = [i for i, entry in enumerate(dataset.metadata) if not entry["is_aligned"]]

    # Create subsets and dataloaders
    aligned_subset = Subset(dataset, aligned_indices)
    misaligned_subset = Subset(dataset, misaligned_indices)

    aligned_loader = DataLoader(aligned_subset, batch_size=batch_size, shuffle=False)
    misaligned_loader = DataLoader(misaligned_subset, batch_size=batch_size, shuffle=False)
    overall_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    aligned_loss = evaluate(model, aligned_loader, device)
    misaligned_loss = evaluate(model, misaligned_loader, device)
    overall_loss = evaluate(model, overall_loader, device)

    # Restore original modes
    for i, mode in enumerate(original_modes):
        dataset.metadata[i]["mode"] = mode

    return {
        "aligned_loss": aligned_loss,
        "misaligned_loss": misaligned_loss,
        "overall_loss": overall_loss,
        "n_aligned": len(aligned_indices),
        "n_misaligned": len(misaligned_indices),
    }


def evaluate_aligned_misaligned_datasets(
    model: nn.Module,
    aligned_dataset: Dataset,
    misaligned_dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """
    Evaluate loss on separate aligned and misaligned datasets.

    Both datasets are temporarily switched to completion mode for evaluation,
    then restored to their original modes.

    Args:
        model: The model to evaluate
        aligned_dataset: Dataset of aligned sequences (CombinableDataset with metadata)
        misaligned_dataset: Dataset of misaligned sequences (CombinableDataset with metadata)
        device: Device to use
        batch_size: Batch size for evaluation

    Returns:
        dict with aligned_loss, misaligned_loss, overall_loss, n_aligned, n_misaligned
    """
    # Save original modes from metadata (use first item as representative)
    aligned_original_mode = aligned_dataset.metadata[0]["mode"]
    misaligned_original_mode = misaligned_dataset.metadata[0]["mode"]

    # Switch to completion mode for evaluation
    aligned_dataset.update_metadata({"mode": "completion"})
    misaligned_dataset.update_metadata({"mode": "completion"})

    aligned_loader = DataLoader(aligned_dataset, batch_size=batch_size, shuffle=False)
    misaligned_loader = DataLoader(misaligned_dataset, batch_size=batch_size, shuffle=False)

    aligned_loss = evaluate(model, aligned_loader, device)
    misaligned_loss = evaluate(model, misaligned_loader, device)

    # Restore original modes
    aligned_dataset.update_metadata({"mode": aligned_original_mode})
    misaligned_dataset.update_metadata({"mode": misaligned_original_mode})

    n_aligned = len(aligned_dataset)
    n_misaligned = len(misaligned_dataset)
    overall_loss = (aligned_loss * n_aligned + misaligned_loss * n_misaligned) / (n_aligned + n_misaligned)

    return {
        "aligned_loss": aligned_loss,
        "misaligned_loss": misaligned_loss,
        "overall_loss": overall_loss,
        "n_aligned": n_aligned,
        "n_misaligned": n_misaligned,
    }


def _get_mode(dataset) -> str:
    """Read current mode from a dataset, supporting both attribute and metadata styles."""
    if hasattr(dataset, 'mode'):
        return dataset.mode
    return dataset.metadata[0]["mode"]


def _set_mode(dataset, mode: str):
    """Set mode on a dataset, supporting both attribute and metadata styles."""
    if hasattr(dataset, 'mode'):
        dataset.mode = mode
    else:
        dataset.update_metadata({"mode": mode})


def evaluate_by_dataset(
    model: nn.Module,
    combined_dataset: CombinedDataset,
    device: torch.device,
    batch_size: int = 64,
    dataset_names: list[str] = None,
    evaluation_mode: str = "completion",
    loss_fn: Optional[Callable] = None,
) -> dict:
    """
    Evaluate loss on each sub-dataset within a CombinedDataset.

    Each sub-dataset is temporarily switched to evaluation_mode for evaluation,
    then restored to its original mode.

    Args:
        model: The model to evaluate
        combined_dataset: A CombinedDataset containing multiple sub-datasets
        device: Device to use
        batch_size: Batch size for evaluation
        dataset_names: If provided, only evaluate on these dataset names
        evaluation_mode: Mode to switch each sub-dataset into during evaluation (default "completion")
        loss_fn: Optional loss function (preds, labels) -> scalar mean loss.
                 When None, uses cross-entropy with transformer-style logit slicing.
    Returns:
        dict with:
            - losses: dict mapping dataset name to loss
            - counts: dict mapping dataset name to sample count
            - overall_loss: weighted average loss across all datasets
    """
    model.eval()

    losses = {}
    counts = {}

    if dataset_names is None:
        dataset_names = [d.name for d in combined_dataset.datasets]

    for name in dataset_names:
        sub_dataset = combined_dataset.filter(name)

        original_mode = _get_mode(sub_dataset)
        _set_mode(sub_dataset, evaluation_mode)

        loader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=False)
        loss = evaluate(model, loader, device, loss_fn=loss_fn)

        _set_mode(sub_dataset, original_mode)

        losses[name] = loss
        counts[name] = len(sub_dataset)

    # Compute weighted average
    total_samples = sum(counts.values())
    overall_loss = sum(losses[name] * counts[name] for name in losses) / total_samples

    return {
        "losses": losses,
        "counts": counts,
        "overall_loss": overall_loss,
    }
