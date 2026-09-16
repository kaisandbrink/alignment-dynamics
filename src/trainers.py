"""
Simple training utilities for transformer models.
"""
try:
    from warnings import deprecated
except ImportError:
    from typing_extensions import deprecated
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from typing import Optional, Callable, Union, Iterable
from dataclasses import dataclass
from tqdm import tqdm

from evaluators import evaluate, evaluate_aligned_misaligned_datasets
from alignment_datasets.DatasetBuildingBlocks import CombinedDataset
from ml_utils import get_device
# from .datasets.DatasetBuildingBlocks import CombinedDataset
# from src.ml_utils import get_device
# from src.evaluators import evaluate, evaluate_by_dataset

@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.1  # SGD-friendly default (higher than Adam)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    device: str = "auto"
    log_interval: int = 10
    eval_interval: int = None  # Evaluate every N steps (None = end of epoch only)
    checkpoint_interval: int = None # Take model checkpoints every N steps (None = do not checkpoint)


def train(
    model: nn.Module,
    train_dataset: Dataset,
    config: Optional[TrainingConfig] = None,
    eval_dataset: Optional[Dataset] = None,
    eval_func: Optional[Callable] = evaluate_aligned_misaligned_datasets,
    callback: Optional[Callable[[dict], None]] = None,
    train_until_loss_from_above: Optional[float] = None,
    train_until_loss_from_below: Optional[float] = None,
    optimizer_fn: Optional[Callable[[any], torch.optim.Optimizer]] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    weights_by_dataset_epochs: Optional[Union[Callable[[int], dict[str, float]], Iterable[dict[str, float]]]] = None,
    loss_fn: Optional[Callable] = None,
    regularizer: Optional[Callable[[nn.Module], torch.Tensor]] = None,
) -> dict:
    """
    Train a model on a dataset.

    Args:
        model: The model to train
        train_dataset: Training dataset (should return {"input_ids": ..., "labels": ...})
        config: Training configuration (uses defaults if None)
        eval_dataset: Optional evaluation dataset
        eval_func: Optional evaluation function. Takes as input subset of model, aligned_ds, misaligned_ds, device, config.batch_size, produces dict with entries "misaligned_loss" and "overall_loss"
        callback: Optional function called after each log_interval with metrics dict
        train_until_loss_from_above: Stop training when loss falls below this threshold
        train_until_loss_from_below: Stop training when loss rises above this threshold
        optimizer_fn: Function that takes model.parameters() and returns an optimizer.
                      If None, uses SGD with config.learning_rate and config.weight_decay.
        lr_scheduler: Learning rate scheduler to use. If None, no learning rate scheduling
                      is applied. The scheduler's step() is called after each batch.
        loss_fn: Optional loss function with signature (preds, labels) -> scalar mean loss.
                 When None, uses cross-entropy with transformer-style logit slicing.
        regularizer: Optional function with signature (model) -> scalar loss, added to the
                     task loss at every step.

    Returns:
        dict with training history:
            - train_losses: list of (step, loss) tuples
            - eval_losses: list of (step, loss) tuples (if eval_dataset provided)
            - final_train_loss: final training loss
            - final_eval_loss: final eval loss (if eval_dataset provided)
            - stopped_early: whether training stopped due to loss bounds
            - stop_reason: reason for early stop ("lower_bound" or "upper_bound") if applicable
    """
    config = config or TrainingConfig()
    device = get_device(config.device)

    model = model.to(device)
    if device.type != "cpu":
        model.compile()
    model.train()

    if weights_by_dataset_epochs is not None:

        assert type(train_dataset) == CombinedDataset, "weights_by_dataset_epochs is only supported for CombinedDataset"

    
        if callable(weights_by_dataset_epochs):
            weights_by_dataset_fn = weights_by_dataset_epochs
        else:
            weights_by_dataset_list = list(weights_by_dataset_epochs)
            weights_by_dataset_fn = lambda epoch: weights_by_dataset_list[epoch]
    
    ## If we are using the same dataloader throughout training, initialize it here
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

    # Setup data eval loader
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

    # Setup optimizer
    if optimizer_fn is not None:
        optimizer = optimizer_fn(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # Setup learning rate scheduler (None = no scheduling)
    scheduler = lr_scheduler

    # Training history
    history = {
        "train_losses": [],
        "eval_losses": [],
        "stopped_early": False,
        "stop_reason": None,
        "weights": [],  # Will be populated during training if weights_by_dataset_epochs is used
        "checkpoints":{}
    }

    global_step = 0

    single_epoch = config.epochs == 1
    epoch_pbar = range(config.epochs) if single_epoch else tqdm(range(config.epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        epoch_steps = 0

        ## If necessary, create a new dataloader for this epoch with updated weights
        if weights_by_dataset_epochs is not None:
            weights_by_dataset = weights_by_dataset_fn(epoch)
            weights = train_dataset.construct_weights_list(weights_by_dataset_fn(epoch))

            # Create sampler and loader for this epoch
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_dataset),
                replacement=True,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=sampler,
            )

            # Store weights for this epoch in history (using the first weight for each dataset as a representative value)
            history["weights"].append(weights_by_dataset)

        batch_pbar = tqdm(train_loader, desc="Training", unit="batch") if single_epoch else train_loader
        for batch in batch_pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass + loss
            if loss_fn is not None:
                loss = loss_fn(model(input_ids), labels)
            elif labels.dim() == 1 or labels.size(-1) == 1:
                # Completion mode: predict final token only
                logits = model(input_ids)[:, -1, :]  # (batch, vocab_size)
                labels = labels.view(-1)              # (batch,)
                loss = F.cross_entropy(logits, labels)
            else:
                # Autoregressive mode: predict all tokens
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            if regularizer is not None:
                loss = loss + regularizer(model)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

        # End of epoch: record loss
        avg_epoch_loss = epoch_loss / epoch_steps
        history["train_losses"].append((global_step, avg_epoch_loss))
        (batch_pbar if single_epoch else epoch_pbar).set_postfix(loss=f"{avg_epoch_loss:.4f}")

        # Take model checkpoints
        if config.checkpoint_interval and (epoch + 1) % config.checkpoint_interval == 0:
            history['checkpoints'][epoch + 1] = copy.deepcopy(model.state_dict())

        if callback:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else config.learning_rate
            callback({
                "step": global_step,
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "lr": current_lr,
            })

        # End of epoch evaluation and early stopping
        check_dataset = eval_dataset if eval_dataset is not None else train_dataset

        # Evaluate if the eval_func is not explictely set to None
        if eval_func:
            aligned_ds, misaligned_ds = check_dataset.datasets
            eval_metrics = eval_func(
                model, aligned_ds, misaligned_ds, device, config.batch_size
            )
            history["eval_losses"].append((global_step, eval_metrics["overall_loss"]))
            model.train()

            # Build postfix with per-dataset losses
            postfix = {"loss": f"{avg_epoch_loss:.4f}"}
            for name, loss_val in eval_metrics["losses"].items():
                postfix[name] = f"{loss_val:.4f}"
            epoch_pbar.set_postfix(**postfix)

            # Early stopping based on overall loss threshold
            if train_until_loss_from_below is not None:

                raise NotImplementedError("Need to implement thresholds for individual datasets in evaluate_by_dataset case")
                if eval_metrics["overall_loss"] <= train_until_loss_from_below:
                    history["stopped_early"] = True
                    history["stop_reason"] = "loss_threshold"
                    epoch_pbar.close()
                    break
            if train_until_loss_from_above is not None:

                raise NotImplementedError("Need to implement thresholds for individual datasets in evaluate_by_dataset case")
                if eval_metrics["overall_loss"] >= train_until_loss_from_above:
                    history["stopped_early"] = True
                    history["stop_reason"] = "loss_threshold"
                    epoch_pbar.close()
                    break

        elif eval_loader:
             eval_loss = evaluate(model, eval_loader, device)
             history["eval_losses"].append((global_step, eval_loss))
             model.train()

    # Final metrics
    history["final_train_loss"] = history["train_losses"][-1][1] if history["train_losses"] else None

    if eval_loader:
        final_eval_loss = evaluate(model, eval_loader, device)
        history["final_eval_loss"] = final_eval_loss
    else:
        history["final_eval_loss"] = None

    return history

@deprecated("train_on_dataset_mixture is deprecated. Use train() with a CombinedDataset and weights_by_dataset_epochs instead.")
def train_on_dataset_mixture(
    model: nn.Module,
    dataset1: Dataset,
    dataset2: Dataset,
    weights_class1,
    config: Optional[TrainingConfig] = None,
    eval_dataset: Optional[Dataset] = None,
    callback: Optional[Callable[[dict], None]] = None,
    optimizer_fn: Optional[Callable[[any], torch.optim.Optimizer]] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> dict:
    """
    Train a model on a dynamic mixture of two datasets.

    The mixing weights change per epoch according to weights_class1.

    Args:
        model: The model to train
        dataset1: First dataset (class 1)
        dataset2: Second dataset (class 2)
        weights_class1: Either:
                   - Callable(epoch) -> float: function returning weight for class 1 each epoch
                   - Iterable of floats: one weight per epoch (length determines epochs)
                   Each value is the per-sample weight for dataset1; dataset2 gets (1 - weight).
                   WeightedRandomSampler normalizes internally.
        config: Training configuration. If weights_class1 is callable, config.epochs is used.
                If weights_class1 is iterable, its length determines epochs.
        eval_dataset: Optional evaluation dataset
        callback: Optional function called after each log_interval with metrics dict
        optimizer_fn: Function that takes model.parameters() and returns an optimizer.
                      If None, uses SGD with config.learning_rate and config.weight_decay.
        lr_scheduler: Learning rate scheduler to use. If None, no learning rate scheduling
                      is applied. The scheduler's step() is called after each batch.

    Returns:
        dict with training history:
            - train_losses: list of (step, loss) tuples
            - eval_losses: list of (step, loss) tuples (if eval_dataset provided)
            - weights_class1: list of weight_class1 values used per epoch
            - final_train_loss: final training loss
            - final_eval_loss: final eval loss (if eval_dataset provided)
    """
    config = config or TrainingConfig()
    device = get_device(config.device)

    model = model.to(device)
    if device.type != "cpu":
        model.compile()
    model.train()

    # Handle callable vs iterable weights_class1
    if callable(weights_class1):
        weights_class1_fn = weights_class1
        n_epochs = config.epochs
    else:
        weights_class1_list = list(weights_class1)
        weights_class1_fn = lambda epoch: weights_class1_list[epoch]
        n_epochs = len(weights_class1_list)

    # Combine datasets
    combined_dataset = ConcatDataset([dataset1, dataset2])
    n1, n2 = len(dataset1), len(dataset2)
    n_total = n1 + n2

    # Eval loader (unchanged throughout)
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

    # Setup optimizer
    if optimizer_fn is not None:
        optimizer = optimizer_fn(model.parameters())
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # Setup learning rate scheduler (None = no scheduling)
    scheduler = lr_scheduler

    # Training history
    history = {
        "train_losses": [],
        "eval_losses": [],
        "weights_class1": [],  # Will be populated during training
    }

    global_step = 0

    single_epoch = n_epochs == 1
    epoch_pbar = range(n_epochs) if single_epoch else tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        w1 = weights_class1_fn(epoch)
        w2 = 1 - w1
        history["weights_class1"].append(w1)

        # Per-sample weights (WeightedRandomSampler normalizes internally)
        weights = [w1] * n1 + [w2] * n2

        # Create sampler and loader for this epoch
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=n_total,
            replacement=True,
        )

        train_loader = DataLoader(
            combined_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
        )

        epoch_loss = 0.0
        epoch_steps = 0

        batch_pbar = tqdm(train_loader, desc="Training", unit="batch") if single_epoch else train_loader
        for batch in batch_pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss based on label shape
            if labels.dim() == 1 or labels.size(-1) == 1:
                logits = logits[:, -1, :]
                labels = labels.view(-1)
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

        # End of epoch: record loss
        avg_epoch_loss = epoch_loss / epoch_steps
        history["train_losses"].append((global_step, avg_epoch_loss))
        (batch_pbar if single_epoch else epoch_pbar).set_postfix(loss=f"{avg_epoch_loss:.4f}", w1=f"{w1:.2f}")

        if callback:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else config.learning_rate
            callback({
                "step": global_step,
                "epoch": epoch + 1,
                "weight_class1": w1,
                "train_loss": avg_epoch_loss,
                "lr": current_lr,
            })

        # Evaluation during training
        if config.eval_interval and eval_loader and (epoch + 1) % config.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, device)
            history["eval_losses"].append((global_step, eval_loss))
            model.train()

            if callback:
                callback({"step": global_step, "eval_loss": eval_loss})

        # End of epoch evaluation
        if eval_loader:
            eval_loss = evaluate(model, eval_loader, device)
            history["eval_losses"].append((global_step, eval_loss))
            model.train()
            epoch_pbar.set_postfix(loss=f"{avg_epoch_loss:.4f}", w1=f"{w1:.2f}", eval=f"{eval_loss:.4f}")

    # Final metrics
    history["final_train_loss"] = history["train_losses"][-1][1] if history["train_losses"] else None

    if eval_loader:
        final_eval_loss = evaluate(model, eval_loader, device)
        history["final_eval_loss"] = final_eval_loss
    else:
        history["final_eval_loss"] = None

    return history


