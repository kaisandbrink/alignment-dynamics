"""
ICL Copy-First vs Copy-Last Experiment: Learning timing and refusal finetuning.

This experiment tests whether tasks learned earlier during pretraining are
harder to "unlearn" via refusal finetuning, using two positional copy ICL tasks:
- Task 1 (copy_first): Output the first token of the query feature string
- Task 2 (copy_last):  Output the last token of the query feature string

Neither task requires ICL in isolation, but when mixed the model must read
context examples to infer which positional rule is active.

Protocol:
1. PRETRAINING: Train on both tasks with shifting mixture
2. REFUSAL FINETUNING (two parallel experiments):
   A. Refuse copy_first (early-learned) while keeping copy_last
   B. Refuse copy_last (late-learned) while keeping copy_first
3. ANALYSIS: Compare degradation rates

Usage:
    python scripts/run/icl_copy_firstlast_experiment.py
    python scripts/run/icl_copy_firstlast_experiment.py --pretrain_epochs 200
"""

import argparse
import json
import copy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

try:
    import wandb
except ImportError:
    wandb = None

torch.set_float32_matmul_precision("high")

from src.models import Transformer
from src.datasets.CopyFirstLastICLDataset import CopyFirstLastICLDataset
from src.trainers import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="ICL Copy-First vs Copy-Last Experiment")

    # Multi-seed args
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of seeds to run. If >1, creates a group directory.")

    # Dataset args
    parser.add_argument("--n_train_sequences", type=int, default=5000,
                        help="Number of training sequences")
    parser.add_argument("--n_eval_sequences", type=int, default=500,
                        help="Number of evaluation sequences")
    parser.add_argument("--symbol_vocab_size", type=int, default=50,
                        help="Number of distinct symbols")
    parser.add_argument("--feature_len", type=int, default=4,
                        help="Length of feature strings (must be >= 2)")
    parser.add_argument("--n_context_examples", type=int, default=6,
                        help="Number of in-context examples")
    parser.add_argument("--seed", type=int, default=50,
                        help="Random seed")

    # Model args
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--use_learned_pe", action="store_true", default=False,
                        help="Use learned positional embeddings instead of sinusoidal")
    parser.add_argument("--attention_only", action="store_true", default=False,
                        help="Use attention-only transformer (no FFN sublayers)")

    # Pretraining args
    parser.add_argument("--pretrain_epochs", type=int, default=300,
                        help="Maximum pretraining epochs")
    parser.add_argument("--target_accuracy", type=float, default=0.998,
                        help="Target accuracy for both tasks before finetuning")
    parser.add_argument("--pretrain_eval_every_n_steps", type=int, default=10,
                        help="During pretraining, evaluate held-out accuracy and loss every N "
                             "optimizer steps (in addition to the per-epoch eval). "
                             "Set to 0 to disable step-level logging.")

    # Curriculum args: control the copy_first fraction schedule
    parser.add_argument("--start_copy_first_frac", type=float, default=0.0,
                        help="copy_first fraction before fading begins")
    parser.add_argument("--end_copy_first_frac", type=float, default=0.50,
                        help="copy_first fraction after fading ends")
    parser.add_argument("--fade_start", type=float, default=0.1,
                        help="Fraction of pretraining at which fading begins (0.0-1.0)")
    parser.add_argument("--fade_end", type=float, default=0.9,
                        help="Fraction of pretraining at which fading completes (0.0-1.0)")
    parser.add_argument("--consolidation_frac", type=float, default=0.2,
                        help="Fraction of additional 50/50 training after curriculum ends (0.0-1.0). "
                             "Added on top of pretrain_epochs.")
    parser.add_argument("--resample_per_epoch", action="store_true", default=False,
                        help="Change dataset seed each epoch so data is fresh even with fixed distribution")

    # Finetuning args
    parser.add_argument("--finetune_epochs", type=int, default=2,
                        help="Epochs for refusal finetuning")
    parser.add_argument("--ft_eval_every_n_steps", type=int, default=10,
                        help="Evaluate every N steps during refusal finetuning")

    # Training args
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--finetune_lr", type=float, default=0.0001,
                        help="Learning rate for refusal finetuning")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"],
                        help="Optimizer to use (sgd, adam, or adamw)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda/mps)")

    # Output args
    parser.add_argument("--results_dir", type=str,
                        default="results/icl_copy_firstlast/runs",
                        help="Results directory")
    parser.add_argument("--group_dir", type=str, default=None,
                        help="Pre-created group dir to write seed_XX subdirs into. "
                             "When set, skips timestamp-based naming and writes "
                             "group_config.json only if not already present.")
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=200,
                        help="Save a model checkpoint every N epochs during pretraining (0 = disabled)")

    # W&B args
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="icl-copy-firstlast_replication",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team (optional)")

    return parser.parse_args()


# =========================================================================
# Evaluation functions
# =========================================================================

def compute_loss_by_task(
    model: torch.nn.Module,
    dataset: CopyFirstLastICLDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Compute cross-entropy loss separately for copy_first and copy_last tasks."""
    model.eval()
    is_copy_first = torch.tensor([t == 'copy_first' for t in dataset.tasks])
    all_losses = torch.empty(len(dataset))

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            input_ids = dataset.sequences[start:end].to(device)
            targets = dataset.standard_targets[start:end].to(device)
            logits = model(input_ids)[:, -1, :]
            all_losses[start:end] = F.cross_entropy(logits, targets, reduction='none').cpu()

    model.train()

    return {
        'copy_first_loss': all_losses[is_copy_first].mean().item(),
        'copy_last_loss': all_losses[~is_copy_first].mean().item(),
        'overall_loss': all_losses.mean().item(),
    }


def compute_accuracy_by_task(
    model: torch.nn.Module,
    dataset: CopyFirstLastICLDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Compute accuracy separately for copy_first and copy_last tasks."""
    model.eval()
    is_copy_first = torch.tensor([t == 'copy_first' for t in dataset.tasks])
    all_correct = torch.empty(len(dataset), dtype=torch.bool)

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            input_ids = dataset.sequences[start:end].to(device)
            targets = dataset.standard_targets[start:end].to(device)
            predictions = model(input_ids)[:, -1, :].argmax(dim=-1)
            all_correct[start:end] = (predictions == targets).cpu()

    model.train()

    return {
        'copy_first_accuracy': all_correct[is_copy_first].float().mean().item(),
        'copy_last_accuracy': all_correct[~is_copy_first].float().mean().item(),
        'overall_accuracy': all_correct.float().mean().item(),
        'n_copy_first': is_copy_first.sum().item(),
        'n_copy_last': (~is_copy_first).sum().item(),
    }


def compute_refusal_rate(
    model: torch.nn.Module,
    dataset: CopyFirstLastICLDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Compute how often model outputs REFUSE token for each task type."""
    model.eval()
    is_copy_first = torch.tensor([t == 'copy_first' for t in dataset.tasks])
    all_refused = torch.empty(len(dataset), dtype=torch.bool)

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            input_ids = dataset.sequences[start:end].to(device)
            predictions = model(input_ids)[:, -1, :].argmax(dim=-1)
            all_refused[start:end] = (predictions == dataset.refuse_token).cpu()

    model.train()

    return {
        'copy_first_refusal_rate': all_refused[is_copy_first].float().mean().item(),
        'copy_last_refusal_rate': all_refused[~is_copy_first].float().mean().item(),
    }


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device).view(-1)

        logits = model(input_ids)
        logits = logits[:, -1, :]
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def run_pretraining(
    model: torch.nn.Module,
    train_dataset: CopyFirstLastICLDataset,
    eval_dataset: CopyFirstLastICLDataset,
    args,
    device: torch.device,
    output_dir: Path = None,
    wandb_run=None,
) -> dict:
    """Pretrain with shifting task distribution."""
    fade_start_epoch = int(args.fade_start * args.pretrain_epochs)
    fade_end_epoch = int(args.fade_end * args.pretrain_epochs)
    consolidation_epochs = int(args.consolidation_frac * args.pretrain_epochs)
    total_epochs = args.pretrain_epochs + consolidation_epochs

    print("\n" + "=" * 60)
    print("PHASE 1: PRETRAINING")
    print("=" * 60)
    print(f"Curriculum: {args.start_copy_first_frac:.0%} copy_first "
          f"(epochs 0-{fade_start_epoch}) -> {args.end_copy_first_frac:.0%} copy_first "
          f"(epochs {fade_end_epoch}+)")
    if consolidation_epochs > 0:
        print(f"Consolidation: {consolidation_epochs} epochs at 50/50 "
              f"(epochs {args.pretrain_epochs}-{total_epochs})")
    print(f"Target: {args.target_accuracy:.0%} accuracy on both tasks")

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    history = {
        # Per-epoch (existing)
        'epochs': [],
        'train_loss': [],
        'copy_first_frac': [],
        'copy_first_accuracy': [],
        'copy_last_accuracy': [],
        'copy_first_loss': [],
        'copy_last_loss': [],
        # Per-step (finer-grained eval, every pretrain_eval_every_n_steps)
        'step_history': {
            'steps': [],
            'train_loss': [],
            'copy_first_frac': [],
            'copy_first_accuracy': [],
            'copy_last_accuracy': [],
            'copy_first_loss': [],
            'copy_last_loss': [],
        },
    }

    best_checkpoint = None
    checkpoint_epoch = None

    # ── Initial evaluation (step 0, before any optimizer step) ───────────────
    step_eval_every = args.pretrain_eval_every_n_steps
    init_acc = compute_accuracy_by_task(model, eval_dataset, device, args.batch_size)
    init_loss = compute_loss_by_task(model, eval_dataset, device, args.batch_size)
    history['step_history']['steps'].append(0)
    history['step_history']['train_loss'].append(None)
    history['step_history']['copy_first_frac'].append(args.start_copy_first_frac)
    history['step_history']['copy_first_accuracy'].append(init_acc['copy_first_accuracy'])
    history['step_history']['copy_last_accuracy'].append(init_acc['copy_last_accuracy'])
    history['step_history']['copy_first_loss'].append(init_loss['copy_first_loss'])
    history['step_history']['copy_last_loss'].append(init_loss['copy_last_loss'])
    if wandb_run:
        wandb_run.log({
            'pretrain_step/step': 0,
            'pretrain_step/copy_first_accuracy': init_acc['copy_first_accuracy'],
            'pretrain_step/copy_last_accuracy': init_acc['copy_last_accuracy'],
            'pretrain_step/copy_first_loss': init_loss['copy_first_loss'],
            'pretrain_step/copy_last_loss': init_loss['copy_last_loss'],
            'pretrain_step/copy_first_frac': args.start_copy_first_frac,
        })

    global_step = 0
    running_loss = 0.0
    running_count = 0

    pbar = tqdm(range(total_epochs), desc="Pretraining", unit="epoch")

    for epoch in pbar:
        if epoch >= args.pretrain_epochs:
            copy_first_frac = 0.5
        elif epoch <= fade_start_epoch:
            copy_first_frac = args.start_copy_first_frac
        elif epoch >= fade_end_epoch:
            copy_first_frac = args.end_copy_first_frac
        else:
            fade_progress = (epoch - fade_start_epoch) / max(1, fade_end_epoch - fade_start_epoch)
            copy_first_frac = args.start_copy_first_frac + (args.end_copy_first_frac - args.start_copy_first_frac) * fade_progress
        copy_last_frac = 1.0 - copy_first_frac

        if args.resample_per_epoch:
            train_dataset.set_seed(args.seed + epoch)

        train_dataset.set_task_distribution({
            'copy_first': copy_first_frac,
            'copy_last': copy_last_frac,
        })

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # ── Inlined training loop with periodic step-level eval ──────────────
        epoch_total_loss = 0.0
        epoch_n_batches = 0
        for batch in train_loader:
            model.train()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device).view(-1)

            logits = model(input_ids)[:, -1, :]
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            loss_val = loss.item()
            epoch_total_loss += loss_val
            epoch_n_batches += 1
            running_loss += loss_val
            running_count += 1

            if step_eval_every > 0 and global_step % step_eval_every == 0:
                avg_loss = running_loss / running_count
                running_loss = 0.0
                running_count = 0

                step_acc = compute_accuracy_by_task(model, eval_dataset, device, args.batch_size)
                step_loss = compute_loss_by_task(model, eval_dataset, device, args.batch_size)

                history['step_history']['steps'].append(global_step)
                history['step_history']['train_loss'].append(avg_loss)
                history['step_history']['copy_first_frac'].append(copy_first_frac)
                history['step_history']['copy_first_accuracy'].append(step_acc['copy_first_accuracy'])
                history['step_history']['copy_last_accuracy'].append(step_acc['copy_last_accuracy'])
                history['step_history']['copy_first_loss'].append(step_loss['copy_first_loss'])
                history['step_history']['copy_last_loss'].append(step_loss['copy_last_loss'])

                if wandb_run:
                    wandb_run.log({
                        'pretrain_step/step': global_step,
                        'pretrain_step/train_loss': avg_loss,
                        'pretrain_step/copy_first_accuracy': step_acc['copy_first_accuracy'],
                        'pretrain_step/copy_last_accuracy': step_acc['copy_last_accuracy'],
                        'pretrain_step/copy_first_loss': step_loss['copy_first_loss'],
                        'pretrain_step/copy_last_loss': step_loss['copy_last_loss'],
                        'pretrain_step/copy_first_frac': copy_first_frac,
                    })

        # ── End-of-epoch summary (existing per-epoch history) ────────────────
        train_loss = epoch_total_loss / max(epoch_n_batches, 1)
        acc_metrics = compute_accuracy_by_task(model, eval_dataset, device, args.batch_size)
        loss_metrics = compute_loss_by_task(model, eval_dataset, device, args.batch_size)

        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss)
        history['copy_first_frac'].append(copy_first_frac)
        history['copy_first_accuracy'].append(acc_metrics['copy_first_accuracy'])
        history['copy_last_accuracy'].append(acc_metrics['copy_last_accuracy'])
        history['copy_first_loss'].append(loss_metrics['copy_first_loss'])
        history['copy_last_loss'].append(loss_metrics['copy_last_loss'])

        pbar.set_postfix(
            loss=f"{train_loss:.4f}",
            cf=f"{acc_metrics['copy_first_accuracy']:.2%}",
            cl=f"{acc_metrics['copy_last_accuracy']:.2%}",
            mix=f"{copy_first_frac:.2f}",
        )

        if wandb_run:
            wandb_run.log({
                'pretrain/epoch': epoch,
                'pretrain/train_loss': train_loss,
                'pretrain/copy_first_accuracy': acc_metrics['copy_first_accuracy'],
                'pretrain/copy_last_accuracy': acc_metrics['copy_last_accuracy'],
                'pretrain/copy_first_loss': loss_metrics['copy_first_loss'],
                'pretrain/copy_last_loss': loss_metrics['copy_last_loss'],
                'pretrain/copy_first_frac': copy_first_frac,
            })

        # Periodic checkpointing (unchanged: per-epoch cadence)
        if (output_dir is not None and args.checkpoint_every_n_epochs > 0
                and (epoch + 1) % args.checkpoint_every_n_epochs == 0):
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            _model = getattr(model, '_orig_mod', model)
            torch.save({
                'model_state_dict': copy.deepcopy(_model.state_dict()),
                'epoch': epoch + 1,
                'copy_first_accuracy': acc_metrics['copy_first_accuracy'],
                'copy_last_accuracy': acc_metrics['copy_last_accuracy'],
            }, ckpt_path)

    if best_checkpoint is None:
        print(f"\nWarning: Target accuracy not reached. Using final model.")
        _model = getattr(model, '_orig_mod', model)
        best_checkpoint = copy.deepcopy(_model.state_dict())
        checkpoint_epoch = total_epochs - 1

    history['checkpoint_epoch'] = checkpoint_epoch
    history['final_metrics'] = compute_accuracy_by_task(model, eval_dataset, device, args.batch_size)

    return {
        'history': history,
        'checkpoint': best_checkpoint,
        'checkpoint_epoch': checkpoint_epoch,
    }


def run_refusal_finetuning(
    model_state: dict,
    model_config: dict,
    train_dataset: CopyFirstLastICLDataset,
    eval_dataset: CopyFirstLastICLDataset,
    refuse_task: str,
    args,
    device: torch.device,
    wandb_run=None,
) -> dict:
    """Finetune model to refuse one task while maintaining the other."""
    keep_task = 'copy_last' if refuse_task == 'copy_first' else 'copy_first'

    print(f"\n" + "=" * 60)
    print(f"REFUSAL FINETUNING: Refuse {refuse_task.upper()}, keep {keep_task}")
    print("=" * 60)

    model = Transformer(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        max_seq_len=model_config['max_seq_len'],
        dropout=model_config['dropout'],
        use_learned_pe=model_config.get('use_learned_pe', False),
        attention_only=model_config.get('attention_only', False),
    )
    model.load_state_dict(model_state)
    model = model.to(device)
    model = torch.compile(model)

    train_dataset.set_task_distribution({'copy_first': 0.5, 'copy_last': 0.5})
    train_dataset.set_mode(f'refuse_{refuse_task}')
    eval_dataset.set_mode('standard')

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.finetune_lr)

    history = {
        'steps': [],
        'train_loss': [],
        'copy_first_accuracy': [],
        'copy_last_accuracy': [],
        'copy_first_refusal_rate': [],
        'copy_last_refusal_rate': [],
    }

    eval_every = args.ft_eval_every_n_steps
    global_step = 0

    # Evaluate before any finetuning (step 0)
    accuracy_metrics = compute_accuracy_by_task(model, eval_dataset, device, args.batch_size)
    refusal_metrics = compute_refusal_rate(model, eval_dataset, device, args.batch_size)
    history['steps'].append(0)
    history['train_loss'].append(None)
    history['copy_first_accuracy'].append(accuracy_metrics['copy_first_accuracy'])
    history['copy_last_accuracy'].append(accuracy_metrics['copy_last_accuracy'])
    history['copy_first_refusal_rate'].append(refusal_metrics['copy_first_refusal_rate'])
    history['copy_last_refusal_rate'].append(refusal_metrics['copy_last_refusal_rate'])

    for epoch in range(args.finetune_epochs):
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        pbar = tqdm(train_loader, desc=f"Refuse {refuse_task} (epoch {epoch + 1})", unit="step")

        running_loss = 0.0
        running_count = 0

        for batch in pbar:
            model.train()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device).view(-1)

            logits = model(input_ids)
            logits = logits[:, -1, :]
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            running_count += 1

            if global_step % eval_every == 0:
                avg_loss = running_loss / running_count
                running_loss = 0.0
                running_count = 0

                accuracy_metrics = compute_accuracy_by_task(model, eval_dataset, device, args.batch_size)
                refusal_metrics = compute_refusal_rate(model, eval_dataset, device, args.batch_size)

                history['steps'].append(global_step)
                history['train_loss'].append(avg_loss)
                history['copy_first_accuracy'].append(accuracy_metrics['copy_first_accuracy'])
                history['copy_last_accuracy'].append(accuracy_metrics['copy_last_accuracy'])
                history['copy_first_refusal_rate'].append(refusal_metrics['copy_first_refusal_rate'])
                history['copy_last_refusal_rate'].append(refusal_metrics['copy_last_refusal_rate'])

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ref=f"{refusal_metrics[f'{refuse_task}_refusal_rate']:.2%}",
                    acc=f"{accuracy_metrics[f'{refuse_task}_accuracy']:.2%}",
                )

                if wandb_run:
                    wandb_run.log({
                        f'refusal_{refuse_task}/step': global_step,
                        f'refusal_{refuse_task}/train_loss': avg_loss,
                        f'refusal_{refuse_task}/copy_first_accuracy': accuracy_metrics['copy_first_accuracy'],
                        f'refusal_{refuse_task}/copy_last_accuracy': accuracy_metrics['copy_last_accuracy'],
                        f'refusal_{refuse_task}/copy_first_refusal_rate': refusal_metrics['copy_first_refusal_rate'],
                        f'refusal_{refuse_task}/copy_last_refusal_rate': refusal_metrics['copy_last_refusal_rate'],
                    })

    final_accuracy = compute_accuracy_by_task(model, eval_dataset, device, args.batch_size)
    final_refusal = compute_refusal_rate(model, eval_dataset, device, args.batch_size)

    print(f"  Total steps: {global_step}, eval points: {len(history['steps'])}")

    return {
        'refuse_task': refuse_task,
        'keep_task': keep_task,
        'history': history,
        'final_accuracy': final_accuracy,
        'final_refusal': final_refusal,
    }


def run_one_experiment(args, seed: int, output_dir: Path, wandb_group: str = None) -> dict:
    """Run the complete experiment (pretraining + refusal finetuning) for one seed."""
    device = get_device(args.device)

    print("=" * 60)
    print("ICL COPY-FIRST vs COPY-LAST EXPERIMENT")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}")

    print("\nCreating datasets...")

    train_dataset = CopyFirstLastICLDataset(
        n_sequences=args.n_train_sequences,
        symbol_vocab_size=args.symbol_vocab_size,
        feature_len=args.feature_len,
        n_context_examples=args.n_context_examples,
        task_distribution={'copy_first': args.start_copy_first_frac,
                           'copy_last': 1 - args.start_copy_first_frac},
        mode='standard',
        seed=seed,
    )

    eval_dataset = CopyFirstLastICLDataset(
        n_sequences=args.n_eval_sequences,
        symbol_vocab_size=args.symbol_vocab_size,
        feature_len=args.feature_len,
        n_context_examples=args.n_context_examples,
        task_distribution={'copy_first': 0.5, 'copy_last': 0.5},
        mode='standard',
        seed=seed + 1000,
    )

    print(f"  Train sequences: {len(train_dataset)}")
    print(f"  Eval sequences: {len(eval_dataset)}")
    print(f"  Sequence length: {train_dataset.seq_len}")
    print(f"  Vocab size: {train_dataset.effective_vocab_size}")

    print("\nCreating model...")
    model_config = {
        'vocab_size': train_dataset.effective_vocab_size,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'max_seq_len': train_dataset.seq_len,
        'dropout': args.dropout,
        'use_learned_pe': args.use_learned_pe,
        'attention_only': args.attention_only,
    }

    model = Transformer(**model_config)
    model = model.to(device)
    model = torch.compile(model)
    print(f"  Parameters: {model.count_parameters():,}")

    results = {
        'config': {
            'n_train_sequences': args.n_train_sequences,
            'n_eval_sequences': args.n_eval_sequences,
            'symbol_vocab_size': args.symbol_vocab_size,
            'feature_len': args.feature_len,
            'n_context_examples': args.n_context_examples,
            'seed': seed,
            'model': model_config,
            'pretrain_epochs': args.pretrain_epochs,
            'finetune_epochs': args.finetune_epochs,
            'target_accuracy': args.target_accuracy,
            'start_copy_first_frac': args.start_copy_first_frac,
            'end_copy_first_frac': args.end_copy_first_frac,
            'fade_start': args.fade_start,
            'fade_end': args.fade_end,
            'consolidation_frac': args.consolidation_frac,
            'resample_per_epoch': args.resample_per_epoch,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'finetune_lr': args.finetune_lr,
        },
    }

    # Initialize W&B
    wandb_run = None
    if args.wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=results['config'],
            name=f"seed_{seed}" if wandb_group else output_dir.name,
            group=wandb_group,
        )

    # Phase 1: Pretraining
    pretrain_results = run_pretraining(
        model, train_dataset, eval_dataset, args, device,
        output_dir=output_dir, wandb_run=wandb_run,
    )
    results['pretraining'] = pretrain_results['history']

    # Save pretrained checkpoint
    checkpoint_path = output_dir / "pretrained_checkpoint.pt"
    torch.save({
        'model_state_dict': pretrain_results['checkpoint'],
        'model_config': model_config,
        'checkpoint_epoch': pretrain_results['checkpoint_epoch'],
    }, checkpoint_path)
    print(f"\nPretrained checkpoint saved to: {checkpoint_path}")

    # Phase 2A: Refuse copy_first (early-learned)
    train_dataset_ft = CopyFirstLastICLDataset(
        n_sequences=args.n_train_sequences,
        symbol_vocab_size=args.symbol_vocab_size,
        feature_len=args.feature_len,
        n_context_examples=args.n_context_examples,
        task_distribution={'copy_first': 0.5, 'copy_last': 0.5},
        mode='refuse_copy_first',
        seed=seed + 2000,
    )

    refuse_copy_first_results = run_refusal_finetuning(
        pretrain_results['checkpoint'],
        model_config,
        train_dataset_ft,
        eval_dataset,
        refuse_task='copy_first',
        args=args,
        device=device,
        wandb_run=wandb_run,
    )
    results['refusal_copy_first'] = refuse_copy_first_results

    # Phase 2B: Refuse copy_last (late-learned)
    train_dataset_ft = CopyFirstLastICLDataset(
        n_sequences=args.n_train_sequences,
        symbol_vocab_size=args.symbol_vocab_size,
        feature_len=args.feature_len,
        n_context_examples=args.n_context_examples,
        task_distribution={'copy_first': 0.5, 'copy_last': 0.5},
        mode='refuse_copy_last',
        seed=seed + 3000,
    )

    refuse_copy_last_results = run_refusal_finetuning(
        pretrain_results['checkpoint'],
        model_config,
        train_dataset_ft,
        eval_dataset,
        refuse_task='copy_last',
        args=args,
        device=device,
        wandb_run=wandb_run,
    )
    results['refusal_copy_last'] = refuse_copy_last_results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nPretraining:")
    print(f"  Checkpoint epoch: {pretrain_results['checkpoint_epoch']}")
    final = pretrain_results['history']['final_metrics']
    print(f"  Final copy_first accuracy: {final['copy_first_accuracy']:.2%}")
    print(f"  Final copy_last accuracy:  {final['copy_last_accuracy']:.2%}")

    print("\nRefusal finetuning - Refuse COPY_FIRST (early-learned):")
    print(f"  Final copy_first accuracy: {refuse_copy_first_results['final_accuracy']['copy_first_accuracy']:.2%}")
    print(f"  Final copy_first refusal:  {refuse_copy_first_results['final_refusal']['copy_first_refusal_rate']:.2%}")
    print(f"  Final copy_last accuracy:  {refuse_copy_first_results['final_accuracy']['copy_last_accuracy']:.2%}")

    print("\nRefusal finetuning - Refuse COPY_LAST (late-learned):")
    print(f"  Final copy_last accuracy:  {refuse_copy_last_results['final_accuracy']['copy_last_accuracy']:.2%}")
    print(f"  Final copy_last refusal:   {refuse_copy_last_results['final_refusal']['copy_last_refusal_rate']:.2%}")
    print(f"  Final copy_first accuracy: {refuse_copy_last_results['final_accuracy']['copy_first_accuracy']:.2%}")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if wandb_run:
        wandb_run.finish()

    return results


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    if args.n_seeds == 1 and args.group_dir is None:
        output_dir = Path(args.results_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        run_one_experiment(args, args.seed, output_dir)
    else:
        if args.group_dir is not None:
            group_dir = Path(args.group_dir)
            group_dir.mkdir(parents=True, exist_ok=True)
        else:
            group_dir = (
                Path(args.results_dir)
                / f"{timestamp}_startCopyFirstFrac{args.start_copy_first_frac}"
                  f"_endCopyFirstFrac{args.end_copy_first_frac}_n{args.n_seeds}seeds"
            )
            group_dir.mkdir(parents=True, exist_ok=True)

        seeds = [args.seed + i for i in range(args.n_seeds)]

        # Write group_config only if not already present (parallel batches share one dir)
        config_path = group_dir / "group_config.json"
        if not config_path.exists():
            group_config = {k: v for k, v in vars(args).items()}
            group_config['seeds'] = seeds
            with open(config_path, "w") as f:
                json.dump(group_config, f, indent=2)

        print(f"Group dir: {group_dir}")
        print(f"Running {args.n_seeds} seeds: {seeds}")

        for i, seed in enumerate(seeds):
            print(f"\n{'=' * 60}")
            print(f"SEED {i + 1}/{args.n_seeds}  (seed={seed})")
            print(f"{'=' * 60}")
            seed_dir = group_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            run_one_experiment(args, seed, seed_dir, wandb_group=group_dir.name)

        print(f"\nAll {args.n_seeds} seeds complete. Group dir: {group_dir}")


if __name__ == "__main__":
    main()
