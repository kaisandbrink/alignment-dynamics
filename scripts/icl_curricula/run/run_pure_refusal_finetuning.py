"""
Pure-refusal finetuning runner.

Loads pretrained checkpoints from existing group runs and performs refusal
finetuning where ONLY the refused task is present in the training data
(unlike the original run_refusal_finetuning which keeps 50/50 distribution).

Seeds that did not learn both tasks above --min_accuracy are skipped.

Usage:
    python scripts/icl_curricula/run/run_pure_refusal_finetuning.py \
        --group_dirs results/icl_copy_firstlast/runs/20260407..._0.9... \
                     results/icl_copy_firstlast/runs/20260407..._0.1... \
        --finetune_epochs 30 --finetune_lr 0.0001
"""

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

from src.models import Transformer
from src.datasets.CopyFirstLastICLDataset import CopyFirstLastICLDataset
from src.trainers import get_device


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (copied from icl_copy_firstlast_experiment.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy_by_task(model, dataset, device, batch_size=64):
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
    }


def compute_refusal_rate(model, dataset, device, batch_size=64):
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-sublayer weight-delta tracking
# ─────────────────────────────────────────────────────────────────────────────

# Block 0 (0-indexed) corresponds to "B1" in 1-indexed paper notation, etc.
SUBLAYER_PATTERNS = {
    'B1 ATT':    [r'^blocks\.0\.ln1\.', r'^blocks\.0\.attn\.'],
    'B1 FFN':    [r'^blocks\.0\.ln2\.', r'^blocks\.0\.ff\.'],
    'B2 ATT':    [r'^blocks\.1\.ln1\.', r'^blocks\.1\.attn\.'],
    'B2 FFN':    [r'^blocks\.1\.ln2\.', r'^blocks\.1\.ff\.'],
    'token_emb': [r'^token_emb\.'],
    'pos_enc':   [r'^pos_enc\.'],
    'ln_f':      [r'^ln_f\.'],
    'lm_head':   [r'^lm_head\.'],
}


def _sublayer_for_key(key: str):
    for name, patterns in SUBLAYER_PATTERNS.items():
        if any(re.match(p, key) for p in patterns):
            return name
    return None


def _strip_compile_prefix(key: str) -> str:
    return key[len('_orig_mod.'):] if key.startswith('_orig_mod.') else key


def snapshot_state(model) -> dict:
    """Return a detached, on-device clone of model.state_dict() with compile prefix stripped."""
    return {_strip_compile_prefix(k): v.detach().clone()
            for k, v in model.state_dict().items()}


@torch.no_grad()
def compute_sublayer_deltas(model, ref_state: dict) -> dict:
    """Per-sublayer Frobenius norm of (current - ref). Returns {sublayer_name: float}."""
    sq_sums = defaultdict(float)
    cur = model.state_dict()
    for raw_k, v in cur.items():
        k = _strip_compile_prefix(raw_k)
        if k not in ref_state:
            continue
        sub = _sublayer_for_key(k)
        if sub is None:
            continue
        sq_sums[sub] += (v.detach() - ref_state[k]).pow(2).sum().item()
    return {sub: float(sq_sums[sub] ** 0.5) for sub in SUBLAYER_PATTERNS}


# ─────────────────────────────────────────────────────────────────────────────
# Pure-refusal finetuning (dataset already configured; no redistribution)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_state_dict(model) -> dict:
    """Return model.state_dict() with the torch.compile '_orig_mod.' prefix
    stripped so the checkpoint loads directly into a fresh, uncompiled
    Transformer instance."""
    raw_state = model.state_dict()
    return {
        (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
        for k, v in raw_state.items()
    }


def pure_refusal_finetune(
    model_state: dict,
    model_config: dict,
    train_dataset: CopyFirstLastICLDataset,
    eval_dataset: CopyFirstLastICLDataset,
    refuse_task: str,
    finetune_lr: float,
    finetune_epochs: int,
    batch_size: int,
    ft_eval_every_n_steps: int,
    device: torch.device,
    checkpoint_save_path: Path = None,
    step_checkpoints_dir: Path = None,
    save_step_checkpoints_every_n_steps: int = 0,
) -> dict:
    """
    Finetune model to refuse one task using ONLY that task's sequences in training.
    train_dataset must already have task_distribution set to 100% refused task
    and mode set to refuse_{refuse_task}.
    """
    model = Transformer(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        max_seq_len=model_config['max_seq_len'],
        dropout=model_config['dropout'],
        use_learned_pe=model_config.get('use_learned_pe', False),
    )
    model.load_state_dict(model_state)
    model = model.to(device)
    # Snapshot pretrained weights *before* compile so the keys have no
    # `_orig_mod.` prefix; compute_sublayer_deltas strips it from current state
    # at each call.
    ref_state = snapshot_state(model)
    model = torch.compile(model)

    eval_dataset.set_mode('standard')

    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)

    history = {
        'steps': [],
        'train_loss': [],
        'copy_first_accuracy': [],
        'copy_last_accuracy': [],
        'copy_first_refusal_rate': [],
        'copy_last_refusal_rate': [],
        'delta_norm_per_sublayer': {sub: [] for sub in SUBLAYER_PATTERNS},
    }

    def _record_deltas():
        deltas = compute_sublayer_deltas(model, ref_state)
        for sub in SUBLAYER_PATTERNS:
            history['delta_norm_per_sublayer'][sub].append(deltas.get(sub, 0.0))

    global_step = 0

    # Evaluate before any finetuning
    acc = compute_accuracy_by_task(model, eval_dataset, device, batch_size)
    ref = compute_refusal_rate(model, eval_dataset, device, batch_size)
    history['steps'].append(0)
    history['train_loss'].append(None)
    history['copy_first_accuracy'].append(acc['copy_first_accuracy'])
    history['copy_last_accuracy'].append(acc['copy_last_accuracy'])
    history['copy_first_refusal_rate'].append(ref['copy_first_refusal_rate'])
    history['copy_last_refusal_rate'].append(ref['copy_last_refusal_rate'])
    _record_deltas()

    for epoch in range(finetune_epochs):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        pbar = tqdm(train_loader,
                    desc=f"Refuse {refuse_task} epoch {epoch + 1}/{finetune_epochs}",
                    unit="step")

        running_loss = 0.0
        running_count = 0

        for batch in pbar:
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
            running_loss += loss.item()
            running_count += 1

            if global_step % ft_eval_every_n_steps == 0:
                avg_loss = running_loss / running_count
                running_loss = 0.0
                running_count = 0

                acc = compute_accuracy_by_task(model, eval_dataset, device, batch_size)
                ref = compute_refusal_rate(model, eval_dataset, device, batch_size)

                history['steps'].append(global_step)
                history['train_loss'].append(avg_loss)
                history['copy_first_accuracy'].append(acc['copy_first_accuracy'])
                history['copy_last_accuracy'].append(acc['copy_last_accuracy'])
                history['copy_first_refusal_rate'].append(ref['copy_first_refusal_rate'])
                history['copy_last_refusal_rate'].append(ref['copy_last_refusal_rate'])
                _record_deltas()

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ref=f"{ref[f'{refuse_task}_refusal_rate']:.2%}",
                )

            # Periodic step-checkpoint save (independent cadence from eval)
            if (save_step_checkpoints_every_n_steps > 0
                    and step_checkpoints_dir is not None
                    and global_step % save_step_checkpoints_every_n_steps == 0):
                step_ckpt_path = step_checkpoints_dir / f"step_{global_step:04d}.pt"
                torch.save(
                    {'model_state_dict': _clean_state_dict(model),
                     'model_config': model_config},
                    step_ckpt_path,
                )

    final_acc = compute_accuracy_by_task(model, eval_dataset, device, batch_size)
    final_ref = compute_refusal_rate(model, eval_dataset, device, batch_size)

    if checkpoint_save_path is not None:
        torch.save(
            {'model_state_dict': _clean_state_dict(model),
             'model_config': model_config},
            checkpoint_save_path,
        )

    return {
        'history': history,
        'final_accuracy': final_acc,
        'final_refusal': final_ref,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Pure-refusal finetuning from pretrained checkpoints")
    parser.add_argument("--group_dirs", nargs="+", required=True,
                        help="One or more group run directories containing seed_XX subdirs")
    parser.add_argument("--min_accuracy", type=float, default=0.9,
                        help="Skip seeds where either task accuracy < this threshold (default 0.9)")
    parser.add_argument("--finetune_lr", type=float, default=0.0001,
                        help="Learning rate for refusal finetuning")
    parser.add_argument("--finetune_epochs", type=int, default=30,
                        help="Number of finetuning epochs")
    parser.add_argument("--ft_eval_every_n_steps", type=int, default=10,
                        help="Evaluate every N steps during finetuning")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--n_train_sequences", type=int, default=20000,
                        help="Number of training sequences for finetuning dataset")
    parser.add_argument("--n_eval_sequences", type=int, default=500,
                        help="Number of evaluation sequences")
    parser.add_argument("--kept_task_frac", type=float, default=0.0,
                        help="Fraction of kept (non-refused) task in training data (default 0.0 = pure refusal)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--no_skip_existing", action="store_true", default=False,
                        help="Re-run even if results already exist")
    parser.add_argument("--save_step_checkpoints_every_n_steps", type=int, default=0,
                        help="If > 0, save model checkpoints to "
                             "<output_subdir>/step_checkpoints/step_NNNN.pt every N "
                             "optimizer steps. 0 disables.")
    return parser.parse_args()


def process_group(group_dir: Path, args, device: torch.device):
    print(f"\n{'=' * 60}")
    print(f"GROUP: {group_dir.name}")
    print(f"{'=' * 60}")

    seed_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        print("  No seed dirs found, skipping.")
        return

    for seed_dir in seed_dirs:
        results_path = seed_dir / "results.json"
        ckpt_path = seed_dir / "pretrained_checkpoint.pt"

        if not results_path.exists():
            print(f"  {seed_dir.name}: no results.json, skipping.")
            continue
        if not ckpt_path.exists():
            print(f"  {seed_dir.name}: no pretrained_checkpoint.pt, skipping.")
            continue

        with open(results_path) as f:
            results = json.load(f)

        final_metrics = results.get("pretraining", {}).get("final_metrics", {})
        cf_acc = final_metrics.get("copy_first_accuracy", 0.0)
        cl_acc = final_metrics.get("copy_last_accuracy", 0.0)

        if cf_acc < args.min_accuracy or cl_acc < args.min_accuracy:
            print(f"  {seed_dir.name}: skipped (CF={cf_acc:.2%}, CL={cl_acc:.2%} < {args.min_accuracy:.0%})")
            continue

        print(f"\n  {seed_dir.name}: CF={cf_acc:.2%}, CL={cl_acc:.2%} — loading checkpoint...")

        ckpt = torch.load(ckpt_path, map_location=device)
        model_state = ckpt['model_state_dict']
        model_config = ckpt['model_config']

        cfg = results["config"]
        seed = cfg["seed"]

        # Eval dataset: fixed 50/50 standard (same across both refuse tasks)
        eval_dataset = CopyFirstLastICLDataset(
            n_sequences=args.n_eval_sequences,
            symbol_vocab_size=cfg["symbol_vocab_size"],
            feature_len=cfg["feature_len"],
            n_context_examples=cfg["n_context_examples"],
            task_distribution={'copy_first': 0.5, 'copy_last': 0.5},
            mode='standard',
            seed=seed + 1000,
        )

        for refuse_task in ['copy_first', 'copy_last']:
            other_task = 'copy_last' if refuse_task == 'copy_first' else 'copy_first'
            kept_frac = args.kept_task_frac
            suffix = f"_kept{kept_frac:.2f}" if kept_frac > 0 else ""
            output_subdir = seed_dir / f"pure_refusal_{refuse_task}{suffix}"
            output_file = output_subdir / "results.json"
            ckpt_file = output_subdir / "finetuned_checkpoint.pt"

            has_trajectory = False
            if output_file.exists():
                try:
                    with open(output_file) as f:
                        existing = json.load(f)
                    has_trajectory = 'delta_norm_per_sublayer' in existing.get('history', {})
                except Exception:
                    pass

            if (not args.no_skip_existing
                    and output_file.exists()
                    and ckpt_file.exists()
                    and has_trajectory):
                print(f"    refuse_{refuse_task}: already exists, skipping.")
                continue

            output_subdir.mkdir(exist_ok=True)
            print(f"    refuse_{refuse_task}: training ({args.finetune_epochs} epochs)...")

            # Per-step checkpoints: ensure directory exists, optionally wipe
            # any old step_*.pt files so the new run cleanly overwrites.
            step_ckpt_dir = None
            if args.save_step_checkpoints_every_n_steps > 0:
                step_ckpt_dir = output_subdir / 'step_checkpoints'
                step_ckpt_dir.mkdir(exist_ok=True)
                for old in step_ckpt_dir.glob('step_*.pt'):
                    old.unlink()

            train_dataset = CopyFirstLastICLDataset(
                n_sequences=args.n_train_sequences,
                symbol_vocab_size=cfg["symbol_vocab_size"],
                feature_len=cfg["feature_len"],
                n_context_examples=cfg["n_context_examples"],
                task_distribution={refuse_task: 1.0 - kept_frac, other_task: kept_frac},
                mode=f"refuse_{refuse_task}",
                seed=seed + 2000 + (0 if refuse_task == 'copy_first' else 1000),
            )

            ft_results = pure_refusal_finetune(
                model_state=model_state,
                model_config=model_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                refuse_task=refuse_task,
                finetune_lr=args.finetune_lr,
                finetune_epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                ft_eval_every_n_steps=args.ft_eval_every_n_steps,
                device=device,
                checkpoint_save_path=output_subdir / 'finetuned_checkpoint.pt',
                step_checkpoints_dir=step_ckpt_dir,
                save_step_checkpoints_every_n_steps=args.save_step_checkpoints_every_n_steps,
            )

            output = {
                "refuse_task": refuse_task,
                "config": {
                    "finetune_lr": args.finetune_lr,
                    "finetune_epochs": args.finetune_epochs,
                    "n_train_sequences": args.n_train_sequences,
                    "n_eval_sequences": args.n_eval_sequences,
                    "batch_size": args.batch_size,
                    "pretrain_seed": seed,
                    "symbol_vocab_size": cfg["symbol_vocab_size"],
                    "feature_len": cfg["feature_len"],
                    "n_context_examples": cfg["n_context_examples"],
                    "start_copy_first_frac": cfg.get("start_copy_first_frac"),
                    "kept_task_frac": kept_frac,
                    "model_config": model_config,
                },
                "pretrain_final_accuracy": final_metrics,
                "history": ft_results["history"],
                "final_accuracy": ft_results["final_accuracy"],
                "final_refusal": ft_results["final_refusal"],
            }

            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)

            fa = ft_results["final_accuracy"]
            fr = ft_results["final_refusal"]
            print(f"      done. refuse_rate={fr[f'{refuse_task}_refusal_rate']:.2%}, "
                  f"kept_acc={fa[f'{other_task}_accuracy']:.2%}")


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Min accuracy threshold: {args.min_accuracy:.0%}")

    for group_dir_str in args.group_dirs:
        group_dir = Path(group_dir_str)
        if not group_dir.exists():
            print(f"WARNING: {group_dir} does not exist, skipping.")
            continue
        process_group(group_dir, args, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
