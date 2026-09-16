#!/usr/bin/env bash

# Mixed-refusal finetuning sweep: 90% refused task + 10% kept task
# Runs sequentially over the three curricula group dirs produced by
# run_experiments_copy_firstlast.sh.
#
# Set GROUP_DIR_* to the timestamped directories from that run before executing.

# ── Set these to match your pretraining output dirs ───────────────────────────
GROUP_DIR_09="results/icl_copy_firstlast/runs/<TIMESTAMP>_startCopyFirstFrac0.9_endCopyFirstFrac0.5_n30seeds"
GROUP_DIR_01="results/icl_copy_firstlast/runs/<TIMESTAMP>_startCopyFirstFrac0.1_endCopyFirstFrac0.5_n30seeds"
GROUP_DIR_05="results/icl_copy_firstlast/runs/<TIMESTAMP>_startCopyFirstFrac0.5_endCopyFirstFrac0.5_n30seeds"
# ──────────────────────────────────────────────────────────────────────────────

COMMON="python scripts/icl_curricula/run/run_pure_refusal_finetuning.py \
  --finetune_lr 0.00005 \
  --finetune_epochs 3 \
  --ft_eval_every_n_steps 1 \
  --batch_size 64 \
  --n_train_sequences 20000 \
  --min_accuracy 0.9 \
  --kept_task_frac 0.1 \
  --save_step_checkpoints_every_n_steps 20 \
  --no_skip_existing"

echo "Running: copy_first early (start=0.9)"
$COMMON --group_dirs "$GROUP_DIR_09"

echo "Running: copy_first late (start=0.1)"
$COMMON --group_dirs "$GROUP_DIR_01"

echo "Running: balanced (start=0.5)"
$COMMON --group_dirs "$GROUP_DIR_05"

echo "All mixed-refusal finetuning runs complete."
