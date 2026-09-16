#!/usr/bin/env bash

# Copy-First/Last experiment sweep: 3 curricula × N_SEEDS seeds (sequential)

# ── Configuration ─────────────────────────────────────────────────────────────
N_SEEDS=30       # seeds per curriculum
START_SEED=50    # first seed value (seeds are START_SEED … START_SEED+N_SEEDS-1)
N_LAYERS=2
LR=0.0003
# ──────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d%H%M%S)
RESULTS_DIR="results/icl_copy_firstlast/runs"

DIR_09="${RESULTS_DIR}/${TIMESTAMP}_startCopyFirstFrac0.9_endCopyFirstFrac0.5_n${N_SEEDS}seeds"
DIR_01="${RESULTS_DIR}/${TIMESTAMP}_startCopyFirstFrac0.1_endCopyFirstFrac0.5_n${N_SEEDS}seeds"
DIR_05="${RESULTS_DIR}/${TIMESTAMP}_startCopyFirstFrac0.5_endCopyFirstFrac0.5_n${N_SEEDS}seeds"
mkdir -p "$DIR_09" "$DIR_01" "$DIR_05"

COMMON="python scripts/icl_curricula/run/icl_copy_firstlast_experiment.py \
  --pretrain_epochs 100 \
  --end_copy_first_frac 0.5 \
  --feature_len 5 \
  --n_context_examples 4 \
  --lr $LR \
  --optimizer adamw \
  --n_train_sequences 20000 \
  --use_learned_pe \
  --d_model 128 \
  --symbol_vocab_size 50 \
  --checkpoint_every_n_epochs 5 \
  --n_layers $N_LAYERS \
  --n_seeds $N_SEEDS \
  --seed $START_SEED"

echo "Running curriculum: start_copy_first_frac=0.9"
$COMMON --start_copy_first_frac 0.9 --group_dir "$DIR_09"

echo "Running curriculum: start_copy_first_frac=0.1"
$COMMON --start_copy_first_frac 0.1 --group_dir "$DIR_01"

echo "Running curriculum: start_copy_first_frac=0.5"
$COMMON --start_copy_first_frac 0.5 --group_dir "$DIR_05"

echo "All runs complete."
