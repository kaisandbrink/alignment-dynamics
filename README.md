# Pretraining Curricula Enable Selective Fine-tuning — Code Supplement

This repository contains the code for all experiments from the paper.

---

## Setup

```bash
conda env create -f environment.yml
conda activate pretraining_curricula
pip install -e .
```

---

## Reproducing the paper figures

### Figures 1–4: ICL copy-first/copy-last experiment

**Step 1 — Pretraining (three curricula)**

```bash
bash scripts/icl_curricula/run/run_experiments_copy_firstlast.sh
```

Trains models under three curricula (copy-first fraction starting at 0.9, 0.1, and 0.5) for 30 seeds each, sequentially. Results are written to timestamped directories under `results/icl_copy_firstlast/runs/`.

**Step 2 — Refusal finetuning**

Set the `GROUP_DIR_*` variables in `run_mixed_refusal_sweep.sh` to the timestamped directories from Step 1, then run:

```bash
bash scripts/icl_curricula/run/run_mixed_refusal_sweep.sh
```

**Step 3 — Plot figures**

Open and run the notebooks in `scripts/icl_curricula/analyze/`:

| Notebook | Figure |
|---|---|
| `icl_copy_firstlast_analysis.ipynb` | Figure 1 |
| `icl_copy_firstlast_pure_refusal_analysis.ipynb` | Figure 2 |
| `icl_copy_firstlast_mechanism_analysis.ipynb` | Figure 3 |
| `icl_copy_firstlast_activation_patching.ipynb` | Figure 4A |
| `icl_copy_firstlast_refusal_l2summary.ipynb` | Figure 4B |

### Figures 5B–7: SLL task experiment

**Step 1 — Generate the corpus**

```bash
python scripts/sll_task/generate_alignment_corpus.py
```

This produces the dataset at `scripts/sll_task/dataset/natural_learning_seed_566_rules_5/corpus_data.json`.

**Step 2 — Train models under each curriculum**

Run the pipeline separately for each curriculum (options: `static`, `clean_early`):

```bash
python scripts/sll_task/run_interpolation_curriculum_pipeline.py --curriculum clean_early --n_seeds 5
python scripts/sll_task/run_interpolation_curriculum_pipeline.py --curriculum static --n_seeds 5
```

Results are saved to `scripts/sll_task/results/linear_interpolation_curriculum/{curriculum}/`.

**Step 3 — Plot figures**

Open and run the notebooks in `scripts/sll_task`:

| Notebook | Figure |
|---|---|
| `scripts/sll_task/plot_fig5b.ipynb` | Figure 5b |
| `scripts/sll_task/plot_fig6.ipynb` | Figure 6 |
| `scripts/sll_task/plot_fig7.ipynb` | Figure 7 |

---

## Project structure

```
scripts/
├── icl_curricula/
│   ├── run/
│   │   ├── run_experiments_copy_firstlast.sh       # ICL pretraining sweep
│   │   ├── run_mixed_refusal_sweep.sh              # ICL refusal finetuning
│   │   ├── icl_copy_firstlast_experiment.py        # Core pretraining script
│   │   └── run_pure_refusal_finetuning.py          # Core finetuning script
│   └── analyze/
│       ├── icl_copy_firstlast_analysis.ipynb               # Figure 1
│       ├── icl_copy_firstlast_pure_refusal_analysis.ipynb  # Figure 2
│       ├── icl_copy_firstlast_mechanism_analysis.ipynb     # Figure 3
│       ├── icl_copy_firstlast_activation_patching.ipynb    # Figure 4A
│       └── icl_copy_firstlast_refusal_l2summary.ipynb      # Figure 4B
└── sll_task/
    ├── generate_alignment_corpus.py                # Corpus generation
    ├── run_interpolation_curriculum_pipeline.py    # Training pipeline
    ├── plot_fig5b.ipynb                            # Figure 5B
    ├── plot_fig6.ipynb                             # Figure 6
    └── plot_fig7.ipynb                             # Figure 7

src/
├── models.py                              # Transformer architecture
├── trainers.py                            # Training utilities
├── plot_utils.py                          # Plotting helpers
├── datasets/
│   ├── CopyFirstLastICLDataset.py         # ICL task dataset
│   └── DeonticEthicsDataset.py            # SLL task dataset
└── deontic_evals/                         # SLL evaluation utilities
```
