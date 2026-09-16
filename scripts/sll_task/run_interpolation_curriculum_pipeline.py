"""
06/05/2026

Train transformers on the Synthetic Language Learning (SLL) task under different
pretraining curricula and evaluate alignment fine-tuning dynamics.

The script supports multiple curriculum schedules controlling the temporal
distribution of aligned, neutral, and misaligned examples during pretraining,
including: static, clean_early, clean_late, static_low, static_high

After pretraining, models are fine-tuned exclusively on aligned and neutral 
examples to study how curricula influence downstream alignment behavior and
representation learning.

For each random seed, the script:
    1. Loads the SLL dataset and rule system
    2. Constructs curriculum-dependent sampling schedules
    3. Trains a transformer model during pretraining
    4. Fine-tunes the model on aligned data
    5. Saves checkpoints, histories, and training parameters
Results are saved separately for each seed.

"""

# --------------------------------------
## imports
# --------------------------------------

import argparse
import json
import os
import random
from datetime import datetime
import numpy as np
import torch

import datasets.DeonticEthicsDataset as deontic_utils
from datasets.DeonticEthicsDataset import TemplateDataset
from alignment_datasets.DatasetBuildingBlocks import CombinedDataset
from models import Transformer
from trainers import train, TrainingConfig

# --------------------------------------
## params
# --------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--max_seq_len', type=int, default=60)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=800)
parser.add_argument('--n_epochs_fine_tune', type=int, default=200)
parser.add_argument('--checkpoint_interval', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--data_path', type=str, default="scripts/sll_task/dataset/natural_learning_seed_566_rules_5/corpus_data.json")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_seeds', type=int, default=1)
parser.add_argument('--corruption_level', type=float, default=0.4, help='the average level of misaligned weight during pretraining')
parser.add_argument('--misaligned_low', type=float, default=0.05, help='the lowest level of misaligned weight in the curriculum')
parser.add_argument('--curriculum', type=str, default='static', help='curriculum, options: static, static_low, static_high, clean_early, clean_late')
parser.add_argument('--initial_plateau', type=float, default=0.15)
parser.add_argument('--finetuning_aligned', type=float, default=0.5)
args = parser.parse_args()

# --------------------------------------
## load data
# --------------------------------------

data_path = args.data_path
with open(data_path , 'r') as file:
    data = json.load(file)

corpus = data['corpus']
RuleSystem = deontic_utils.RuleSet.from_dict(data['rules_plus_grammar'])
grammar = RuleSystem.grammar
n_rules = len(RuleSystem.rules)
train_data = data['train_data']

stoi = grammar.stoi
itos = grammar.itos
vocab_size = grammar.vocab_size
eos_id = grammar.eos_id
features = grammar.variables
preambles = grammar.preambles
init_str = grammar.init_str

# reconstruct derived args
args.vocab_size = vocab_size
args.use_rotary_pe = True
args.finetuning_weights = {'aligned': args.finetuning_aligned, 'misaligned': 0, 'neutral': 1 - args.finetuning_aligned}

# set seed
rng = random.Random(args.seed)

# --------------------------------------
## curriculum
# --------------------------------------

def linear_weight_interpolation(epoch, start=0.1, end=0.5, plateau=0.1, n_epochs=100, reverse=False):
    """
    Piecewise schedule: hold at `start` for an initial plateau, then linearly ramp to `end`.
    Args:
        epoch: Current epoch (0-based).
        start: Value during plateau.
        end: Final value at last epoch.
        plateau: Fraction of epochs at constant `start`.
        n_epochs: Total number of epochs.
        reverse: reverse for counter
    Returns:
        Interpolated value for this epoch.
    """
    assert n_epochs > 1, 'Too few epochs'

    if reverse:
        epoch = n_epochs - 1 - epoch

    plateau_steps = int(n_epochs * plateau)
    plateau_steps = min(max(plateau_steps, 0), n_epochs - 1)

    if epoch < plateau_steps:
        return start

    ramp_steps = n_epochs - plateau_steps
    t = (epoch - plateau_steps) / max(ramp_steps - 1, 1)

    return start + (end - start) * t


def find_misaligned_end(start, plateau, corruption_level):
    """
    Solve for `end` so a plateau+linear schedule has a given average.
    Args:
        start: Plateau value.
        plateau: Fraction of training at plateau.
        corruption_level: Desired average value over training.
    """
    if plateau >= 1.0:
        return start
    return 2 * (corruption_level - plateau * start) / (1 - plateau) - start

# --------------------------------------
## curriculum
# --------------------------------------

misaligned_high = find_misaligned_end(args.misaligned_low, args.initial_plateau, args.corruption_level)
assert 0.0 <= misaligned_high <= 1.0

if args.curriculum == "static":
    pretraining_weights = lambda epoch: {
        "misaligned": args.corruption_level,
        "neutral": (1.0 - args.corruption_level) / 2,
        "aligned": (1.0 - args.corruption_level) / 2,
    }
elif args.curriculum == "clean_early":
    def pretraining_weights(epoch):
        w_mis = linear_weight_interpolation(
            epoch, args.misaligned_low, misaligned_high, args.initial_plateau, args.n_epochs
        )
        return {
            "misaligned": w_mis,
            "neutral": (1.0 - w_mis) / 2,
            "aligned": (1.0 - w_mis) / 2,
        }
elif args.curriculum == "clean_late":
    def pretraining_weights(epoch):
        w_mis = linear_weight_interpolation(
            epoch, args.misaligned_low, misaligned_high, args.initial_plateau, args.n_epochs,
            reverse = True
        )
        return {
            "misaligned": w_mis,
            "neutral": (1.0 - w_mis) / 2,
            "aligned": (1.0 - w_mis) / 2,
        }
elif args.curriculum == "static_low":
    pretraining_weights = lambda epoch: {
        "misaligned": args.misaligned_low,
        "neutral": (1.0 - args.misaligned_low) / 2,
        "aligned": (1.0 - args.misaligned_low) / 2,
    }
elif args.curriculum == "static_high":
    pretraining_weights = lambda epoch: {
        "misaligned": misaligned_high,
        "neutral": (1.0 - misaligned_high) / 2,
        "aligned": (1.0 - misaligned_high) / 2,
    }
else:
    raise ValueError(f"Unknown curriculum: {args.curriculum}")

# --------------------------------------
## datasets
# --------------------------------------

## form datasets
train_aligned_dataset = TemplateDataset(train_data['aligned'], name='aligned')
train_misaligned_dataset = TemplateDataset(train_data['misaligned'], name='misaligned')
train_neutral_dataset = TemplateDataset(train_data['neutral'], name='neutral')

train_dataset = CombinedDataset([train_aligned_dataset, train_misaligned_dataset, train_neutral_dataset])
eval_dataset= CombinedDataset([train_aligned_dataset])

# --------------------------------------
## saving
# --------------------------------------

ts = datetime.now().strftime("%Y%m%d%H%M%S")
save_dir = f'scripts/sll_task/results/linear_interpolation_curriculum/{args.curriculum}/{ts}/'

# --------------------------------------
## run model
# --------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'using device: {device}')

config1 = TrainingConfig(epochs = args.n_epochs,
                        batch_size=min(args.batch_size, len(train_dataset)),
                        learning_rate = args.lr,
                        checkpoint_interval= args.checkpoint_interval,
                        device = device
                        )

config2 = TrainingConfig(epochs = args.n_epochs_fine_tune,
                        batch_size=min(args.batch_size, len(train_dataset)),
                        learning_rate = args.lr,
                        checkpoint_interval= args.checkpoint_interval,
                        device = device
                        )

for seed_i in range(args.n_seeds):

    print(f'############ running network {seed_i} #####################')

    random.seed(args.seed + seed_i)
    np.random.seed(args.seed + seed_i)
    torch.manual_seed(args.seed + seed_i)

    model = Transformer(
        vocab_size=args.vocab_size,  # +1 for rejection token
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_rotary_pe=args.use_rotary_pe
        )
    
    # optimizer = torch.optim.SGD
    optimizer = torch.optim.Adam
    
    history1 = train(
                    model=model,
                    train_dataset = train_dataset,
                    eval_dataset= eval_dataset,
                    optimizer_fn= optimizer,
                    config=config1,
                    weights_by_dataset_epochs = pretraining_weights,
                    eval_func = None)

    history2 = train(
                    model=model,
                    train_dataset = train_dataset,
                    eval_dataset= eval_dataset,
                    optimizer_fn= optimizer,
                    config=config2,
                    weights_by_dataset_epochs = lambda epoch: args.finetuning_weights,
                    eval_func = None)
    
    
    history = {}
    for k in history1:
        v1, v2 = history1[k], history2[k]
        if isinstance(v1, list):
            history[k] = v1 + v2 
        elif isinstance(v1, dict):  # checkpoints
            offset1 = config1.epochs
            history[k] = {**v1, **{ck + offset1: cv for ck, cv in v2.items()}}
        else:
            history[k] = v2  # take the second phase's value for scalars

    ### saving -----------------
    out_dir =  f'{save_dir}{args.seed + seed_i}/'
    os.makedirs(out_dir, exist_ok=True)
    ## save history
    torch.save(history, os.path.join(out_dir, "history.pt"))
    ## save config
    with open(os.path.join(out_dir, "model_config_train.json"), "w") as f:
        json.dump(config1.__dict__, f, indent=2)
    with open(os.path.join(out_dir, "model_config_finetune.json"), "w") as f:
        json.dump(config2.__dict__, f, indent=2)
    ## save params
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f'saved to {out_dir}')
