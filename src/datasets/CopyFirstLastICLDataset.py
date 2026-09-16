"""
In-context learning dataset for copy-first vs copy-last classification.

The model must infer the task rule from labeled examples, then apply it to a query.
No explicit task cue - pure in-context learning.

Task 1 (copy_first): Label = first token of the feature string.
Task 2 (copy_last):  Label = last token of the feature string.

Features are random sequences drawn from the full symbol vocabulary.
The symbols are independently sampled per position, so repeats are possible.

Neither task requires ICL in isolation (the model could always copy position 0 or -1
without looking at context). When both tasks are mixed, context is required to infer
which task is active.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class CopyFirstLastICLDataset(Dataset):
    """
    In-context learning dataset for copy-first vs copy-last classification.

    Each sequence contains:
    - N in-context examples: feature strings with labels (the copied symbol)
    - 1 query: feature string without label (model must predict)

    The model must infer from the examples whether labels correspond to the
    first token or the last token of each feature string.

    Sequence format (tokenized):
    [feat1_tokens] [COLON] [label1] [SEP] ... [query_tokens] [COLON]
    Target: the correct symbol token (or REFUSE token in refusal mode)

    Args:
        n_sequences: Number of sequences to generate
        symbol_vocab_size: Number of distinct symbols to sample from (default 20)
        feature_len: Length of each feature string, must be >= 2 (default 6)
        n_context_examples: Number of labeled examples before query (default 6)
        task_distribution: Dict mapping task names to probabilities
                          e.g., {'copy_first': 0.5, 'copy_last': 0.5}
        mode: 'standard', 'refuse_copy_first', or 'refuse_copy_last'
        seed: Random seed for reproducibility
    """

    VALID_MODES = ('standard', 'refuse_copy_first', 'refuse_copy_last')

    def __init__(
        self,
        n_sequences: int,
        symbol_vocab_size: int = 20,
        feature_len: int = 6,
        n_context_examples: int = 6,
        task_distribution: dict = None,
        mode: str = 'standard',
        seed: int = 42,
    ):
        assert feature_len >= 2, "feature_len must be at least 2 (need distinct first and last positions)"
        assert n_context_examples >= 0, "n_context_examples must be non-negative"
        assert mode in self.VALID_MODES, f"mode must be one of {self.VALID_MODES}"

        self.n_sequences = n_sequences
        self.symbol_vocab_size = symbol_vocab_size
        self.feature_len = feature_len
        self.n_context_examples = n_context_examples
        self.mode = mode
        self._seed = seed

        # Task distribution (default 50/50)
        self.task_distribution = task_distribution or {'copy_first': 0.5, 'copy_last': 0.5}
        assert set(self.task_distribution.keys()) == {'copy_first', 'copy_last'}
        assert abs(sum(self.task_distribution.values()) - 1.0) < 1e-6

        # Special tokens (after symbol vocab)
        self.colon_token = symbol_vocab_size      # :
        self.sep_token = symbol_vocab_size + 1    # |
        self.refuse_token = symbol_vocab_size + 2  # [REFUSE]

        # Total vocabulary size (symbols + special tokens)
        self.total_vocab_size = symbol_vocab_size + 3

        # Sequence length: (feature_len + COLON + LABEL + SEP) * n_context + (feature_len + COLON)
        self.seq_len = (feature_len + 3) * n_context_examples + (feature_len + 1)

        # Pre-generate all sequences
        self._generate_all_sequences()

    def _generate_all_sequences(self):
        """Pre-generate all sequences deterministically."""
        rng = np.random.default_rng(self._seed)

        self.sequences = []
        self.targets = []
        self.tasks = []
        self.standard_targets = []

        task_probs = [self.task_distribution['copy_first'], self.task_distribution['copy_last']]
        task_choices = rng.choice(
            ['copy_first', 'copy_last'],
            size=self.n_sequences,
            p=task_probs,
        )

        for i in range(self.n_sequences):
            task = task_choices[i]
            seq_data = self._generate_sequence(task, rng)
            self.sequences.append(seq_data['input'])
            self.tasks.append(task)
            self.standard_targets.append(seq_data['standard_target'])

            if self.mode == 'standard':
                self.targets.append(seq_data['standard_target'])
            elif self.mode == 'refuse_copy_first' and task == 'copy_first':
                self.targets.append(self.refuse_token)
            elif self.mode == 'refuse_copy_last' and task == 'copy_last':
                self.targets.append(self.refuse_token)
            else:
                self.targets.append(seq_data['standard_target'])

        self.sequences = torch.stack(self.sequences)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        self.standard_targets = torch.tensor(self.standard_targets, dtype=torch.long)

    def _generate_sequence(self, task: str, rng) -> dict:
        """Generate a single ICL sequence."""
        seen_features = set()
        examples = []

        n_needed = self.n_context_examples + 1  # +1 for query

        attempts = 0
        max_attempts = n_needed * 100

        while len(examples) < n_needed and attempts < max_attempts:
            # Sample a random feature from the full symbol vocabulary
            feature = [int(rng.integers(self.symbol_vocab_size)) for _ in range(self.feature_len)]
            feature_tuple = tuple(feature)

            if feature_tuple not in seen_features:
                seen_features.add(feature_tuple)
                label = feature[0] if task == 'copy_first' else feature[-1]
                examples.append({'feature': feature, 'label': label})

            attempts += 1

        if len(examples) < n_needed:
            raise ValueError(
                f"Could not generate {n_needed} unique features. "
                f"Try increasing symbol_vocab_size or decreasing feature_len / n_context_examples."
            )

        context_examples = examples[:-1]
        query_example = examples[-1]

        # Build token sequence
        tokens = []
        for ex in context_examples:
            tokens.extend(ex['feature'])
            tokens.append(self.colon_token)
            tokens.append(ex['label'])
            tokens.append(self.sep_token)

        tokens.extend(query_example['feature'])
        tokens.append(self.colon_token)

        return {
            'input': torch.tensor(tokens, dtype=torch.long),
            'standard_target': query_example['label'],
            'task': task,
        }

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> dict:
        return {
            'input_ids': self.sequences[idx],
            'labels': self.targets[idx:idx + 1],
            'task': self.tasks[idx],
        }

    def get_task(self, idx: int) -> str:
        return self.tasks[idx]

    def get_standard_target(self, idx: int) -> int:
        return self.standard_targets[idx].item()

    def get_copy_first_indices(self) -> torch.Tensor:
        return torch.tensor([i for i, t in enumerate(self.tasks) if t == 'copy_first'])

    def get_copy_last_indices(self) -> torch.Tensor:
        return torch.tensor([i for i, t in enumerate(self.tasks) if t == 'copy_last'])

    def set_mode(self, mode: str) -> None:
        assert mode in self.VALID_MODES, f"mode must be one of {self.VALID_MODES}"
        self.mode = mode

        new_targets = []
        for i in range(self.n_sequences):
            task = self.tasks[i]
            standard_target = self.standard_targets[i].item()

            if mode == 'standard':
                new_targets.append(standard_target)
            elif mode == 'refuse_copy_first' and task == 'copy_first':
                new_targets.append(self.refuse_token)
            elif mode == 'refuse_copy_last' and task == 'copy_last':
                new_targets.append(self.refuse_token)
            else:
                new_targets.append(standard_target)

        self.targets = torch.tensor(new_targets, dtype=torch.long)

    def set_seed(self, seed: int) -> None:
        """Update the seed used for sequence generation (takes effect on next regeneration)."""
        self._seed = seed

    def set_task_distribution(self, distribution: dict) -> None:
        self.task_distribution = distribution
        self._generate_all_sequences()

    @property
    def effective_vocab_size(self) -> int:
        return self.total_vocab_size
