from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset

class CombinableDataset(Dataset, ABC):
    """Abstract base class for datasets that can be combined with metadata tracking."""

    name: str
    metadata: list

    def __init__(self, name: str, length: int, metadata: dict | None = None):
        self.name = name
        base = {"datasets": [name]}
        if metadata:
            base.update(metadata)
        self.metadata = [base.copy() for _ in range(length)]

    def update_metadata(self, metadata: dict, apply_to_datasets: set | list | None = None):
        for entry in self.metadata:
            if apply_to_datasets is not None:
                # Only update if at least one of the entry's datasets is in apply_to_datasets
                if not any(d in apply_to_datasets for d in entry.get("datasets", [])):
                    continue
            entry.update(metadata)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class CombinedDataset(ConcatDataset):
    """A concatenated dataset that aggregates metadata from CombinableDataset instances."""

    def __init__(self, datasets: list[CombinableDataset], name: str | None = None):
        super().__init__(datasets)

        self.name = name or "+".join(d.name for d in datasets)

        self.metadata = []
        for d in datasets:
            self.metadata.extend(d.metadata)

    def update_metadata(self, metadata: dict, apply_to_datasets: set | list | None = None):
        for d in self.datasets:
            d.update_metadata(metadata, apply_to_datasets)

    def construct_weights_list(self, weights_by_dataset: dict) -> list[float]:
        """Construct a list of weights for each item in the combined dataset based on the provided weights_dict."""
        weights_list = []
        for d in self.datasets:
            weight = weights_by_dataset.get(d.name, 1.0)
            weights_list.extend([weight] * len(d))
        return weights_list

    def filter(self, name: str) -> CombinableDataset:
        """Get a sub-dataset by name."""
        for d in self.datasets:
            if d.name == name:
                return d
        raise KeyError(f"No sub-dataset with name '{name}' found. Available: {[d.name for d in self.datasets]}")
