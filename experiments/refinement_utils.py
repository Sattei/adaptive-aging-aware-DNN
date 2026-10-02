"""Reusable, non-architectural helpers for temporal-refinement experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset, random_split


def split_lengths(total_len: int) -> tuple[int, int, int]:
    """Mirror ``TrainingPipeline``'s established train/validation/test split policy."""
    if total_len <= 0:
        raise ValueError("A non-empty dataset is required")
    if total_len == 1:
        return 1, 0, 0
    if total_len == 2:
        return 1, 0, 1

    train_len = max(int(0.8 * total_len), 1)
    val_len = max(int(0.1 * total_len), 1)
    test_len = total_len - train_len - val_len
    if test_len <= 0:
        test_len = 1
        if train_len > val_len:
            train_len -= 1
        else:
            val_len -= 1
    return train_len, val_len, test_len


def split_indices(dataset: Any, seed: int) -> dict[str, list[int]]:
    """Return the exact deterministic memberships used by the training pipeline."""
    lengths = split_lengths(len(dataset))
    generator = torch.Generator().manual_seed(int(seed))
    subsets = random_split(dataset, lengths, generator=generator)
    return {
        name: [int(index) for index in subset.indices]
        for name, subset in zip(("train", "val", "test"), subsets)
    }


class HistoryWindowDataset(Dataset):
    """Expose the latest causal history window without altering source samples."""

    def __init__(self, dataset: Any, history_length: int):
        if len(dataset) == 0:
            raise ValueError("HistoryWindowDataset requires a non-empty dataset")
        self.dataset = dataset
        self.history_length = int(history_length)
        available = int(dataset[0].x_history.shape[1])
        if not 1 <= self.history_length <= available:
            raise ValueError(
                f"history_length must be in [1, {available}], got {self.history_length}"
            )
        self.available_history_length = available
        for attribute in ("root", "split", "seed"):
            if hasattr(dataset, attribute):
                setattr(self, attribute, getattr(dataset, attribute))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index].clone()
        sample.x_history = sample.x_history[:, -self.history_length :, :].clone()
        if hasattr(sample, "temperature_history_k"):
            sample.temperature_history_k = sample.temperature_history_k[:, -self.history_length :].clone()
        return sample
