"""Data loading utilities for training."""

import os
import torch
import numpy as np
from huggingface_hub import snapshot_download


def download_dataset(
    repo_id: str = "weights-and-wires/fineweb-6b",
    local_dir: str = "data_bin",
    allow_patterns: list | None = None,
) -> str:
    """
    Download dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace dataset repository ID.
        local_dir: Local directory to save the dataset.
        allow_patterns: File patterns to download (default: ["tokenized/*.bin", "*.json"]).

    Returns:
        Path to the local directory containing the dataset.
    """
    if allow_patterns is None:
        allow_patterns = ["tokenized/*.bin", "*.json"]

    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )

    print(f"Data downloaded to ./{local_dir}")
    return local_dir


def get_batch(
    split: str, data_dir: str, batch_size: int, block_size: int, device: str = "cuda"
):
    """
    Get a random batch of training or validation data.

    Args:
        split: Data split ("train" or "val").
        data_dir: Directory containing the binary data files.
        batch_size: Number of sequences per batch.
        block_size: Sequence length (context size).
        device: Device to move tensors to (default: "cuda").

    Returns:
        Tuple of (inputs, targets) tensors of shape [batch_size, block_size].
    """
    data = np.memmap(f"{data_dir}/tokenized/{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )
    return x.to(device), y.to(device)


class DataLoader:
    """Simple data loader for memory-mapped binary files."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_size: int,
        block_size: int,
        device: str = "cuda",
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing the binary data files.
            split: Data split ("train" or "val").
            batch_size: Number of sequences per batch.
            block_size: Sequence length (context size).
            device: Device to move tensors to.
        """
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self._data = None

    @property
    def data(self):
        """Lazily load the memory-mapped data."""
        if self._data is None:
            self._data = np.memmap(
                f"{self.data_dir}/tokenized/{self.split}.bin", dtype=np.uint16, mode="r"
            )
        return self._data

    def get_batch(self):
        """Get a random batch of data."""
        return get_batch(
            self.split, self.data_dir, self.batch_size, self.block_size, self.device
        )

    def __len__(self):
        """Return the approximate number of batches in the dataset."""
        return (len(self.data) - self.block_size) // (self.batch_size * self.block_size)
