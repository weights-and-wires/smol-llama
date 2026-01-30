"""Utility functions for smol-llama training."""

from .rotary import precompute_freqs_cis, apply_rotary_emb
from .lr_schedule import get_cosine_lr, update_lr
from .data import download_dataset, get_batch, DataLoader
from .checkpoint import (
  save_checkpoint,
  load_checkpoint,
  upload_to_huggingface,
  get_hf_repo_id,
  save_and_upload_checkpoint,
  find_latest_checkpoint,
  load_hf_token,
)
from .model import ModelArgs, Llama
from .logging import init_wandb, log_metrics, finish_wandb


__all__ = [
  # Model
  "ModelArgs",
  "Llama",
  # Rotary embeddings
  "precompute_freqs_cis",
  "apply_rotary_emb",
  # Learning rate schedule
  "get_cosine_lr",
  "update_lr",
  # Data utilities
  "download_dataset",
  "get_batch",
  "DataLoader",
  # Checkpoint utilities
  "save_checkpoint",
  "load_checkpoint",
  "upload_to_huggingface",
  "get_hf_repo_id",
  "save_and_upload_checkpoint",
  "find_latest_checkpoint",
  "load_hf_token",
  # Logging
  "init_wandb",
  "log_metrics",
  "finish_wandb",
]
