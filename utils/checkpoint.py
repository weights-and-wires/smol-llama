"""Checkpoint saving, loading, and HuggingFace upload utilities."""

import os
import glob
import torch
from typing import Optional, Dict, Any


def load_hf_token() -> Optional[str]:
  """
  Load HuggingFace token from environment or .env file.

  Returns:
      HF token string or None if not found.
  """
  token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
  if token:
    return token

  env_path = ".env"
  if os.path.exists(env_path):
    with open(env_path, "r") as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
          if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
              os.environ[key] = value
              return value
  return None


def save_checkpoint(model, optimizer, step: int, loss: float, args: Dict[str, Any], path: str) -> str:
  """
  Save a training checkpoint locally.

  Args:
      model: The model to save.
      optimizer: The optimizer to save.
      step: Current training step.
      loss: Current loss value.
      args: Training arguments dictionary.
      path: Path to save the checkpoint.

  Returns:
      Path to the saved checkpoint.
  """
  ckpt = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
    "loss": loss,
    "args": args,
  }
  torch.save(ckpt, path)
  return path


def load_checkpoint(path: str, model=None, optimizer=None, device: str = "cuda") -> Dict[str, Any]:
  """
  Load a training checkpoint.

  Args:
      path: Path to the checkpoint file.
      model: Optional model to load weights into.
      optimizer: Optional optimizer to load state into.
      device: Device to map the checkpoint to.

  Returns:
      Dictionary containing checkpoint data.
  """
  ckpt = torch.load(path, map_location=device)

  if model is not None:
    model.load_state_dict(ckpt["model"])

  if optimizer is not None:
    optimizer.load_state_dict(ckpt["optimizer"])

  return ckpt


def upload_to_huggingface(
  local_path: str,
  repo_id: str,
  path_in_repo: Optional[str] = None,
  repo_type: str = "model",
  remove_local: bool = True,
) -> bool:
  """
  Upload a file to HuggingFace Hub.

  Args:
      local_path: Local path to the file to upload.
      repo_id: HuggingFace repository ID.
      path_in_repo: Path within the repo (default: checkpoints/{filename}).
      repo_type: Type of repository ("model" or "dataset").
      remove_local: Whether to remove the local file after upload.

  Returns:
      True if upload succeeded, False otherwise.
  """
  try:
    from huggingface_hub import HfApi

    token = load_hf_token()
    if not token:
      print("No HF_TOKEN found in environment or .env file")
      print("Attempting upload without explicit token (may fail)...")

    if path_in_repo is None:
      filename = os.path.basename(local_path)
      path_in_repo = f"checkpoints/{filename}"

    api = HfApi(token=token)
    api.upload_file(
      path_or_fileobj=local_path,
      path_in_repo=path_in_repo,
      repo_id=repo_id,
      repo_type=repo_type,
    )

    print(f"Uploaded checkpoint to HuggingFace: {repo_id}")

    if remove_local:
      os.remove(local_path)

    return True

  except Exception as e:
    print(f"Failed to upload to HuggingFace: {e}")
    return False


def get_hf_repo_id(config_path: str = "hf_config.txt") -> Optional[str]:
  """
  Read HuggingFace repository ID from config file.

  Args:
      config_path: Path to the config file.

  Returns:
      Repository ID string or None if not found.
  """
  if os.path.exists(config_path):
    with open(config_path, "r") as f:
      return f.read().strip()
  return None


def save_and_upload_checkpoint(
  model,
  optimizer,
  step: int,
  loss: float,
  args: Dict[str, Any],
  hf_config_path: str = "hf_config.txt",
  is_final: bool = False,
) -> None:
  """
  Save checkpoint locally and upload to HuggingFace if configured.

  Args:
      model: The model to save.
      optimizer: The optimizer to save.
      step: Current training step.
      loss: Current loss value.
      args: Training arguments dictionary.
      hf_config_path: Path to HuggingFace config file.
      is_final: Whether this is the final checkpoint (saves with special name).
  """
  if is_final:
    print(f"Saving final checkpoint at step {step}...")
    ckpt_path = "checkpoint_final.pt"
  else:
    print(f"Saving checkpoint at step {step}...")
    ckpt_path = f"checkpoint_step_{step}.pt"

  save_checkpoint(model, optimizer, step, loss, args, ckpt_path)

  repo_id = get_hf_repo_id(hf_config_path)
  if repo_id:
    upload_to_huggingface(ckpt_path, repo_id)
  else:
    print(f"hf_config.txt not found. Run HuggingFace setup cell first!")
    print(f"Checkpoint saved locally at {ckpt_path}")


def find_latest_checkpoint(checkpoint_dir: str = ".") -> Optional[str]:
  """
  Find the latest checkpoint file in a directory.

  Args:
      checkpoint_dir: Directory to search for checkpoints.

  Returns:
      Path to the latest checkpoint or None if not found.
  """
  checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
  if not checkpoints:
    return None

  def get_step(path):
    basename = os.path.basename(path)
    try:
      return int(basename.replace("checkpoint_step_", "").replace(".pt", ""))
    except ValueError:
      return -1

  checkpoints.sort(key=get_step, reverse=True)
  return checkpoints[0] if checkpoints else None
