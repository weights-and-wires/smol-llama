"""Logging utilities including wandb integration."""

import os
import wandb
from typing import Optional, Dict, Any


def load_env_token(key: str) -> Optional[str]:
  """
  Load a token from environment or .env file.

  Args:
      key: Environment variable name to look for.

  Returns:
      Token string or None if not found.
  """
  token = os.environ.get(key)
  if token:
    return token

  env_path = ".env"
  if os.path.exists(env_path):
    with open(env_path, "r") as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
          if "=" in line:
            k, value = line.split("=", 1)
            k = k.strip()
            value = value.strip().strip('"').strip("'")
            if k == key:
              os.environ[k] = value
              return value
  return None


def init_wandb(
  project: str = "smol-llama",
  run_name: Optional[str] = None,
  config: Optional[Dict[str, Any]] = None,
  resume: bool = True,
) -> bool:
  """
  Initialize wandb for logging.

  Args:
      project: wandb project name.
      run_name: Optional run name.
      config: Configuration dict to log.
      resume: Whether to resume existing run if available.

  Returns:
      True if wandb initialized successfully, False otherwise.
  """
  try:
    token = load_env_token("WANDB_API_KEY") or load_env_token("WANDB_TOKEN")
    if token:
      wandb.login(key=token)

    wandb.init(
      project=project,
      name=run_name,
      config=config,
      resume="allow" if resume else None,
    )
    print(f"wandb initialized: {wandb.run.url}")
    return True

  except ImportError:
    print("wandb not installed.")
    return False
  except Exception as e:
    print(f"wandb initialization failed: {e}")
    return False


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
  """
  Log metrics to wandb if available.

  Args:
      metrics: Dictionary of metrics to log.
      step: Optional step number.
  """
  try:
    if wandb.run is not None:
      wandb.log(metrics, step=step)
  except ImportError:
    pass


def finish_wandb() -> None:
  """Finish wandb run if active."""
  try:
    if wandb.run is not None:
      wandb.finish()
      print("wandb run finished")
  except ImportError:
    pass
