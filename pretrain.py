import os
import gc
import time
import torch
import traceback
import subprocess
import torch._dynamo.config
import torch._inductor.config

from dataclasses import dataclass

from utils import (
  ModelArgs,
  Llama,
  get_cosine_lr,
  update_lr,
  download_dataset,
  get_batch,
  save_and_upload_checkpoint,
  load_checkpoint,
  find_latest_checkpoint,
  init_wandb,
  log_metrics,
  finish_wandb,
)


@dataclass
class TrainArgs:
  data_dir: str = "data_bin"
  batch_size: int = 64
  block_size: int = 2048
  grad_accum: int = 8
  lr: float = 3e-4
  max_iters: int = 5725
  warmup_iters: int = 900
  device: str = "cuda"
  wandb_project: str = "smol-llama"
  checkpoint_interval: int = 200  # Save every 'n' steps
  resume: bool = True  # Auto-resume from latest checkpoint
  log_interval: int = 1  # Log to wandb every 'n' steps


def evaluate(model, args, num_batches = 20):
  model.eval()
  total_loss = 0.0

  with torch.no_grad():
    for _ in range(num_batches):
      X, Y = get_batch("val", args.data_dir, args.batch_size, args.block_size, args.device)
      _, loss = model(X, Y)
      total_loss += loss.item()

  avg_loss = total_loss / num_batches
  perplexity = torch.exp(torch.tensor(avg_loss)).item()

  model.train()
  return avg_loss, perplexity


def train():
  args = TrainArgs()
  download_dataset()

  model_args = ModelArgs()
  wandb_enabled = init_wandb(
    project=args.wandb_project,
    config={
      "train": vars(args),
      "model": vars(model_args),
    },
    resume=args.resume,
  )

  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"CUDA memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

  model = Llama(model_args).to(args.device)
  print(f"Model created on {args.device} ({model.num_parameters():,} parameters)")

  model = model.to(torch.bfloat16)
  model.freqs_cos = model.freqs_cos.to(torch.float32)
  model.freqs_sin = model.freqs_sin.to(torch.float32)

  try:
    torch._dynamo.config.suppress_errors = True
    torch._inductor.config.triton.cudagraphs = False
    model = torch.compile(model, mode="default")
  except Exception as e:
    print(f"Compilation failed: {e}")

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.95),
    fused=True,
  )

  # Resume from checkpoint if available
  start_step = 0
  if args.resume:
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
      print(f"Found checkpoint: {latest_ckpt}")
      print("Resuming training...")
      ckpt = load_checkpoint(latest_ckpt, device=args.device)
      try:
        model.load_state_dict(ckpt["model"])
      except RuntimeError:
        model._orig_mod.load_state_dict(ckpt["model"])
      optimizer.load_state_dict(ckpt["optimizer"])
      start_step = ckpt["step"] + 1
      print(f"Resumed from step {ckpt['step']} (loss: {ckpt['loss']:.4f})")
    else:
      print("No checkpoint found, starting from scratch.")

  print(f"Training for {args.max_iters} steps with {args.grad_accum} gradient accumulation steps")
  if start_step > 0:
    print(f"Resuming from step {start_step}")

  t0 = time.time()

  for step in range(start_step, args.max_iters):
    if step == start_step and start_step == 0:
      print(f"Step {step}: Starting...")

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(args.grad_accum):
      X, Y = get_batch("train", args.data_dir, args.batch_size, args.block_size, args.device)

      try:
        _, loss = model(X, Y)
        loss = loss / args.grad_accum
        loss.backward()
        loss_accum += loss.item()

        if step == 0 and micro_step == 0:
          print(f"First forward/backward pass completed! Loss: {loss.item():.4f}")
          if torch.cuda.is_available():
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
      except RuntimeError as e:
        if "out of memory" in str(e):
          print(f"CUDA OOM at step {step}, micro_step {micro_step}")
          torch.cuda.empty_cache()
          gc.collect()
          raise
        else:
          print(f"ERROR at step {step}, micro_step {micro_step}: {type(e).__name__}: {e}")
          traceback.print_exc()
          raise

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_cosine_lr(step, args.lr, args.warmup_iters, args.max_iters)
    update_lr(optimizer, lr)
    optimizer.step()

    if step == 0:
      print(f"Step {step}: Optimizer step completed, accumulated loss: {loss_accum:.4f}")

    if step % args.log_interval == 0 or step == 0:
      dt = time.time() - t0
      t0 = time.time()

      if step > 0:
        tps = args.batch_size * args.block_size * args.grad_accum / dt
        print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.6f} | tps {tps:,.0f}")

        if wandb_enabled:
          log_metrics(
            {
              "train/loss": loss_accum,
              "train/lr": lr,
              "train/tps": tps,
              "train/step": step,
            },
            step=step,
          )
      else:
        print(f"step {step:5d} | loss {loss_accum:.4f} | (warming up...)")
        if wandb_enabled:
          log_metrics({"train/loss": loss_accum}, step=step)

    # Eval
    if step % args.checkpoint_interval == 0:
      val_loss, val_pplx = evaluate(model, args)
      print(f"val_loss: {val_loss:.4f} | val_pplx: {val_pplx:.2f}")

      if wandb_enabled:
        log_metrics({"val/loss": val_loss, "val/pplx": val_pplx}, step=step)

    # Save checkpoint
    if step % args.checkpoint_interval == 0 and step > 0:
      save_and_upload_checkpoint(model, optimizer, step, loss_accum, vars(args))

  save_and_upload_checkpoint(model, optimizer, args.max_iters, loss_accum, vars(args), is_final=True)
  print("Final checkpoint saved and uploaded!")

  if wandb_enabled:
    finish_wandb()

  # Terminate RunPod instance
  pod_id = os.environ.get("RUNPOD_POD_ID")
  if pod_id:
    subprocess.run(["runpodctl", "remove", "pod", pod_id])
  else:
    print("RUNPOD_POD_ID not set, skipping instance termination")


if __name__ == "__main__":
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  train()
