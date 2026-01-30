"""Rotary Position Embedding utilities for LLaMA-style models."""

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  """
  Precompute the cosine and sine frequencies for rotary position embeddings.

  Args:
      dim: The dimension of the head (head_dim).
      end: The maximum sequence length.
      theta: The base frequency for RoPE (default: 10000.0).

  Returns:
      Tuple of (freqs_cos, freqs_sin) tensors of shape [end, dim//2].
  """
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
  t = torch.arange(end)
  freqs = torch.outer(t, freqs)
  # Return cos and sin separately instead of complex tensor
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis.real, freqs_cis.imag


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
  """
  Apply rotary position embeddings to query and key tensors.

  Args:
      xq: Query tensor of shape [B, T, H, D].
      xk: Key tensor of shape [B, T, H, D].
      freqs_cos: Cosine frequencies of shape [1, T, 1, D/2].
      freqs_sin: Sine frequencies of shape [1, T, 1, D/2].

  Returns:
      Tuple of (xq_out, xk_out) with rotary embeddings applied.
  """
  # Reshape to separate real/imag pairs
  xq_ = xq.reshape(*xq.shape[:-1], -1, 2)
  xk_ = xk.reshape(*xk.shape[:-1], -1, 2)

  freqs_cos = freqs_cos.to(xq.dtype)
  freqs_sin = freqs_sin.to(xq.dtype)

  # Apply rotation using real arithmetic
  xq_r = xq_[..., 0]
  xq_i = xq_[..., 1]
  xk_r = xk_[..., 0]
  xk_i = xk_[..., 1]

  xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
  xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
  xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
  xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

  xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
  xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)

  return xq_out, xk_out
