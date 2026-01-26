"""LLaMA model architecture components."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

from .rotary import precompute_freqs_cis, apply_rotary_emb

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


@dataclass
class ModelArgs:
    dim: int = 960               # Hidden dimension
    n_layers: int = 32           # Number of layers
    n_heads: int = 15            # Query heads
    n_kv_heads: int = 5          # Key/Value heads (GQA)
    vocab_size: int = 49152      # Tokenizer vocab size
    multiple_of: int = 256       # MLP hidden layer multiple
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        if HAS_FLASH_ATTN:
            out = flash_attn_func(q, k, v, causal=True)
        else:
            # Fallback to standard attention
            q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2)  # (B, T, n_heads, head_dim)

        out = out.reshape(B, T, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden = int(2 * (4 * args.dim) / 3)
        hidden = args.multiple_of * ((hidden + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden, bias=False)
        self.w3 = nn.Linear(args.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, args.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = CausalSelfAttention(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn = FeedForward(args)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn(self.attn_norm(x), freqs_cos, freqs_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerBlockChunk(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, freqs_cos, freqs_sin):
        for block in self.blocks:
            x = block(x, freqs_cos, freqs_sin)
        return x


class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])

        chunk = 4
        self.layer_chunks = nn.ModuleList([
            TransformerBlockChunk(self.layers[i:i+chunk])
            for i in range(0, args.n_layers, chunk)
        ])

        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Register freqs_cos and freqs_sin as separate float32 buffers
        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        h = self.embed(tokens)
        freqs_cos = self.freqs_cos[:T].unsqueeze(0).unsqueeze(2)
        freqs_sin = self.freqs_sin[:T].unsqueeze(0).unsqueeze(2)

        for chunk in self.layer_chunks:
            h = checkpoint(chunk, h, freqs_cos, freqs_sin, use_reentrant=False)

        h = self.norm(h)
        logits = self.lm_head(h)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return logits, loss

        return logits, None

    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

