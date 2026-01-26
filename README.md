# smol-llama 🦙

A minimal, from-scratch implementation of a LLaMA-style language model for pre-training on custom data.

## Model Architecture

| Component | Value |
|-----------|-------|
| **Parameters** | ~360M |
| **Hidden dimension** | 960 |
| **Layers** | 32 |
| **Attention heads** | 15 (Query) / 5 (KV) |
| **Context length** | 2048 |
| **Vocab size** | 49,152 |

**Key features:**
- Grouped Query Attention (GQA) for efficient inference
- RoPE (Rotary Position Embeddings)
- RMSNorm (instead of LayerNorm)
- SwiGLU activation in FFN
- Flash Attention 2 (with fallback to SDPA)
- Gradient checkpointing for memory efficiency
- `torch.compile` for faster training

## Project Structure

```
smol-llama/
├── pretrain.py              # Main training script
├── utils/
│   ├── model.py             # LLaMA model architecture
│   ├── rotary.py            # Rotary position embeddings
│   ├── data.py              # Data loading utilities
│   ├── checkpoint.py        # Checkpoint save/load + HF upload
│   ├── lr_schedule.py       # Cosine LR schedule with warmup
│   └── logging.py           # Weights & Biases integration
├── notebooks/
│   └── 1-train-tokenizer.ipynb  # Tokenizer training notebook
├── data_bin/                # Downloaded training data
└── hf_config.txt            # HuggingFace repo ID for uploads
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Set up HuggingFace upload (optional)

Create a `.env` file with your HuggingFace token:
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=abcdef123456
```

Set the target repository in `hf_config.txt`:
```
your-username/smol-llama
```

### 3. Run training

```bash
uv run ./pretrain.py
```

The script will:
1. Download the pre-tokenized dataset from HuggingFace
2. Initialize the model and optimizer
3. Train with gradient accumulation and mixed precision
4. Save checkpoints every 200 steps (uploaded to HF if configured)
5. Log metrics to Weights & Biases

## Training Configuration

Edit `pretrain.py` to customize training:

```python
@dataclass
class TrainArgs:
    data_dir: str = "data_bin"
    batch_size: int = 64
    block_size: int = 2048           # Context length
    grad_accum: int = 8              # Effective batch = 64 × 8 = 512
    lr: float = 3e-4                 # Peak learning rate
    max_iters: int = 5725            # ~6B tokens (1 epoch)
    warmup_iters: int = 900          # Linear warmup steps
    checkpoint_interval: int = 200   # Save every N steps
    resume: bool = True              # Auto-resume from checkpoint
```

**Tokens per step:** `batch_size × block_size × grad_accum = ~1M tokens`

## Hardware & Cost

| Resource | Spec |
|----------|------|
| **GPU** | 1× NVIDIA H100 (80GB PCIe) |
| **Training speed** | ~75,000 tokens/sec |
| **Time per epoch** | ~22 hours |
| **Cloud cost** | ~$2.40/hr on [RunPod](https://runpod.io) |
| **Total cost (1 epoch)** | ~$53 |

The training script auto-terminates the RunPod instance when complete (if `RUNPOD_POD_ID` env is set).

## Dataset

The model is trained on [weights-and-wires/fineweb-6b](https://huggingface.co/datasets/weights-and-wires/fineweb-6b), a pre-tokenized subset of FineWeb containing ~6B tokens.

The dataset includes:
- `tokenized/train.bin` - 11.3 GB of training tokens
- `tokenized/val.bin` - 57 MB of validation tokens
- Tokenizer files (vocab size: 49,152)

## Using the Pre-trained Model

The trained model is available at [weights-and-wires/smol-llama](https://huggingface.co/weights-and-wires/smol-llama).

### Quick Inference (HuggingFace Format)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "weights-and-wires/smol-llama",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("weights-and-wires/smol-llama")

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Loading Custom Checkpoints

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast
from utils.model import Llama, ModelArgs

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="weights-and-wires/smol-llama",
    filename="training_checkpoints/checkpoint_step_5000.pt",
    repo_type="model"
)

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("weights-and-wires/smol-llama")

# Create model and load weights
model = Llama(ModelArgs()).cuda().to(torch.bfloat16)
ckpt = torch.load(checkpoint_path, map_location="cuda")

# Handle torch.compile prefix
state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt['model'].items()}
model.load_state_dict(state_dict)
model.eval()

# Generate
def generate(prompt, max_tokens=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(input_ids[:, -2048:])
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0])

print(generate("The meaning of life is"))
```

## Checkpoints

Training checkpoints are saved every 200 steps to HuggingFace at [weights-and-wires/smol-llama](https://huggingface.co/weights-and-wires/smol-llama):

| Checkpoint | Steps | Tokens Seen |
|------------|-------|-------------|
| `checkpoint_step_200.pt` | 200 | ~200M |
| `checkpoint_step_400.pt` | 400 | ~400M |
| ... | ... | ... |
| `checkpoint_step_5000.pt` | 5,000 | ~5B |

The final model is available as `model.safetensors` in the root of the repository.

## Training a Custom Tokenizer

See `notebooks/1-train-tokenizer.ipynb` for training a byte-level BPE tokenizer on your own data.

## Resources

- **Pre-trained Model:** [weights-and-wires/smol-llama](https://huggingface.co/weights-and-wires/smol-llama)
- **Training Data:** [weights-and-wires/fineweb-6b](https://huggingface.co/datasets/weights-and-wires/fineweb-6b)
- **Reference:** [HuggingFaceTB/SmolLM-360M](https://huggingface.co/HuggingFaceTB/SmolLM-360M)

## License

MIT
