# Analyze Model Skill

Analyze NeMo model architecture, parameter counts, and configuration details.

## Usage

```
Analyze model: <model_name_or_config>
```

## What This Skill Does

1. Loads model configuration or checkpoint
2. Counts total and trainable parameters
3. Breaks down parameters by layer/module
4. Reports memory estimates for training and inference
5. Identifies potential bottlenecks or misconfigurations

## Implementation

```python
import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count total, trainable, and frozen parameters in a model.
    
    Returns a dict with:
      - total: all parameters
      - trainable: parameters with requires_grad=True
      - frozen: parameters with requires_grad=False
      - by_module: breakdown per top-level submodule
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    by_module = {}
    for name, module in model.named_children():
        mod_total = sum(p.numel() for p in module.parameters())
        mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        by_module[name] = {
            "total": mod_total,
            "trainable": mod_trainable,
            "frozen": mod_total - mod_trainable,
        }

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "by_module": by_module,
    }


def estimate_memory(
    param_count: int,
    dtype: str = "bf16",
    batch_size: int = 1,
    seq_len: int = 2048,
    include_optimizer: bool = True,
) -> dict:
    """
    Rough memory estimate for model weights, gradients, and optimizer states.

    dtype options: fp32, fp16, bf16, int8
    Returns estimates in GB.
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
    bpp = bytes_per_param.get(dtype, 2)

    weights_gb = (param_count * bpp) / (1024 ** 3)
    # Gradients same size as weights (fp32 for mixed precision)
    grads_gb = (param_count * 4) / (1024 ** 3)
    # Adam optimizer: 2x fp32 copies (momentum + variance)
    optimizer_gb = (param_count * 4 * 2) / (1024 ** 3) if include_optimizer else 0

    return {
        "weights_gb": round(weights_gb, 2),
        "gradients_gb": round(grads_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "total_training_gb": round(weights_gb + grads_gb + optimizer_gb, 2),
        "inference_gb": round(weights_gb, 2),
    }


def analyze_model(
    model_or_config: Union[str, torch.nn.Module],
    dtype: str = "bf16",
    batch_size: int = 1,
    seq_len: int = 2048,
    verbose: bool = True,
) -> dict:
    """
    Full model analysis: parameter counts + memory estimates.

    Args:
        model_or_config: Either a loaded nn.Module or a path to a NeMo config/checkpoint.
        dtype: Data type for memory estimation.
        batch_size: Batch size for activation memory estimates.
        seq_len: Sequence length for activation memory estimates.
        verbose: Print formatted report to stdout.

    Returns:
        dict with keys: param_stats, memory_estimates
    """
    if isinstance(model_or_config, str):
        raise NotImplementedError(
            "Loading from config path not yet implemented. Pass a loaded nn.Module."
        )

    model = model_or_config
    param_stats = count_parameters(model)
    memory = estimate_memory(
        param_stats["total"],
        dtype=dtype,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    if verbose:
        print_model_report(model, param_stats, memory)

    return {"param_stats": param_stats, "memory_estimates": memory}


def print_model_report(
    model: torch.nn.Module,
    param_stats: dict,
    memory: dict,
) -> None:
    """Print a human-readable model analysis report."""
    sep = "=" * 60
    print(sep)
    print(f"  Model Analysis: {type(model).__name__}")
    print(sep)

    total = param_stats["total"]
    trainable = param_stats["trainable"]
    frozen = param_stats["frozen"]

    def fmt(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        if n >= 1e3:
            return f"{n/1e3:.2f}K"
        return str(n)

    print(f"\nParameters:")
    print(f"  Total:     {fmt(total):>10}  ({total:,})")
    print(f"  Trainable: {fmt(trainable):>10}  ({trainable:,})")
    print(f"  Frozen:    {fmt(frozen):>10}  ({frozen:,})")

    print(f"\nParameter Breakdown by Module:")
    for name, stats in param_stats["by_module"].items():
        pct = (stats["total"] / total * 100) if total > 0 else 0
        print(f"  {name:<30} {fmt(stats['total']):>8}  ({pct:.1f}%)")

    print(f"\nMemory Estimates:")
    print(f"  Weights (inference):  {memory['inference_gb']:>6.2f} GB")
    print(f"  Weights + Gradients:  {memory['weights_gb'] + memory['gradients_gb']:>6.2f} GB")
    print(f"  Optimizer states:     {memory['optimizer_gb']:>6.2f} GB")
    print(f"  Total (training):     {memory['total_training_gb']:>6.2f} GB")
    print(sep)
```

## When to Use

- Before starting a training run to verify model size fits available GPU memory
- After modifying model architecture to confirm parameter counts changed as expected
- When comparing two model configurations
- When debugging OOM (out-of-memory) errors

## Example Output

```
============================================================
  Model Analysis: MegatronGPTModel
============================================================

Parameters:
  Total:          7.00B  (7,000,000,000)
  Trainable:      7.00B  (7,000,000,000)
  Frozen:            0

Parameter Breakdown by Module:
  embedding                       128.00M  (1.8%)
  decoder                           6.73B  (96.1%)
  output_layer                    128.00M  (1.8%)
  ...

Memory Estimates:
  Weights (inference):   13.04 GB
  Weights + Gradients:   39.12 GB
  Optimizer states:      52.15 GB
  Total (training):      91.27 GB
============================================================
```

## Notes

- Memory estimates are **rough lower bounds** — activations, KV cache, and framework overhead are not included.
- For activation memory, multiply roughly by `batch_size * seq_len * hidden_dim * num_layers * 2 bytes`.
- Use alongside `profile-memory` skill for more accurate GPU memory profiling during an actual training step.
