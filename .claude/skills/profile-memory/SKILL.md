# Profile Memory Skill

Profile GPU/CPU memory usage during NeMo training runs to identify bottlenecks and optimize memory efficiency.

## Usage

```
Profile memory usage for a training run
```

## Steps

### 1. Setup Memory Profiling

Enable memory profiling in the training config:

```python
# In your experiment config
trainer:
  enable_progress_bar: true

model:
  # Enable activation checkpointing to reduce memory
  activations_checkpoint_method: uniform
  activations_checkpoint_num_layers: 1
```

### 2. Run with PyTorch Memory Profiler

```bash
# Profile with torch.cuda.memory_stats
python -c "
import torch
import subprocess
import json

# Start training with memory tracking
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# Run training for a few steps
result = subprocess.run([
    'python', 'examples/nlp/language_modeling/megatron_gpt_pretraining.py',
    '--config-path=conf',
    '--config-name=megatron_gpt_config',
    'trainer.max_steps=10',
    'trainer.val_check_interval=10',
], capture_output=True, text=True)

print(result.stdout)
print(result.stderr)

mem_stats = torch.cuda.memory_stats()
print(f\"Peak memory allocated: {mem_stats['allocated_bytes.all.peak'] / 1e9:.2f} GB\")
print(f\"Peak memory reserved: {mem_stats['reserved_bytes.all.peak'] / 1e9:.2f} GB\")
"
```

### 3. Analyze Memory Timeline

```python
# scripts/profile_memory.py
import torch
from torch.profiler import profile, ProfilerActivity, record_function

def profile_training_step(model, batch, device='cuda'):
    """Profile a single training step for memory usage."""
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("training_step"):
            loss = model.training_step(batch, 0)
            loss.backward()
    
    # Export memory timeline
    prof.export_memory_timeline("memory_timeline.html", device=device)
    
    # Print top memory consumers
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=20
    ))
    
    return prof
```

### 4. Check Memory Breakdown

```bash
# Get detailed memory breakdown
python << 'EOF'
import torch

def print_memory_breakdown():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    for i in range(torch.cuda.device_count()):
        print(f"\n=== GPU {i}: {torch.cuda.get_device_name(i)} ===")
        
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(i) / 1e9
        max_reserved = torch.cuda.max_memory_reserved(i) / 1e9
        
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        print(f"  Total VRAM:      {total:.2f} GB")
        print(f"  Allocated:       {allocated:.2f} GB ({100*allocated/total:.1f}%)")
        print(f"  Reserved:        {reserved:.2f} GB ({100*reserved/total:.1f}%)")
        print(f"  Peak Allocated:  {max_allocated:.2f} GB ({100*max_allocated/total:.1f}%)")
        print(f"  Peak Reserved:   {max_reserved:.2f} GB ({100*max_reserved/total:.1f}%)")
        print(f"  Free:            {total - reserved:.2f} GB")

print_memory_breakdown()
EOF
```

### 5. Identify OOM Issues

Common causes of OOM errors and fixes:

| Issue | Symptom | Fix |
|-------|---------|-----|
| Batch size too large | OOM at first step | Reduce `model.micro_batch_size` |
| Sequence length too long | OOM with long inputs | Reduce `model.encoder_seq_length` |
| No activation checkpointing | OOM mid-training | Enable `activations_checkpoint_method` |
| Large vocab embedding | High baseline memory | Use `share_token_embeddings: true` |
| Optimizer states | 3x model size in memory | Use CPU offloading or ZeRO |

### 6. Memory Optimization Checklist

```yaml
# Recommended memory optimizations in config
model:
  # Reduce precision
  precision: bf16
  
  # Activation checkpointing
  activations_checkpoint_method: block  # or uniform
  activations_checkpoint_num_layers: 1
  
  # Sequence parallelism (for large models)
  sequence_parallel: true
  
  # Tensor parallelism
  tensor_model_parallel_size: 2  # Splits model across GPUs
  
  # Pipeline parallelism  
  pipeline_model_parallel_size: 1

optimizer:
  # Use fused optimizer to reduce memory overhead
  name: fused_adam
```

### 7. Report Memory Profile

After profiling, document findings:

```
Memory Profile Report
====================
Model: <model_name>
GPU: <gpu_type> x <num_gpus>
Config: TP=<tp> PP=<pp> DP=<dp>

Memory Usage:
- Model params: X.XX GB
- Optimizer states: X.XX GB  
- Activations: X.XX GB
- Peak total: X.XX / Y.YY GB (ZZ%)

Bottlenecks:
- <identified bottleneck 1>
- <identified bottleneck 2>

Recommendations:
- <optimization 1>
- <optimization 2>
```

## Tips

- Always profile with `torch.cuda.reset_peak_memory_stats()` before the run
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
- For multi-node runs, profile each rank separately with `LOCAL_RANK` env var
- Memory timeline HTML files can be viewed in Chrome's `chrome://tracing`
