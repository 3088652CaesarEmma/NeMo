# Benchmark Skill

Run performance benchmarks for NeMo models and training configurations.

## Usage

```
benchmark [model] [config] [options]
```

## Description

This skill automates the process of running performance benchmarks on NeMo models,
collecting metrics like throughput, memory usage, and convergence speed.

## Steps

1. **Setup Environment**
   - Verify GPU availability and CUDA version
   - Check NeMo installation and dependencies
   - Validate benchmark configuration

2. **Prepare Benchmark**
   - Load model configuration
   - Set up data loaders with synthetic or real data
   - Configure logging and metrics collection

3. **Run Warmup**
   - Execute 2-3 warmup iterations to stabilize GPU performance
   - Discard warmup metrics from final results

4. **Execute Benchmark**
   - Run specified number of iterations (default: 20)
   - Collect per-step metrics:
     - Samples per second (throughput)
     - GPU memory utilization
     - Step time (forward + backward)
     - Loss values (for convergence benchmarks)

5. **Collect Results**
   - Aggregate metrics across iterations
   - Compute mean, median, p95, p99 statistics
   - Compare against baseline if provided

6. **Generate Report**
   - Output summary table to stdout
   - Save detailed JSON report to `benchmark_results/`
   - Flag regressions if performance drops > 5% from baseline

## Configuration

```yaml
benchmark:
  iterations: 20
  warmup_iterations: 3
  batch_sizes: [1, 2, 4, 8]
  sequence_lengths: [512, 1024, 2048]
  precision: [fp32, fp16, bf16]
  baseline_file: null  # path to previous results for comparison
  output_dir: benchmark_results/
```

## Metrics Collected

| Metric | Unit | Description |
|--------|------|-------------|
| throughput | samples/sec | Training samples processed per second |
| step_time | ms | Time per training step |
| memory_peak | GB | Peak GPU memory usage |
| memory_reserved | GB | Reserved GPU memory |
| tflops | TFLOPS | Theoretical FLOP/s utilization |

## Example Output

```
=== Benchmark Results: GPT-2 (124M) ===
Config: batch_size=8, seq_len=1024, precision=bf16
GPU: NVIDIA A100 80GB

Metric          Mean      Median    P95       P99
-----------     -------   -------   -------   -------
Throughput      1842      1856      1891      1903    samples/sec
Step Time       4.34      4.31      4.21      4.19    ms
Memory Peak     12.4      12.4      12.5      12.5    GB
TFLOPS          312.4     314.8     321.2     323.1

Comparison vs baseline: +2.3% throughput improvement
```

## Regression Detection

If a baseline file is provided, the skill will:
- Compare current results against baseline metrics
- Flag any metric that regresses more than the threshold (default 5%)
- Output a clear PASS/FAIL status for CI integration

## Notes

- Always run on dedicated hardware without other GPU workloads
- Use `--sync-cuda` flag for accurate timing measurements
- For multi-GPU benchmarks, results reflect aggregate throughput
- Memory metrics are per-GPU unless otherwise noted
