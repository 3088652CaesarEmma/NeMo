# Quick Start: Dynamic Evaluation

Get started with dynamic voice agent evaluation in 3 steps.

## Step 1: Start the Agents

### Terminal 1 - Evaluator Agent
```bash
cd examples/voice_agent
export SERVER_CONFIG_PATH="evaluation/configs/evaluator_config.yaml"
export WEBSOCKET_PORT=8765
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH
python server/bot_websocket.py
```

### Terminal 2 - Target Agent
```bash
cd examples/voice_agent
export SERVER_CONFIG_PATH="evaluation/configs/target_config.yaml"
export WEBSOCKET_PORT=8766
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH
python server/bot_websocket.py
```

## Step 2: Run Evaluation

### Option A: Quick Start Script (Easiest)
```bash
cd examples/voice_agent/evaluation
./run_evaluation.sh
```

### Option B: Python Script (More Options)
```bash
cd examples/voice_agent/evaluation
python dynamic_evaluation_runner.py \
    --evaluator-url ws://localhost:8765 \
    --target-url ws://localhost:8766
```

## Step 3: View Results

Results are saved to `./eval_results/` by default:

```bash
# View summary
cat eval_results/summary_*.txt

# View conversation
cat eval_results/conversation_*.log

# View latencies (CSV)
cat eval_results/latencies_*.csv
```

## Common Commands

### Run with Custom Scenarios

```bash
# Customer service
./run_evaluation.sh --scenarios scenarios/customer_service.json

# Technical support
./run_evaluation.sh --scenarios scenarios/technical_support.json

# Conversation quality
./run_evaluation.sh --scenarios scenarios/conversation_quality.json
```

### Run with Custom Duration

```bash
./run_evaluation.sh --duration 120
```

### Run with Custom Output Directory

```bash
./run_evaluation.sh --output-dir ./my_results
```

### Full Example

```bash
./run_evaluation.sh \
    --scenarios scenarios/customer_service.json \
    --duration 90 \
    --output-dir ./results/cs_test_001
```

## What Gets Measured?

### Response Latency
Time from when evaluator stops speaking to when target's first audio arrives:
- Target STT processing
- Target LLM inference
- Target TTS generation
- Network transmission (if distributed)

### Statistics Collected
- **Mean**: Average latency
- **Median**: Middle value (50th percentile)
- **P95**: 95th percentile
- **Min/Max**: Fastest and slowest

### Output Files
1. **conversation_*.log** - Full transcript with timestamps
2. **results_*.json** - Detailed metrics in JSON
3. **latencies_*.csv** - Spreadsheet-friendly latency data
4. **summary_*.txt** - Human-readable summary
5. **audio/*.wav** - Stereo audio recordings (evaluator=left, target=right)
6. **audio/*.seglst** - Segment list with timing and speaker labels

## Scenario Files

Located in `scenarios/`:
- `customer_service.json` - Customer service scenarios
- `technical_support.json` - Tech support scenarios
- `conversation_quality.json` - General conversation tests

### Create Your Own Scenarios

Create a JSON file with:
```json
[
  {
    "name": "Scenario Name",
    "evaluator_prompt": "How the evaluator should behave",
    "target_prompt": "Optional: How the target should behave",
    "duration": 90
  }
]
```

## Troubleshooting

### "Agent not running" error
Make sure both agents are started in separate terminals before running evaluation.

### "Connection refused" error
Check that ports match:
- Evaluator: `WEBSOCKET_PORT=8765`
- Target: `WEBSOCKET_PORT=8766`

### No latency measurements
- Increase `--duration` to allow more conversation
- Check that agents are actually speaking (view conversation log)

### Out of memory
Edit config files to use smaller models or reduce `gpu_memory_utilization`.

## Next Steps

- Read [DYNAMIC_EVALUATION.md](DYNAMIC_EVALUATION.md) for detailed documentation
- Create custom scenarios in `scenarios/`
- Modify agent configs in `configs/`
- Analyze results with the CSV exports

## Example Session

```bash
# Terminal 1
cd examples/voice_agent
export SERVER_CONFIG_PATH="evaluation/configs/evaluator_config.yaml"
export WEBSOCKET_PORT=8765
python server/bot_websocket.py

# Terminal 2
cd examples/voice_agent
export SERVER_CONFIG_PATH="evaluation/configs/target_config.yaml"
export WEBSOCKET_PORT=8766
python server/bot_websocket.py

# Terminal 3
cd examples/voice_agent/evaluation
./run_evaluation.sh --scenarios scenarios/customer_service.json

# View results
cat eval_results/summary_*.txt
```

Expected output:
```
EVALUATION SUMMARY
================================================================================

Total Scenarios: 4
Total Duration: 390.0s
Total Turns: 48

Per-Scenario Results:
--------------------------------------------------------------------------------

Customer Service - Friendly Customer:
  Turns: 12
  Duration: 90.0s
  Turns/min: 8.0
  Latency Measurements: 6
    Mean: 850.5ms
    Median: 830.0ms
    P95: 920.0ms
    Min: 750.0ms
    Max: 950.0ms

[...]

Overall Latency Statistics:
--------------------------------------------------------------------------------
  Total Measurements: 24
  Mean: 870.2ms
  Median: 855.0ms
  P95: 950.0ms
  Min: 720.0ms
  Max: 1020.0ms
```

## Help

For more options:
```bash
./run_evaluation.sh --help
python dynamic_evaluation_runner.py --help
```

For detailed documentation:
```bash
cat DYNAMIC_EVALUATION.md
```
