# Dynamic Voice Agent Evaluation

Evaluation system with dynamic system prompt updates and response latency measurement.

## Features

- **Dynamic Prompts**: Update system prompts during evaluation without restarting agents
- **Response Latency**: Accurate measurement from evaluator stop → target first audio frame
- **Multiple Scenarios**: Run multiple test scenarios in a single session
- **Audio Recording**: Record stereo WAV files (evaluator=left, target=right) with automatic resampling
- **segLST Format**: Segment list files with precise timing and speaker labels
- **RTVI-Native**: Uses RTVI action protocol for control
- **Distributed**: Works with agents on different machines
- **Comprehensive Metrics**: Turn counts, latency statistics (mean, median, P95), conversation logs

## Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│   Evaluator     │◄──────────────────►│   Evaluation    │
│   Agent         │    Audio + RTVI    │   Bridge        │
│ (Simulated User)│    Control Msgs    │  (Monitor +     │
│                 │    (Bidirectional) │   Controller)   │
└─────────────────┘                    └─────────────────┘
                                               ▲
                                               │
                                               │ Audio + RTVI
                                               │ Control Msgs
                                               │ (Bidirectional)
                                               ▼
                                       ┌─────────────────┐
                                       │   Target        │
                                       │   Agent         │
                                       │ (Being Tested)  │
                                       └─────────────────┘

Audio Flow:
1. Evaluator speaks → Bridge monitors & forwards → Target receives
2. Target responds → Bridge monitors & forwards → Evaluator receives
3. Bridge measures latency: Time from (1) stops to (2) starts

Metrics Collected:
- Response latency (evaluator stop → target start)
- Turn counts and transcripts
- Latency statistics (mean, median, P95, min, max)
```

## Quick Start

### 1. Start the Agents

**Terminal 1 - Evaluator Agent:**
```bash
cd examples/voice_agent
export SERVER_CONFIG_PATH="evaluation/configs/evaluator_config.yaml"
export WEBSOCKET_PORT=8765
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH
python server/bot_websocket.py
```

**Terminal 2 - Target Agent:**
```bash
cd examples/voice_agent
export SERVER_CONFIG_PATH="evaluation/configs/target_config.yaml"
export WEBSOCKET_PORT=8766
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH
python server/bot_websocket.py
```

### 2. Run Evaluation

**Terminal 3 - Evaluation Bridge:**
```bash
cd examples/voice_agent/evaluation
python dynamic_evaluation_runner.py \
    --evaluator-url ws://localhost:8765 \
    --target-url ws://localhost:8766 \
    --output-dir ./results \
    --duration 60
```

## Usage Examples

### Default Scenarios

Run with built-in test scenarios:
```bash
python dynamic_evaluation_runner.py \
    --evaluator-url ws://localhost:8765 \
    --target-url ws://localhost:8766
```

### Custom Scenarios

Use predefined scenario files:
```bash
# Customer service scenarios
python dynamic_evaluation_runner.py \
    --evaluator-url ws://localhost:8765 \
    --target-url ws://localhost:8766 \
    --scenarios-file scenarios/customer_service.json \
    --output-dir ./results/customer_service

# Technical support scenarios
python dynamic_evaluation_runner.py \
    --evaluator-url ws://localhost:8765 \
    --target-url ws://localhost:8766 \
    --scenarios-file scenarios/technical_support.json \
    --output-dir ./results/tech_support

# Conversation quality scenarios
python dynamic_evaluation_runner.py \
    --evaluator-url ws://localhost:8765 \
    --target-url ws://localhost:8766 \
    --scenarios-file scenarios/conversation_quality.json \
    --output-dir ./results/conversation
```

### Custom Duration and Pause

```bash
python dynamic_evaluation_runner.py \
    --evaluator-url ws://localhost:8765 \
    --target-url ws://localhost:8766 \
    --duration 120 \
    --pause 10
```

### Distributed Evaluation (Different Machines)

```bash
# On Machine 1: Start evaluator (192.168.1.100)
export SERVER_CONFIG_PATH="evaluation/configs/evaluator_config.yaml"
export WEBSOCKET_PORT=8765
export SERVER_HOST=0.0.0.0  # Listen on all interfaces
python server/bot_websocket.py

# On Machine 2: Start target (192.168.1.101)
export SERVER_CONFIG_PATH="evaluation/configs/target_config.yaml"
export WEBSOCKET_PORT=8766
export SERVER_HOST=0.0.0.0
python server/bot_websocket.py

# On Machine 3 (or any machine): Run evaluation
python dynamic_evaluation_runner.py \
    --evaluator-url ws://192.168.1.100:8765 \
    --target-url ws://192.168.1.101:8766 \
    --output-dir ./results/distributed
```

## Creating Custom Scenarios

Create a JSON file with scenario definitions:

```json
[
  {
    "name": "Scenario Name",
    "evaluator_prompt": "System prompt for the evaluator (simulated user)",
    "target_prompt": "Optional: System prompt for the target agent",
    "duration": 90
  },
  {
    "name": "Another Scenario",
    "evaluator_prompt": "Different evaluator behavior",
    "duration": 60
  }
]
```

### Scenario Fields

- **name** (required): Descriptive name for the scenario
- **evaluator_prompt** (required): System prompt for the evaluator agent
- **target_prompt** (optional): System prompt for the target agent. If omitted, target keeps its previous prompt
- **duration** (optional): Duration in seconds for this scenario. If omitted, uses `--duration` default

### Example: Custom Scenario File

```json
[
  {
    "name": "Product Expert Test",
    "evaluator_prompt": "You are testing product knowledge. Ask detailed questions about specific products, features, and comparisons. Start with a specific product question.",
    "target_prompt": "You are a product expert with deep knowledge of all products. Provide detailed, accurate information. Be enthusiastic but professional.",
    "duration": 120
  },
  {
    "name": "Sales Objection Handling",
    "evaluator_prompt": "You are a skeptical customer with concerns about price, quality, or competitors. Raise objections that need to be addressed. Start by expressing a concern about pricing.",
    "target_prompt": "You are a skilled sales agent. Handle objections professionally by acknowledging concerns, providing value justification, and finding solutions. Never be pushy.",
    "duration": 90
  }
]
```

## Output Files

After evaluation, the following files are generated in `--output-dir`:

### 1. Conversation Log (`conversation_YYYYMMDD_HHMMSS.log`)

Real-time conversation transcript with timestamps and latencies:

```
RTVI Evaluation Bridge - Conversation Log
================================================================================
Start Time: 2026-01-27T10:30:00.123456
================================================================================

[10:30:05.234] EVALUATOR: Hello, can you help me?
  → Response latency: 850.2ms
[10:30:06.345] TARGET: Hello! I'd be happy to help. What can I assist you with today?
[10:30:08.456] EVALUATOR: I have a question about your products.
  → Response latency: 720.5ms
[10:30:09.567] TARGET: Of course! Which product are you interested in?
```

### 2. Results JSON (`results_YYYYMMDD_HHMMSS.json`)

Detailed metrics for each scenario:

```json
[
  {
    "scenario_name": "Friendly Conversation",
    "scenario_duration": 60.5,
    "total_turns": 12,
    "latency_stats": {
      "count": 6,
      "mean_ms": 785.3,
      "median_ms": 750.0,
      "p95_ms": 920.0,
      "min_ms": 650.0,
      "max_ms": 950.0
    },
    "latencies": [
      {
        "evaluator_transcript": "Hello, can you help me?",
        "target_transcript": "Hello! I'd be happy to help.",
        "latency_ms": 850.2
      }
    ],
    "turns": [...]
  }
]
```

### 3. Latencies CSV (`latencies_YYYYMMDD_HHMMSS.csv`)

Spreadsheet-friendly latency data:

```csv
Scenario,Evaluator_Transcript,Target_Transcript,Latency_ms
Friendly Conversation,"Hello, can you help me?","Hello! I'd be happy to help.",850.2
Friendly Conversation,"I have a question.","Of course! What's your question?",720.5
```

### 4. Summary Text (`summary_YYYYMMDD_HHMMSS.txt`)

Human-readable summary:

```
EVALUATION SUMMARY
================================================================================

Total Scenarios: 3
Total Duration: 180.5s
Total Turns: 36

Per-Scenario Results:
--------------------------------------------------------------------------------

Friendly Conversation:
  Turns: 12
  Duration: 60.2s
  Turns/min: 12.0
  Latency Measurements: 6
    Mean: 785.3ms
    Median: 750.0ms
    P95: 920.0ms
    Min: 650.0ms
    Max: 950.0ms

Overall Latency Statistics:
--------------------------------------------------------------------------------
  Total Measurements: 18
  Mean: 795.5ms
  Median: 765.0ms
  P95: 930.0ms
  Min: 620.0ms
  Max: 980.0ms
```

### 5. Audio Files (`audio/YYYYMMDD_HHMMSS_XX_ScenarioName.wav`)

Stereo WAV files with evaluator and target audio:

```
audio/
├── 20260127_103000_01_Friendly_Conversation.wav
├── 20260127_103000_01_Friendly_Conversation.seglst
├── 20260127_103000_02_Challenging_Questions.wav
└── 20260127_103000_02_Challenging_Questions.seglst
```

**Audio Format:**
- **Channels**: 2 (stereo)
  - Left channel: Evaluator audio
  - Right channel: Target audio
- **Sample rate**: Configurable (default: 16000 Hz)
- **Bit depth**: 16-bit PCM
- **Resampling**: Automatic if evaluator and target use different sample rates

**To disable audio recording:**
```bash
python dynamic_evaluation_runner.py --no-audio
```

**To change output sample rate:**
```bash
python dynamic_evaluation_runner.py --output-sample-rate 48000
```

### 6. segLST Files (`audio/*.seglst`)

Segment list format with precise timing and speaker labels:

```
# segLST format: start_time end_time speaker transcript
# Audio file: 20260127_103000_01_Friendly_Conversation.wav
# Sample rate: 16000 Hz
#
0.000 2.500 evaluator Hello, can you help me?
3.350 6.550 target Hello! I'd be happy to help. What can I assist you with today?
7.300 9.400 evaluator I have a question about your products.
10.150 13.350 target Of course! Which product are you interested in?
```

**Format Details:**
- **start_time**: Segment start time in seconds (relative to audio file start)
- **end_time**: Segment end time in seconds
- **speaker**: "evaluator" or "target"
- **transcript**: Complete utterance text

**Use Cases:**
- **Audio analysis**: Load WAV with timing annotations
- **Forced alignment**: Pre-segmented data for training
- **Quality assessment**: Correlate audio with transcripts
- **Visualization**: Create timeline visualizations

**Loading segLST in Python:**
```python
def load_seglst(path):
    segments = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split(maxsplit=3)
            start, end, speaker, text = float(parts[0]), float(parts[1]), parts[2], parts[3]
            segments.append({'start': start, 'end': end, 'speaker': speaker, 'text': text})
    return segments
```

## Response Latency Measurement

The bridge measures **true response latency** from the evaluator's perspective:

1. **Evaluator Stops Speaking**: Tracks when evaluator sends last audio frame or `bot-stopped-speaking` event
2. **Target Starts Responding**: Tracks when evaluator receives first audio frame from target
3. **Latency = Target Start - Evaluator Stop**

This includes:
- Target's STT processing time
- Target's LLM inference time
- Target's TTS generation time
- Network transmission time (if distributed)

### Latency Statistics

For each scenario and overall:
- **Mean**: Average latency across all responses
- **Median**: Middle value (50th percentile)
- **P95**: 95th percentile (only 5% of responses are slower)
- **Min/Max**: Fastest and slowest responses

## Transcript Handling

The bridge correctly handles **incremental transcripts**:

### How It Works

1. **Segments Arrive Incrementally**: As TTS generates text, `bot-transcription` messages are sent for each segment (not complete sentences)
   - Example: "Hello" → "Hello, how" → "Hello, how can" → "Hello, how can I help"

2. **Accumulation**: The bridge accumulates all segments for the current utterance

3. **Turn Finalization**: When `bot-stopped-speaking` is received, the bridge:
   - Finalizes the complete transcript
   - Creates a single turn entry
   - Logs the complete utterance
   - Clears the accumulator for the next turn

### Example Timeline

```
Time  | Event                        | Action
------|------------------------------|----------------------------------------
10:00 | bot-transcription: "Hello"   | Accumulate: "Hello"
10:01 | bot-transcription: ", how"   | Accumulate: "Hello, how"
10:02 | bot-transcription: " can"    | Accumulate: "Hello, how can"
10:03 | bot-transcription: " I help" | Accumulate: "Hello, how can I help"
10:04 | bot-stopped-speaking         | Finalize turn: "Hello, how can I help"
                                     | Log complete utterance
                                     | Clear accumulator
```

This ensures:
- **One turn per utterance**: Not one turn per segment
- **Complete transcripts**: Full sentences in logs
- **Accurate timing**: Turn timestamp reflects when speaking stopped

## Configuration

### Agent Configurations

Modify `configs/evaluator_config.yaml` and `configs/target_config.yaml` to:
- Change LLM models
- Adjust generation parameters (temperature, max_tokens)
- Configure STT/TTS models
- Set GPU devices for distributed load

### Example: Use More Powerful Evaluator

```yaml
# configs/evaluator_config.yaml
llm:
  model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"  # Larger model
  vllm_server_params:
    tensor_parallel_size: 2  # Use 2 GPUs
    gpu_memory_utilization: 0.90
  vllm_generation_params:
    temperature: 0.9  # More creative
    max_tokens: 512
```

### Example: Optimize Target for Speed

```yaml
# configs/target_config.yaml
llm:
  model: "nvidia/NVIDIA-Nemotron-Nano-9B-v2"  # Faster model
  vllm_server_params:
    gpu_memory_utilization: 0.95
    max_model_len: 4096  # Shorter context
  vllm_generation_params:
    temperature: 0.7
    max_tokens: 256  # Shorter responses
```

## Troubleshooting

### Connection Issues

**Error: Connection refused**
```bash
# Check agents are running
curl http://localhost:8765
curl http://localhost:8766

# Check ports are correct
netstat -tuln | grep 876
```

**Error: Connection timeout**
```bash
# For distributed setup, check firewall
sudo ufw allow 8765
sudo ufw allow 8766

# Check SERVER_HOST is set to 0.0.0.0
echo $SERVER_HOST
```

### No Latency Measurements

If latency count is 0:
- Check that evaluator is actually speaking (check conversation log)
- Verify audio is being transmitted (check for binary frames in debug logs)
- Try increasing scenario duration
- Check that both agents are responding (not stuck)

### Out of Memory

```bash
# Reduce model sizes or context length
# In config YAML:
vllm_server_params:
  gpu_memory_utilization: 0.80  # Lower from 0.90
  max_model_len: 4096  # Lower from 8192

# Or use smaller models
llm:
  model: "nvidia/NVIDIA-Nemotron-Nano-9B-v2"  # Instead of 30B
```

### Slow Response Times

High latencies can be caused by:
- **Large models**: Use smaller or quantized models
- **CPU inference**: Ensure models are on GPU (`device: "cuda:0"`)
- **Low GPU memory**: Increase `gpu_memory_utilization` to 0.90-0.95
- **Long prompts**: Reduce `max_tokens` in generation params
- **Network latency**: For distributed, check network with `ping`

## Implementation Details

### Pipecat Integration

The bridge uses **pipecat's RTVI message types** directly instead of hardcoded strings:

```python
from pipecat.processors.frameworks.rtvi import (
    RTVIBotStoppedSpeakingMessage,
    RTVIBotStartedSpeakingMessage,
    RTVIBotTranscriptionMessage,
)

# Constants automatically adapt to pipecat API changes
RTVI_BOT_STOPPED_SPEAKING = RTVIBotStoppedSpeakingMessage().type
RTVI_BOT_STARTED_SPEAKING = RTVIBotStartedSpeakingMessage().type
RTVI_BOT_TRANSCRIPTION = RTVIBotTranscriptionMessage(data=...).type
```

**Benefits:**
- ✅ **Future-proof**: Automatically adapts if pipecat changes message type strings
- ✅ **Type-safe**: Uses pipecat's actual message classes
- ✅ **Maintainable**: Single source of truth for message types
- ✅ **No magic strings**: All message types come from pipecat

### Architecture Design

The bridge acts as a **transparent proxy** with monitoring:

```
Evaluator ──► Bridge ──► Target
           │  (monitor)
           │  - Track timing
           │  - Accumulate transcripts
           │  - Calculate latency
           │
Evaluator ◄── Bridge ◄── Target
           │  (forward)
```

**Key Design Decisions:**

1. **No Pipeline Modification**: Works with standard voice agent servers
2. **WebSocket-Based**: Supports distributed deployment
3. **Non-Invasive Monitoring**: Doesn't modify messages, only observes
4. **Incremental Transcript Handling**: Accumulates segments correctly

## Advanced Usage

### Programmatic API

Use the bridge directly in Python:

```python
from rtvi_evaluation_bridge import RTVIEvaluationBridge

async def custom_evaluation():
    bridge = RTVIEvaluationBridge(
        evaluator_url="ws://localhost:8765",
        target_url="ws://localhost:8766",
        log_file="./my_eval.log"
    )

    await bridge.connect()

    # Update prompts dynamically
    await bridge.update_evaluator_prompt("New prompt here")
    await bridge.update_target_prompt("Another prompt")

    # Run for 60 seconds
    await bridge.route_audio(duration=60)

    # Get metrics
    metrics = bridge.get_metrics()
    print(f"Latency: {metrics['latency_stats']['mean_ms']:.1f}ms")

    await bridge.disconnect()

asyncio.run(custom_evaluation())
```

### A/B Testing

Compare two configurations:

```bash
# Test configuration A
python dynamic_evaluation_runner.py \
    --target-url ws://localhost:8766 \
    --scenarios-file scenarios/my_test.json \
    --output-dir ./results/config_A

# Change target config, restart target agent

# Test configuration B
python dynamic_evaluation_runner.py \
    --target-url ws://localhost:8766 \
    --scenarios-file scenarios/my_test.json \
    --output-dir ./results/config_B

# Compare results
diff results/config_A/summary_*.txt results/config_B/summary_*.txt
```

## Best Practices

1. **Scenario Design**:
   - Keep scenarios focused on specific capabilities
   - Use realistic user behaviors
   - Include both easy and challenging cases
   - Test edge cases and error handling

2. **Duration**:
   - 60-90s for simple scenarios
   - 120-180s for complex multi-turn scenarios
   - Allow enough time for meaningful conversation

3. **Prompt Engineering**:
   - Be specific about desired evaluator behavior
   - Include instructions to "start" the conversation
   - Specify tone, complexity level, and goals
   - For target prompts, define clear role and guidelines

4. **Metrics Analysis**:
   - Focus on P95 latency (more relevant than max)
   - Compare mean vs median to detect outliers
   - Analyze latency trends across scenarios
   - Review conversation logs for quality assessment

5. **Reproducibility**:
   - Save all config files and scenario definitions
   - Document model versions and parameters
   - Run multiple iterations for statistical significance
   - Control for variables (same models, GPUs, etc.)

## Related Documentation

- [README.md](README.md) - Overview of evaluation framework
- [WHICH_APPROACH.md](WHICH_APPROACH.md) - Comparison of evaluation approaches
- [DISTRIBUTED_DEPLOYMENT.md](DISTRIBUTED_DEPLOYMENT.md) - Distributed setup guide
- [RESPONSE_TIME_MEASUREMENT.md](RESPONSE_TIME_MEASUREMENT.md) - Latency measurement details

## Contributing

To add new features:
1. Extend `RTVIEvaluationBridge` class in `rtvi_evaluation_bridge.py`
2. Add new scenario types in `scenarios/`
3. Update this documentation
4. Submit PR following NeMo contribution guidelines

## Citation

If you use this evaluation framework in your research:

```bibtex
@misc{nemo-voice-agent-eval,
  title={Dynamic Voice Agent Evaluation Framework},
  author={NVIDIA NeMo Team},
  year={2026},
  url={https://github.com/NVIDIA/NeMo}
}
```
