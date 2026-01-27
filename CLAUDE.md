# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NVIDIA NeMo is a scalable and cloud-native generative AI framework for Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text to Speech (TTS), and Computer Vision (CV) domains. Built on PyTorch and PyTorch Lightning, NeMo is designed to efficiently create, customize, and deploy generative AI models.

**Important**: This repository is pivoting to focus on speech model collections. LLM and VLM support (NeMo 2.0) is deprecated and replaced by [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and [NeMo AutoModel](https://github.com/NVIDIA-NeMo/AutoModel).

## Requirements

- Python 3.10 or above
- PyTorch 2.5 or above
- NVIDIA GPU for model training

## Installation

### From PyPI
```bash
pip install "nemo_toolkit[all]"
```

### From Source (Recommended for Development)
```bash
git clone https://github.com/NVIDIA/NeMo
cd NeMo
pip install '.[all]'
```

### Domain-Specific Installation
```bash
pip install nemo_toolkit['asr']    # ASR only
pip install nemo_toolkit['tts']    # TTS only
pip install nemo_toolkit['multimodal']  # Multimodal only
```

## Common Commands

### Code Style and Linting
```bash
# Check code style (black + isort)
python setup.py style --scope path/to/changed/files

# Auto-fix code style
python setup.py style --scope path/to/changed/files --fix

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Testing

#### Quick Local Tests (without GPU)
```bash
pytest -m "not pleasefixme" --cpu path/to/relevant_tests
```

#### Full Test Suite (with pretrained model downloads)
```bash
pytest -m "not pleasefixme" --with_downloads path/to/relevant_tests
```

#### Run Specific Test Collections
```bash
# ASR tests
pytest tests/collections/asr

# TTS tests
pytest tests/collections/tts

# Core tests
pytest tests/core
```

#### Test Markers
- `unit` - Unit tests for isolated functionality
- `integration` - Integration tests for subsystems
- `system` - High-level system tests
- `pleasefixme` - Broken tests that need fixing (skipped in CI)
- `skipduringci` - Tests addressed by Jenkins but useful for user setups

### Linting Configuration

Different parts of the codebase use different linting configurations:
- `.flake8` - Default flake8 config
- `.flake8.speech` - Speech collection specific
- `.flake8.other` - Other collections
- `.pylintrc`, `.pylintrc.speech`, `.pylintrc.other` - Pylint configs

## Architecture Overview

### Collection-Based Organization

NeMo is organized into **collections** - logical groupings of related neural modules that share a domain:

#### Active Collections (Speech-Focused)
- **nemo/collections/asr/** - Automatic Speech Recognition models (Conformer, FastConformer, Citrinet, etc.)
- **nemo/collections/tts/** - Text-to-Speech synthesis (FastPitch, HiFiGAN, VITS, etc.)
- **nemo/collections/audio/** - Audio processing utilities
- **nemo/collections/multimodal/** - Multimodal models for NeMo 1.0
- **nemo/collections/speechlm/** - Speech language models
- **nemo/collections/speechlm2/** - Next-gen speech language models

#### Deprecated Collections (use previous versions)
- **nemo/collections/nlp/** - NLP models (deprecated in 25.11)
- **nemo/collections/llm/** - LLM models for NeMo 2.0 (deprecated, use Megatron-Bridge)
- **nemo/collections/vlm/** - Vision-Language models (deprecated, use AutoModel)
- **nemo/collections/vision/** - Computer vision models (deprecated)
- **nemo/collections/diffusion/** - Diffusion models (deprecated)

### Core Infrastructure

#### nemo/core/
Fundamental building blocks:
- **classes/** - Base classes for datasets, models, losses, neural modules
- **config/** - Configuration management using OmegaConf/Hydra
- **neural_types/** - Type system for inputs/outputs between modules
- **optim/** - Optimizers and learning rate schedulers
- **connectors/** - Model composition and serialization

#### nemo/lightning/
PyTorch Lightning integration for distributed training:
- **pytorch/trainer.py** - Custom NeMo Trainer wrapping PTL's Trainer
- **pytorch/strategies/megatron_strategy.py** - Strategy for Megatron Core models
- **megatron_parallel.py** - Distributed model parallelism setup (TP/PP/FSDP)
- **pytorch/plugins/mixed_precision.py** - Precision plugins (BF16/FP8)
- **io/** - Checkpointing and model I/O
- **fabric/** - Lightweight training wrapper

#### nemo/export/
Model export and optimization:
- ONNX and TensorRT export
- Model quantization utilities
- VLLM export for LLM serving

#### nemo/deploy/
Model deployment utilities:
- Triton inference server deployment
- REST API service wrappers

### Training Paradigms

#### NeMo 1.0 (Current for Speech)
- YAML-based configuration with Hydra
- Used by ASR, TTS, and Audio collections
- Training scripts in `examples/<domain>/`

#### NeMo 2.0 (Deprecated for LLM/VLM)
- Python-based configuration
- PyTorch Lightning modular abstractions
- NeMo-Run for multi-GPU/multi-node execution
- Recipes in `nemo/collections/llm/recipes/`

### Distributed Training

NeMo supports advanced parallelism strategies:
- **Tensor Parallelism (TP)** - Split model layers across GPUs
- **Pipeline Parallelism (PP)** - Split model stages across GPUs
- **Fully Sharded Data Parallelism (FSDP)** - Shard optimizer states
- **Mixture-of-Experts (MoE)** - Conditional computation
- **Mixed Precision** - BF16/FP8 training

Parallelism is configured through PyTorch Lightning's `MegatronStrategy` in nemo/lightning/.

## Project Structure

```
nemo/
├── collections/        # Domain-specific model collections
│   ├── asr/           # Speech recognition (active)
│   ├── tts/           # Speech synthesis (active)
│   ├── audio/         # Audio processing (active)
│   ├── multimodal/    # Multimodal models (active)
│   ├── speechlm/      # Speech LM (active)
│   ├── llm/           # DEPRECATED - use Megatron-Bridge
│   ├── nlp/           # DEPRECATED
│   └── vlm/           # DEPRECATED - use AutoModel
├── core/              # Base classes and utilities
├── lightning/         # PyTorch Lightning integration
├── export/            # Model export tools
├── deploy/            # Deployment utilities
└── utils/             # General utilities

examples/              # Training and inference examples
├── asr/              # ASR training scripts
├── tts/              # TTS training scripts
├── voice_agent/      # Voice agent demo (STT+LLM+TTS)
└── llm/              # DEPRECATED - use Megatron-Bridge

tests/                # Unit and integration tests
├── collections/      # Collection-specific tests
├── core/             # Core functionality tests
└── functional_tests/ # End-to-end functional tests

scripts/              # Dataset processing and utilities
tools/                # Additional tools (forced aligner, evaluators, etc.)
tutorials/            # Jupyter notebook tutorials
docs/                 # Documentation source
```

## Voice Agent Example

The `examples/voice_agent/` directory contains a complete open-source voice agent combining:
- **ASR**: Streaming FastConformer with EOU detection
- **LLM**: HuggingFace models with vLLM/HF backend
- **TTS**: Multiple TTS backends (Kokoro, FastPitch, Magpie)
- **Diarization**: Speaker identification (up to 4 speakers)
- **Tool Calling**: External tools and behavior control

### Running Voice Agent
```bash
# Set NeMo path
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH

# Optional: HuggingFace token for gated models
export HF_TOKEN="hf_..."

# Optional: Custom server config
export SERVER_CONFIG_PATH="/path/to/config.yaml"

# Start server
python examples/voice_agent/server/server.py

# In another terminal, start client
cd examples/voice_agent/client
npm install
npm run dev
```

Server configs are in `examples/voice_agent/server/server_configs/`:
- `default.yaml` - Default configuration
- `llm_configs/` - LLM-specific configs (Nemotron, Qwen, Llama)
- `tts_configs/` - TTS-specific configs (Kokoro, FastPitch, Magpie)

## Development Workflow

### Before Making Changes
1. Always read files before editing them
2. Check the relevant collection's structure in `nemo/collections/<domain>/`
3. Look at existing examples in `examples/<domain>/`
4. Review tests in `tests/collections/<domain>/`

### Code Style Requirements
- Use `black` (line length 119) for formatting
- Use `isort` for import sorting
- Add docstrings for every class and method exposed to users
- Use Python 3 type hints for all public APIs
- Prefer `RaiseError` over `assert`
- Methods should be atomic (<75 lines)
- Use f-strings over formatted strings
- Use loggers over print: `from nemo.utils import logging`

### Pull Request Guidelines
1. Send PRs to the `main` branch
2. Sign commits with `git commit -s`
3. Ensure relevant unit tests pass before submitting
4. Add "Run CICD" label when ready for CI
5. Tag @nithinraok for NeMo core/ASR PRs, @blisc for TTS PRs

### Testing Before Submitting
```bash
# Run tests for your changes
pytest tests/collections/<domain>/path/to/relevant_tests

# Check code style
python setup.py style --scope path/to/changed/files
```

### CI/CD
- CI tests run when "Run CICD" label is added
- Tests are selective based on changed files
- Lint checks use flake8/pylint on changed files
- Add "skip-linting" label only if absolutely necessary (discouraged)

## Docker Containers

### NGC PyTorch Container (for development)
```bash
docker run --gpus all -it --rm --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:25.09-py3

# Inside container
cd /opt
git clone https://github.com/NVIDIA/NeMo
cd NeMo
bash docker/common/install_dep.sh --library all
pip install ".[all]"
```

### NGC NeMo Container (production-ready)
```bash
docker run --gpus all -it --rm --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/nemo:25.11.01
```

## Key Technical Details

### Configuration System
- NeMo 1.0 uses Hydra/OmegaConf with YAML configs
- NeMo 2.0 (deprecated) uses Python-based configs
- Config resolution in `nemo/core/config/`

### Model Checkpointing
- NeMo models save as `.nemo` files (tar archives containing model weights + config)
- Checkpoint loading/saving in `nemo/core/connectors/`
- Lightning checkpoints handled by `nemo/lightning/io/`

### Neural Types System
- Type checking for inputs/outputs between modules
- Defined in `nemo/core/neural_types/`
- Ensures compatible module connections

### Hydra Integration
- Training scripts use `@hydra.main` decorator
- Configs in `conf/` directories within examples
- Override configs: `python script.py model.optim.lr=0.001`

## Useful Resources

- [Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
- [NeMo 2.0 Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)
- [Pretrained Models (NGC)](https://catalog.ngc.nvidia.com/models?query=nemo)
- [Pretrained Models (HuggingFace)](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia)
- [GitHub Discussions](https://github.com/NVIDIA/NeMo/discussions)
- [Tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html)

## Model Deployment

For production deployment of trained models:
- **ASR/TTS**: Use [NVIDIA Riva](https://developer.nvidia.com/riva)
- **LLM**: Use [NVIDIA NIM](https://developer.nvidia.com/nim) services
- **Custom**: Use export tools in `nemo/export/` for ONNX/TensorRT

## Support Matrix

| Platform               | PyPI Install    | NGC Container |
|------------------------|-----------------|---------------|
| Linux x86_64           | Limited support | Full support  |
| Linux arm64            | Limited support | Limited       |
| macOS x86_64           | Deprecated      | Deprecated    |
| macOS arm64 (M-series) | Limited support | Limited       |
| Windows                | No support yet  | No support    |
