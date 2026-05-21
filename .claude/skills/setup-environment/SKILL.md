# Setup Environment Skill

This skill handles setting up the NeMo development environment, including dependencies, virtual environments, and configuration.

## Usage

```
Setup the NeMo development environment
```

## Steps

### 1. Check Python Version

```bash
python --version  # Should be 3.10+
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install NeMo core dependencies
pip install -e ".[all]"

# Install development dependencies
pip install -e ".[dev]"
```

### 4. Verify CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
NEMO_HOME=/path/to/nemo/cache
WANDB_API_KEY=your_wandb_key  # Optional: for experiment tracking
HUGGINGFACE_TOKEN=your_hf_token  # Optional: for gated models
CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on available GPUs
NEMO_TESTING=0  # Set to 1 during CI/testing
```

Load environment variables:

```bash
export $(cat .env | xargs)
```

### 6. Verify Installation

```python
import nemo
print(f"NeMo version: {nemo.__version__}")

# Test basic imports
from nemo.collections import nlp, asr, tts
print("All collections imported successfully")
```

### 7. Setup Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install

# Run hooks on all files to verify setup
pre-commit run --all-files
```

### 8. Configure Git

```bash
# Set up git to use the correct email/name for contributions
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add upstream remote for syncing with NVIDIA/NeMo
git remote add upstream https://github.com/NVIDIA/NeMo.git
git fetch upstream
```

## Troubleshooting

### CUDA Not Found

```bash
# Check nvidia-smi
nvidia-smi

# Verify CUDA toolkit installation
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu118  # for CUDA 11.8
```

### Missing Dependencies

```bash
# If apex is required for certain features
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    ./
```

### Memory Issues During Setup

```bash
# Limit parallel jobs during compilation
MAX_JOBS=4 pip install -e ".[all]"
```

## Environment Validation Script

Run the full validation to ensure everything is set up correctly:

```bash
python scripts/validate_environment.py
```

Expected output:
```
✓ Python 3.10+
✓ PyTorch with CUDA
✓ NeMo installed
✓ All collections available
✓ Pre-commit hooks installed
✓ Git remotes configured
```
