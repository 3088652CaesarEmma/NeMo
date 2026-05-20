# Skill: Create Experiment

This skill helps you set up and configure new training experiments in NeMo, including model configs, data pipelines, and logging.

## When to Use

- Starting a new fine-tuning or pre-training run
- Adapting an existing recipe to a new dataset or model variant
- Setting up reproducible experiment configurations

## Steps

### 1. Choose a Base Recipe

NeMo provides recipes under `nemo/collections/*/recipes/`. Find the closest match:

```bash
find nemo/collections -name '*.py' -path '*/recipes/*' | head -20
```

Common starting points:
- LLM fine-tuning: `nemo/collections/llm/recipes/`
- ASR: `nemo/collections/asr/`
- Multimodal: `nemo/collections/multimodal/`

### 2. Create an Experiment Config

Create a Python config file under `experiments/<your-experiment-name>/config.py`:

```python
import nemo_run as run
from nemo.collections.llm.recipes import llama3_8b  # example

@run.cli.entrypoint
def configure():
    recipe = llama3_8b.finetune_recipe(
        num_nodes=1,
        num_gpus_per_node=8,
    )
    # Override dataset
    recipe.data.dataset_path = "/data/my_dataset"
    recipe.trainer.max_steps = 1000
    return recipe
```

### 3. Validate the Config

Dry-run to check for config errors before launching:

```bash
python experiments/<your-experiment>/config.py --dry-run
```

Or use the NeMo CLI:

```bash
nemo run experiments/<your-experiment>/config.py:configure --dry-run
```

### 4. Set Up Logging

Enable Weights & Biases or TensorBoard in your config:

```python
from nemo.lightning.pytorch.callbacks import WandbLogger

recipe.trainer.logger = WandbLogger(
    project="my-project",
    name="experiment-v1",
)
```

For local TensorBoard:

```python
from pytorch_lightning.loggers import TensorBoardLogger

recipe.trainer.logger = TensorBoardLogger(
    save_dir="./logs",
    name="experiment-v1",
)
```

### 5. Configure Checkpointing

```python
from nemo.lightning.pytorch.callbacks import ModelCheckpoint

recipe.trainer.callbacks.append(
    ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        every_n_train_steps=500,
        dirpath="./checkpoints/experiment-v1",
    )
)
```

### 6. Launch the Experiment

Locally (single node):

```bash
nemo run experiments/<your-experiment>/config.py:configure
```

On a SLURM cluster:

```bash
nemo run experiments/<your-experiment>/config.py:configure \
  --executor slurm \
  --account my_account \
  --partition gpu \
  --time 04:00:00
```

## Directory Structure

Keep experiments organized:

```
experiments/
  <experiment-name>/
    config.py          # Main experiment config
    README.md          # What, why, results
    results/           # Logged metrics, plots
    checkpoints/       # Symlink or path to checkpoints
```

## Tips

- **Pin dependencies**: Record the exact NeMo commit hash used (`git rev-parse HEAD`) in your experiment README.
- **Small smoke test first**: Set `max_steps=10` and `limit_val_batches=2` to verify the pipeline end-to-end before a full run.
- **Use `nemo_run` partial overrides**: You can override nested config values from the CLI without editing the config file:
  ```bash
  nemo run config.py:configure trainer.max_steps=2000
  ```
- **Reproducibility**: Set `seed_everything=42` in the trainer config.

## Common Issues

| Problem | Fix |
|---|---|
| OOM on first step | Reduce `micro_batch_size` or enable `activation_checkpointing` |
| Slow data loading | Increase `num_workers` in the dataloader config |
| NaN loss at start | Check learning rate — may be too high for the model size |
| Checkpoint not saving | Verify `dirpath` is writable and `save_top_k > 0` |
