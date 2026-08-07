# Adaptive Deep Learning-based AI Model Steganography

This repository is initialized for defensive AI security research on evaluating AI model malware detectors. The project studies benign/random payload embedding signals, host-model utility preservation, and Model X-Ray-style detection metrics. No malware generation, execution, or operational payload code belongs in this repository.

## Scope

- Defensive research only.
- Random benign payloads only.
- No ML models are implemented in this initialization.
- Future work should keep model code, training loops, evaluation logic, and utility code modular.

## Research Context

The project is informed by:

- `papers/Evil_Model.pdf`: neural-network weight steganography threat context, embedding capacity, performance impact, and countermeasure motivation.
- `papers/Model_X_ray.pdf`: detector context using model-weight image representations and few-shot steganalysis.

## Structure

```text
models/       Model definitions and future host-model adapters.
training/     Training loops, objectives, and experiment orchestration.
evaluation/   Metrics and detector-evaluation workflows.
utils/        Shared logging, configuration, seed, and device helpers.
tests/        Unit tests.
configs/      Experiment and environment configuration files.
scripts/      Command-line entry points and reproducible workflows.
outputs/      Generated logs, figures, metrics, and checkpoints.
papers/       Research references.
```

## Setup

Use Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .[dev]
```

## Verification

```bash
pytest
```

## Current Utilities

- `utils.config.load_config`: load `.toml` or `.json` configuration files.
- `utils.logging.configure_logging`: configure console/file logging consistently.
- `utils.seed.set_seed`: seed Python, NumPy, and PyTorch RNGs.
- `utils.device.get_device`: select CPU, CUDA, or MPS devices conservatively.
- `utils.weights`: extract, flatten, restore, and load floating-point PyTorch
  model weights while preserving `state_dict()` tensor ordering.
- `utils.representation`: convert float32 weights to Model X-Ray-style
  IEEE754 byte channels and reconstruct them bit-exactly.
- `utils.payload`: generate benign random payloads, convert payload bytes to
  bit tensors, save/load payload files, create payload datasets, and measure
  bit error rate.
