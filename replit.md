# Adaptive Deep Learning-based AI Model Steganography

## Project overview

Defensive AI security research on evaluating and evading AI model malware
detectors.  The project trains an encoder–decoder network to embed benign
random payloads into neural-network weight representations while simultaneously
preserving the host model's classification accuracy and minimising detectability
by the Model X-Ray detector.

**Scope:** Defensive research only.  Random benign payloads only.  No malware
generation, execution, or operational payload code.

---

## Architecture

```
Host Model
    ↓
Weight Extraction  (utils/weights.py)
    ↓
Model X-Ray 4-Channel Representation  (utils/representation.py)
    ↓
Encoder  (models/encoder.py)  ←─── Payload bits (utils/payload.py)
    ↓
Modified Representation
    ├── Decoder (models/decoder.py)       → Payload logits
    ├── Frozen DifferentiableDetector     → Detector logits (adversarial)
    └── STE → Modified Weights → Host Model → Classification logits
```

### Training objectives (composite loss)

```
L = α · ClassificationLoss   (preserve host model accuracy)
  + β · PayloadLoss           (recover the random payload via BCE)
  + γ · DistortionLoss        (MSE on weight representations)
  + δ · DetectorLoss          (fool the frozen Model X-Ray detector)
```

The detector is an SRNet-inspired CNN (`evaluation/differentiable_detector.py`)
whose **parameters are always frozen** but whose forward activations remain
differentiable, so the encoder learns to fool the detector without ever
updating the detector's weights.

---

## Project structure

```
models/
  encoder.py                  WeightPayloadEncoder — FiLM + attention residual CNN
  decoder.py                  ChunkedPayloadDecoder — shared-head chunk reconstruction
  host_models.py              HostModelAdapter — ResNet18/50, MobileNetV2, VGG16
  pipeline.py                 EmbeddingPipeline — full end-to-end pipeline module

training/
  losses.py                   CompositeLoss + four individual loss modules
  trainer.py                  Generic configurable Trainer with AMP / early stopping
  dataset.py                  SteganographyDataset, SyntheticWeightDataset
  experiment.py               SteganographyExperiment — full orchestration

evaluation/
  differentiable_detector.py  DifferentiableDetector (frozen SRNet-inspired CNN)
  detector.py                 ModelXRayDetector (statistical fallback + SRNet/FSL path)
  fsl_detector.py             Few-shot SRNet training and centroid/1-NN classifiers
  metrics.py                  DetectorMetrics — accuracy / precision / recall / AUC
  plotting.py                 Publication-quality matplotlib figures
  accuracy.py                 AccuracyDrop evaluation
  capacity.py                 Embedding capacity (bits-per-parameter)

utils/
  config.py                   TOML / JSON config loader
  device.py                   Conservative device selection
  gf_image.py                 Grayscale-Fourpart image conversion and resizing
  logging.py                  Console + file logging setup
  payload.py                  Benign random payload generation / bit conversion
  representation.py           IEEE754 4-channel weight image conversion
  seed.py                     Python / NumPy / PyTorch seeding
  weights.py                  Weight extraction, flatten, restore, load

scripts/
  train.py                    CLI: full training run
  evaluate.py                 CLI: evaluate a trained checkpoint
  run_demo.py                 End-to-end embedding + SRNet/FSL detection demo

configs/
  default.toml                Safety and runtime defaults
  experiment.toml             Full experiment hyperparameter reference

tests/                        Unit tests for every module
papers/                       Research reference PDFs
outputs/                      Generated logs, figures, metrics, checkpoints
```

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

## Running tests

```bash
pytest
```

## Training (synthetic data, all defaults)

```bash
python scripts/train.py \
    --host-model resnet18 \
    --epochs 50 \
    --batch-size 16 \
    --payload-size 128KB \
    --output-dir outputs/run_01
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/run_01/checkpoints/best.pt \
    --host-model resnet18 \
    --payload-size 128KB \
    --num-samples 64 \
    --output-dir outputs/run_01/eval
```

## End-to-end SRNet/FSL demo

The demo trains the embedding pipeline, creates benign and payload-embedded
checkpoints, trains the standalone PyTorch SRNet with triplet loss, evaluates
held-out checkpoints, and writes metrics and plots below the output directory.

For a quick, low-memory verification run:

```bash
python scripts/run_demo.py \
    --host-model tiny \
    --num-classes 10 \
    --synthetic-samples 4 \
    --epochs 1 \
    --fsl-epochs 1 \
    --num-samples 2 \
    --test-samples 1 \
    --batch-size 1 \
    --imsize 32 \
    --device cpu \
    --output-dir outputs/demo_smoke
```

For a research run using the default ResNet18 host, omit `--host-model tiny`
and increase `--epochs`, `--fsl-epochs`, and the number of samples. Results are
written to `outputs/<run>/metrics/demo_results.json` and
`outputs/<run>/figures/`.

---

## Key design decisions

- **Frozen detector with differentiable forward pass**: The `DifferentiableDetector`
  calls `requires_grad_(False)` on all parameters at construction time, but the
  computation graph still lets gradients flow through activations to the encoder.
  The encoder therefore learns an adversarial objective without ever updating the
  detector.

- **Straight-Through Estimator (STE)** for weight reconstruction: converting the
  encoder's float representation back to IEEE754 float32 weights requires
  non-differentiable byte packing.  `_ChannelsToWeightsSTE` in `models/pipeline.py`
  uses an exact forward pass with an identity-approximation backward pass so that
  the classification loss still contributes gradients to the encoder.

- **`torch.func.functional_call`** for classification: the host model's weights
  are patched virtually (no in-place mutation) via `functional_call`, enabling
  per-step weight surgery without breaking the gradient graph.

- **SyntheticWeightDataset** for fast iteration: full ImageNet training is not
  required to develop and test the pipeline; the synthetic dataset generates
  random weight representations and payloads on CPU in milliseconds.

---

## User preferences

- Python 3.11.
- Google-style docstrings.
- Type hints throughout.
- Modular code — no monolithic files.
- No hardcoded absolute paths.
- No malware, no operational payloads, benign random data only.
