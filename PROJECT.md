# Project scope and current claims

## Defensive research scope

This repository studies **benign payload embedding in neural-network parameters** and the limits of detectors that inspect model weights. It does not generate, package, or execute malware. A random payload can serve as a controlled carrier for studying parameter modifications, but it is not evidence that an actual malicious model was detected or that an executable payload survives extraction. :codex-annotation{index="1"}

## Implemented pipeline

- A supplied, trained host classifier is kept frozen while an encoder modifies its learnable parameters.
- The encoder represents the learnable FP32 parameter vector as a normalized square image. BatchNorm running-statistic buffers are excluded.
- The main experiment config still uses **reference decoding** for compatibility with existing runs. The `blind-decoder` branch adds an optional blind reveal path trained from the modified model's own normalized weights, with no clean checkpoint passed to the decoder. This is an implementation prototype; it has not yet established blind-recovery performance.
- The blind reveal input combines normalized modified weights with a deterministic local high-pass feature. The receiver utility `scripts/decode_blind.py` loads the reveal network and the modified host checkpoint only; the clean host checkpoint is not part of its inputs.
- The combined trainable encoder and decoder are capped at 1,000,000 parameters. The provided blind config uses 170,606 parameters for those two networks; the host classifier is not counted toward this research constraint.
- Training optimizes host classification and benign payload recovery. In the checked-in experiment config, distortion and detector loss weights are zero, so training does not currently optimize stealth or detector evasion.
- A separate Model X-Ray-style SRNet/FSL detector can be trained on clean checkpoints and benign random LSB-modified checkpoints. That is a surrogate detector experiment, not a detector trained on the learned encoder's output distribution.

## Claims supported by the repository today

The code supports experiments on host-task accuracy, benign random-bit recovery, parameter perturbation, and detection of specified benign modification distributions. Results must report the checkpoint, parameter ordering/representation, payload, reference information available to the decoder, serialization format, detector training distribution, and held-out model identities.

Do not describe results as malware detection, malware embedding, detector evasion, blind extraction, or a demonstrated covert channel unless the corresponding threat model and measurements are actually evaluated. Current 1 MB / approximately 85% bit recovery means a substantial raw bit-error rate and is not a byte-exact recovered file.

## Research-readiness plan

See [`RESEARCH_READINESS.md`](RESEARCH_READINESS.md) for the assessment, publication blockers, and an experiment plan. Highest priority is post-save/reload end-to-end evaluation, reliable payload recovery including net rate after error correction, matched detector training/evaluation distributions, and repeated source-model-held-out trials with uncertainty intervals and strong baselines.
