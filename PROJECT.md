# Project scope and current claims

## Defensive research scope

This repository studies **benign payload embedding in neural-network parameters** and the limits of detectors that inspect model weights. It does not generate, package, or execute malware. A random payload can serve as a controlled carrier for studying parameter modifications, but it is not evidence that an actual malicious model was detected or that an executable payload survives extraction. :codex-annotation{index="1"}

## Implemented pipeline

- A supplied, trained host classifier is kept frozen while an encoder modifies its learnable parameters.
- The encoder represents the learnable FP32 parameter vector as a normalized square image. BatchNorm running-statistic buffers are excluded.
- The decoder currently uses **reference decoding**: it needs the original clean representation to subtract from the modified one. This is a keyed/side-information setting, not blind extraction from a modified checkpoint.
- Training optimizes host classification and benign payload recovery. In the checked-in experiment config, distortion and detector loss weights are zero, so training does not currently optimize stealth or detector evasion.
- A separate Model X-Ray-style SRNet/FSL detector can be trained on clean checkpoints and benign random LSB-modified checkpoints. That is a surrogate detector experiment, not a detector trained on the learned encoder's output distribution.

## Claims supported by the repository today

The code supports experiments on host-task accuracy, benign random-bit recovery, parameter perturbation, and detection of specified benign modification distributions. Results must report the checkpoint, parameter ordering/representation, payload, reference information available to the decoder, serialization format, detector training distribution, and held-out model identities.

Do not describe results as malware detection, malware embedding, detector evasion, blind extraction, or a demonstrated covert channel unless the corresponding threat model and measurements are actually evaluated. Current 1 MB / approximately 85% bit recovery means a substantial raw bit-error rate and is not a byte-exact recovered file.

## Research-readiness plan

See [`RESEARCH_READINESS.md`](RESEARCH_READINESS.md) for the assessment, publication blockers, and an experiment plan. Highest priority is post-save/reload end-to-end evaluation, reliable payload recovery including net rate after error correction, matched detector training/evaluation distributions, and repeated source-model-held-out trials with uncertainty intervals and strong baselines.
