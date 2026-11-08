# Research-readiness review

## Assessment

The repository is a useful prototype and reproducibility scaffold, but the current evidence is not yet sufficient for a journal or conference submission. The strongest credible near-term framing is **reference-based, benign payload embedding in neural-network parameters under task-utility constraints, with detector performance evaluated across specified parameter-modification families**. Do not frame the current result as malware embedding or malware detection.

The central distinction is experimental: an 8.39-million-bit random message is the same *length* as another message, but it does not share its structure, executable semantics, extraction chain, or resulting parameter statistics. It can test generic carriers and detector sensitivity to that carrier distribution; it cannot alone validate a claim about actual malware. The project intentionally uses benign surrogates. That boundary is methodologically sound as long as claims say exactly that.

## What is already useful

- The project has a modular PyTorch encoder/decoder, a frozen host-classification path, configurable payloads, and explicit BER/accuracy/capacity metrics.
- The notebook reports a separately evaluated CIFAR-10 MobileNetV2 checkpoint at 95.09%; this is a useful clean-host reference, not a result for all payload-modified models.
- The current pipeline uses learnable parameters rather than BatchNorm running buffers. The shared helper and capacity accounting now follow that same parameter set.
- There is a source-model split in the detector training script, and variants are created after splitting. This avoids the most direct leakage from putting modified copies of the same checkpoint in both train and test.
- A Model X-Ray-style GF representation and SRNet/FSL training path exist. The paper's released code and paper should be treated as the reproduction baseline, not as this project's novelty.
- The implementation is defensive-only and uses random payloads; it can support careful studies of benign modification detection without distributing harmful artifacts.

## Critical blockers before submission

### P0 — Define the actual communication/threat model

The decoder uses `reference_decoding=true`: it decodes the residual against the exact clean host representation. A receiver therefore needs that original checkpoint (or equivalent side information). State this as a keyed/reference-based design, measure the cost and availability of that side information, and compare against blind decoding if blind recovery is part of the intended contribution. A decoder that is tested with the encoder's in-memory original tensor is not a receiver protocol.

### P0 — Demonstrate a usable recovered payload

The reported 85% bit recovery is roughly 15% BER. For an 8,388,608-bit payload, that means about 1.26 million wrong bits on average, far from recovering a file. Report exact payload bytes/hash match, raw BER, and net user payload after a documented error-correcting code. Evaluate the complete chain after saving and reloading the modified checkpoint. Do not call nominal input length “recovered capacity.”

### P0 — Align the detector experiment with the claim

The standalone FSL detector is trained on X-LSB random payload edits. The learned encoder writes bounded dense residuals in normalized parameter-image space. These are different distributions. Evaluate both cross-family transfer and a detector trained on encoder outputs, with an independently held-out set of source architectures/checkpoints. Clearly distinguish the statistical fallback from trained SRNet/FSL scores. The current training config has detector weight `delta=0`, so the encoder is not optimized for detector evasion.

The local Model X-Ray paper and released implementation provide a useful reproduction target (GF image plus SRNet/FSL). Reproducing that detector is not by itself a new contribution; a paper needs a defensible extension, new threat model, stronger generalization evidence, or a meaningful result about learned parameter carriers. See the [Model X-Ray paper](https://arxiv.org/abs/2409.19310) and its [released implementation](https://github.com/danigil/ModelXRay).

### P0 — Fix and report end-to-end evaluation

Evaluation must use the same exact learnable parameter ordering and normalization as training, persist/reload the candidate state dict, and then perform reference decoding. It must use the trained task checkpoint for task-specific class heads. Detector predictions must use an explicitly selected trained detector checkpoint when claimed as Model X-Ray results. These correctness/usability issues are addressed in the current code changes, but empirical output is still needed.

### P1 — Establish a credible benchmark

- Use independent host checkpoints for train/validation/test; split by model identity, architecture, and preferably training recipe, never by generated image variant.
- Include multiple model families and independently trained instances. The current helper supports source-checkpoint holdout, but a handful of checkpoints is only a pilot.
- Compare clean-vs-modified detectors against simple statistical baselines, raw LSB statistics, the released Model X-Ray implementation, and a detector trained on the proposed encoder's carriers.
- Report ROC-AUC, balanced accuracy, TPR at low fixed FPR, confidence intervals, and per-architecture/per-severity metrics. Repeat training with multiple seeds.
- Include payload sizes, perturbation budgets, clean/stego task accuracy on full held-out task data, and post-serialization BER/file recovery. Report paired intervals for accuracy change.
- Test robustness under realistic transformations relevant to the stated deployment: checkpoint serialization/reload, supported dtype conversion, pruning/quantization if claimed, and architecture coverage. Do not imply such robustness without those trials.

### P1 — Make capacity and utility claims operational

The parameter-count ratio is a raw representation budget, not the usable capacity of a trained system. Usable capacity must include BER/FEC overhead, reference information, decoder availability, and any robustness constraint. The capacity code now counts learnable parameters consistently, but its “effective bits” (`payload_bits * (1-BER)`) is only an approximate count of correct raw bits, not Shannon capacity or an error-corrected payload guarantee.

### P2 — Reproducibility and novelty

Archive exact commit/config, host and detector checkpoint hashes, data splits, seeds, environment versions, and command lines. The notebook previously overwrote the repository TOML with stale values; that overwrite has been removed. Keep a results table generated from raw per-run outputs, not hand-copied logs.

Compare against both model-weight steganography and the model-malware detection literature. Relevant examples include [EvilModel](https://arxiv.org/abs/2107.08590), [parameter-space steganography capacity analysis](https://arxiv.org/abs/2308.15502), and [purified/unified image steganography](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Purified_and_Unified_Steganographic_Network_CVPR_2024_paper.pdf). These works use different objectives and threat models, so position the work with a direct comparison table rather than claiming a universal breakthrough.

## Changes made in this pass

- Added `extract_parameters()` and aligned pipeline/capacity extraction with learnable parameters only.
- Excluded BatchNorm running-stat buffers from the detector's model-state GF extraction so train/evaluation use a consistent parameter carrier.
- Updated `scripts/evaluate.py` to accept the task host checkpoint and trained SRNet/FSL checkpoint pair, clearly label statistical fallback, and simulate FP32 parameter reconstruction before payload decoding.
- Removed the notebook cell that overwrote the project config with stale settings; the notebook now directs users to the committed config. Updated its inline extraction to use learnable parameters.
- Rewrote `PROJECT.md` to state the actual implemented scope and avoid claims that current configuration does not support.

## Recommended next experimental sequence

1. Run the fixed evaluator on the saved 1 MB encoder checkpoint and report post-FP32-roundtrip BER, byte/hash recovery, and full CIFAR-10 accuracy across multiple payload seeds.
2. Add a documented FEC layer and choose a net payload target; do not count parity as user data.
3. Generate a source-model dataset large enough for architecture-held-out testing. Preserve source IDs and train/test split manifests.
4. Evaluate the current X-LSB-trained FSL detector on both its intended distribution and learned encoder outputs; then train a second detector on encoder outputs and test on unseen host identities.
5. Compare unmodified hosts, random X-LSB carriers, bounded random residuals, and learned carriers under matched payload/budget settings.
6. Only after the baseline is measured, choose one central novel claim and run ablations for reference access, distortion budget, classification term, detector term, and payload rate.

## Scope of this implementation pass

The local workspace contains no trained encoder checkpoint or mounted CIFAR-10 data, so this pass fixes evaluation and documentation but cannot establish new empirical results or claim detector quality. A submission decision requires the experiments above and a novelty comparison against the closest papers.
