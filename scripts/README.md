# Scripts

## Train the Model X-Ray-style detector

`train_detector.py` trains the SRNet few-shot detector from scratch on clean
checkpoints and paired checkpoints containing a seeded random byte sequence
in the least significant float32 bits. The payload is a benign random bitstring;
the experiment does not create or execute malware.

The split happens on source checkpoints before payload variants are generated,
so the same model checkpoint cannot appear on both sides of the evaluation. Use
separate model collections for a stronger out-of-distribution (OOD) test:

```bash
python scripts/train_detector.py \
  --clean-dir data/train_models \
  --heldout-dir data/heldout_models \
  --output-dir outputs/detector \
  --payload-size 1MB
```

Without `--heldout-dir`, provide at least five independent clean checkpoints;
the script makes a seeded source-checkpoint holdout. `--train-severity auto`
chooses the smallest X (up to 23) that can fit the requested payload in every
source model's float32 LSB slots. The default evaluation sweeps X=1..23, and
reports accuracy, balanced accuracy, F1, AUROC, and the Model X-Ray weighted
metric. Detector files and `detector_metrics.json` are written to the output
directory.

These results measure detection of benign random LSB modifications. They do
not establish detection of actual malware or learned encoder perturbations.
For a useful estimate, use substantially more than the minimum and keep model
families/tasks in the held-out directory disjoint from the training directory.
