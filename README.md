# ClinicalEval Phase 0

This repository contains a minimal evaluation skeleton. It can run on toy data and a stubbed inference backend. The goal is reproducibility and simplicity.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyyaml pandas
```

## Run

```bash
python scripts/run_eval.py --config configs/base.yaml
# override parts of the config
python scripts/run_eval.py --config configs/base.yaml data.max_samples=5 gen.mode=echo
```

## Outputs

Running the script creates an `outputs/` directory with:

- `metrics.json`: values for `exact_match` and `f1_binary`, plus meta information and a copy of the configuration.
- `samples.csv`: each evaluation sample with columns `id,input,gold,prompt,raw_output,pred_label,match`.
- `run.log`: basic INFO level logs with sample count, generation mode and output path.

The inference stub supports three modes controlled by `gen.mode`:

- `reverse` *(default)*: returns the input string reversed so the word `yes` rarely appears.
- `echo`: returns the original input.
- `uppercase`: returns the input in upper case.

Predictions are mapped to binary labels with a simple rule: if the generated text contains the substring `"yes"` (case insensitive) the predicted label is `yes`, otherwise `no`. Because one sample in `data/sample.jsonl` includes the word `yes`, switching from `reverse` to `echo` or `uppercase` flips that prediction to `yes` and changes the metrics.

Use `data.max_samples` to quickly run on a subset of the data.

## Notes

- This is **Phase 0**. Future phases will abstract data sources, inference backends and metrics into separate modules.

### Appendix: Planned Phase 1 Layout

```
clinicaleval/
├── src/
├── tests/
├── configs/
├── data/
└── scripts/
```
