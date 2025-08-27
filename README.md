# ClinicalEval Phase 0

This repository contains a minimal evaluation skeleton. It can run on toy data and a stubbed inference backend. The goal is reproducibility and simplicity.

## Requirements

- Python >= 3.9
- Modern Python packaging via `pyproject.toml`

## Dependencies

This project uses modern Python packaging with `pyproject.toml` for dependency management:

**Core Dependencies:**
- `pandas>=2.0` - Data manipulation and analysis
- `pyyaml>=6.0` - YAML configuration parsing  
- `numpy>=1.21.0` - Numerical computing support

**Optional Dependencies:**
- `lighteval` - Hugging Face evaluation framework (install with `[hf]` extra)
- `vllm` - High-performance LLM inference (install with `[hf]` extra)

## Setup

### Standard Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode (recommended for development)
pip install -e .
```

### With Hugging Face Integration

For advanced model evaluation using Lighteval and vLLM:

```bash
pip install -e .[hf]
```

### Development Setup

```bash
# Clone and set up for development
git clone <your-repo-url>
cd clinicaleval
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Verify installation
clinicaleval --help
```

## Run

```bash
clinicaleval --config configs/base.yaml
# override parts of the config
clinicaleval --config configs/base.yaml data.max_samples=5 gen.mode=echo
```

Alternative invocations:

```bash
# Run via module
python -m clinicaleval.cli --config configs/base.yaml

# Legacy script path (still supported)
python scripts/run_eval.py --config configs/base.yaml
```

### Use Hugging Face Lighteval with vLLM and Qwen

Install extras (if not already installed):

```bash
pip install -e .[hf]
```

Set your model path and enable the integration via config overrides. Example with Qwen on vLLM:

```bash
clinicaleval \
  --config configs/base.yaml \
  lighteval.enabled=true \
  lighteval.backend=vllm \
  lighteval.model_path=Qwen/Qwen2.5-7B-Instruct \
  lighteval.tasks="leaderboard|truthfulqa:mc|0|0"
```

This will generate a temporary model YAML (under `outputs/`) and invoke the `lighteval` CLI with the `vllm` backend. You can pass advanced settings via:

```bash
clinicaleval \
  --config configs/base.yaml \
  lighteval.enabled=true \
  lighteval.model_path=Qwen/Qwen2.5-7B-Instruct \
  lighteval.model_parameters.tensor_parallel_size=2 \
  lighteval.generation_parameters.max_new_tokens=512
```

Note: Running a full evaluation can be resource- and time-intensive.

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

## Project Structure

```
clinicaleval/
├── src/clinicaleval/          # Main package source
│   ├── __init__.py           # Package initialization
│   └── cli.py                # Command-line interface
├── configs/                  # Configuration files
│   └── base.yaml            # Default configuration
├── data/                    # Sample data
│   └── sample.jsonl         # Test dataset
├── scripts/                 # Legacy scripts
│   └── run_eval.py          # Backward compatibility wrapper
├── outputs/                 # Generated results (created at runtime)
├── pyproject.toml           # Modern Python packaging configuration
├── README.md               # This file
└── LICENSE                 # License information
```

### Appendix: Planned Phase 1 Enhancements

- Abstract data sources into pluggable modules
- Extract inference backends into separate components  
- Modularize metrics computation system
- Add comprehensive test suite
- Enhanced configuration management
