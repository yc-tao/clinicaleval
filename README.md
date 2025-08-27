# ClinicalEval Phase 0

This repository contains a minimal evaluation skeleton. It can run on toy data and a stubbed inference backend. The goal is reproducibility and simplicity.

## Requirements

- Python >= 3.9
- Conda for environment management
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

### 1. Create Conda Environment

```bash
# Create new conda environment with Python 3.9
conda create -n clinicaleval python=3.9 -y
conda activate clinicaleval
```

### 2. Standard Installation

```bash
# Install core dependencies only
pip install -e .

# Verify basic installation
clinicaleval --help
```

### 3. With Hugging Face Integration

For advanced model evaluation using Lighteval and vLLM:

```bash
# Install with HuggingFace extras (includes lighteval, vllm)
pip install -e .[hf]

# Set environment variables for vLLM (if encountering GPU issues)
export VLLM_USE_TRITON_FLASH_ATTN=0
export CUDA_VISIBLE_DEVICES=0
```

### 4. Development Setup

```bash
# Complete development setup from scratch
git clone https://github.com/yc-tao/clinicaleval.git
cd clinicaleval

# Create and activate conda environment
conda create -n clinicaleval python=3.9 -y
conda activate clinicaleval

# Install in editable mode
pip install -e .

# For HF integration (optional)
pip install -e .[hf]

# Verify installation
clinicaleval --help
```

### 5. Environment Variables (Optional)

For vLLM troubleshooting, you may need these environment variables:

```bash
# Disable Triton flash attention if encountering compilation errors
export VLLM_USE_TRITON_FLASH_ATTN=0

# Specify GPU device
export CUDA_VISIBLE_DEVICES=0

# Alternative multiprocessing method
export VLLM_WORKER_MULTIPROC_METHOD=spawn
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

## Troubleshooting

### vLLM Triton Compilation Errors

If you encounter errors like `ConvertTritonGPUToLLVM` failed:

```bash
# Try disabling Triton kernels
export VLLM_USE_TRITON_FLASH_ATTN=0

# Test with a simple model first
clinicaleval \
  --config configs/base.yaml \
  lighteval.enabled=true \
  lighteval.model_path=microsoft/DialoGPT-medium \
  lighteval.tasks="leaderboard|boolq|0|0"
```

### GPU Memory Issues

```bash
# Limit GPU memory usage
clinicaleval \
  --config configs/base.yaml \
  lighteval.enabled=true \
  lighteval.model_path=your-model \
  lighteval.model_parameters.gpu_memory_utilization=0.7
```

### Check Your Setup

```bash
# Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}')"

# Check GPU memory
nvidia-smi
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
