# Benchmark Guide

## Overview
This guide explains how to use the `benchmark_moe_p40.sh` script to compare performance between IQ3_XXS (fully resident in VRAM) and IQ4_XS (spilled to RAM) quantizations for MoE models on NVIDIA P40 GPUs.

## Prerequisites
- NVIDIA P40 GPU (24GB VRAM)
- llama-server binary in PATH or LLAMA_SERVER_BIN environment variable set
- Two GGUF model files: one IQ3_XXS quantized, one IQ4_XS quantized
- nvidia-smi utility (comes with NVIDIA drivers)
- bc calculator (usually pre-installed)

## Environment Variables
The script respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| LLAMA_SERVER_BIN | Path to llama-server binary | `llama-server` (from PATH) |
| LLAMA_SERVER_START_CMD | Full command to restart the server after benchmark | Not set |

## Usage
```bash
./benchmark_moe_p40.sh <model_iq3_xxs> <model_iq4_xs> [options]
```

### Arguments
- `<model_iq3_xxs>`: Path to the IQ3_XXS quantized GGUF model
- `<model_iq4_xs>`: Path to the IQ4_XS quantized GGUF model

### Options
- `--n-predict <N>`: Number of tokens to predict per test (default: 100)
- `--temp <T>`: Temperature for sampling (default: 0.8)
- `--ctx-size <S>`: Context size in tokens (default: 512)
- `--batch-size <B>`: Batch size for processing (default: 512)
- `--gpu-layers <L>`: Number of layers to offload to GPU (default: 99 for full offload)

## Example
```bash
# Basic usage
./benchmark_moe_p40.sh ./models/mixtral-8x7b-iq3_xxs.gguf ./models/mixtral-8x7b-iq4_xs.gguf

# With custom options
./benchmark_moe_p40.sh ./models/mixtral-8x7b-iq3_xxs.gguf ./models/mixtral-8x7b-iq4_xs.gguf --n-predict 200 --temp 0.7 --ctx-size 1024
```

## What the Script Does
1. Stops any existing llama-server processes
2. Runs a benchmark for the IQ3_XXS model:
   - Starts llama-server with the model in background
   - Generates text using a fixed prompt
   - Measures time taken and calculates tokens/second
   - Records VRAM usage via nvidia-smi
   - Stops the server
3. Repeats the same process for the IQ4_XS model
4. Optionally restarts the server using LLAMA_SERVER_START_CMD if set

## Interpreting Results
The script outputs for each model:
- **Model**: Filename of the GGUF model
- **Time for N tokens**: Total wall-clock time to generate the specified number of tokens
- **Tokens per second**: Primary performance metric (higher is better)
- **Average latency per second**: Time per token in seconds (lower is better)
- **VRAM used**: Current GPU memory usage in MB (approximate)

## Expected Outcome
Based on the research in this project, you should observe:
- IQ3_XXS (fully resident in VRAM) achieving higher tokens/second than IQ4_XS
- Despite IQ3_XXS being more aggressively quantized, avoiding RAM spill results in better performance
- The performance difference should align with the 2.4× advantage mentioned in the project documentation

## Notes
- The script uses a simple prompt ("Hello, my name is") for consistency
- VRAM usage is measured after the benchmark run and may not reflect peak usage
- For more accurate VRAM monitoring, consider using external tools like `nvidia-smi dmon` during the benchmark
- The script kills and restarts the server between tests to ensure clean state
- If you have a specific way to start your llama-server (with specific arguments for your setup), set LLAMA_SERVER_START_CMD to automate restart

## Troubleshooting
- "llama-server not found": Ensure llama-server is in your PATH or set LLAMA_SERVER_BIN
- Permission errors: Make sure the script is executable (`chmod +x benchmark_moe_p40.sh`)
- nvidia-smi not found: Install NVIDIA drivers and ensure nvidia-smi is in PATH
- bc not found: Install bc calculator (`sudo apt install bc` on Ubuntu/Debian)