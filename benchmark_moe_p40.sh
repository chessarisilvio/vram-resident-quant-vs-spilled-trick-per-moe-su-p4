#!/bin/bash
# benchmark_moe_p40.sh
# Script to benchmark IQ3_XXS (fully resident) vs IQ4_XS (spilled) on NVIDIA P40 for MoE models.
#
# Usage:
#   LLAMA_SERVER_BIN=/path/to/llama-server ./benchmark_moe_p40.sh <model_iq3_xxs> <model_iq4_xs> [options]
#
# Options:
#   --n-predict <N>   Number of tokens to predict (default: 100)
#   --temp <T>        Temperature (default: 0.8)
#   --ctx-size <S>    Context size (default: 512)
#   --batch-size <B>  Batch size (default: 512)
#   --gpu-layers <L>  Number of layers to offload to GPU (default: 99 for full offload)
#
# Environment variables:
#   LLAMA_SERVER_BIN: Path to llama-server binary (if not in PATH)
#   LLAMA_SERVER_START_CMD: Full command to start the server (optional, if set, used to restart)
#
# The script will:
#   1. Stop any existing llama-server (by killing the process)
#   2. Run benchmark for IQ3_XXS model
#   3. Run benchmark for IQ4_XS model
#   4. Restart the server (if LLAMA_SERVER_START_CMD is set)
#
# Output:
#   Prints a table with results for each model: tokens per second, average latency, VRAM usage.

set -euo pipefail

# Default values
N_PREDICT=100
TEMP=0.8
CTX_SIZE=512
BATCH_SIZE=512
GPU_LAYERS=99

# Parse arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_iq3_xxs> <model_iq4_xs> [options]"
    echo "Options:"
    echo "  --n-predict <N>   Number of tokens to predict (default: $N_PREDICT)"
    echo "  --temp <T>        Temperature (default: $TEMP)"
    echo "  --ctx-size <S>    Context size (default: $CTX_SIZE)"
    echo "  --batch-size <B>  Batch size (default: $BATCH_SIZE)"
    echo "  --gpu-layers <L>  Number of layers to offload to GPU (default: $GPU_LAYERS)"
    exit 1
fi

MODEL_IQ3_XXS="$1"
MODEL_IQ4_XS="$2"
shift 2

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-predict)
            N_PREDICT="$2"
            shift 2
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --ctx-size)
            CTX_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu-layers)
            GPU_LAYERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find llama-server binary
if command -v llama-server &> /dev/null; then
    LLAMA_SERVER_BIN="llama-server"
elif [[ -n "${LLAMA_SERVER_BIN:-}" ]]; then
    # Use the environment variable if set
    :
else
    echo "Error: llama-server not found in PATH and LLAMA_SERVER_BIN not set."
    exit 1
fi

# Function to stop llama-server
stop_llama_server() {
    echo "Stopping any existing llama-server..."
    pkill -f llama-server || true
    sleep 2
}

# Function to run benchmark for a given model
run_benchmark() {
    local model_path="$1"
    local quant_label="$2"

    echo "Running benchmark for $quant_label model: $(basename "$model_path")"

    # Start the server in the background
    "${LLAMA_SERVER_BIN}" -m "$model_path" --n-predict "$N_PREDICT" --temp "$TEMP" --ctx-size "$CTX_SIZE" --batch-size "$BATCH_SIZE" --gpu-layers "$GPU_LAYERS" --log-disable > /dev/null 2>&1 &
    SERVER_PID=$!

    # Give the server a moment to start
    sleep 3

    # We'll use a simple prompt to benchmark
    local prompt="Hello, my name is"

    # Measure time and tokens
    local start_time=$(date +%s.%N)
    local output=$("${LLAMA_SERVER_BIN}" -m "$model_path" -p "$prompt" --n-predict "$N_PREDICT" --temp "$TEMP" --ctx-size "$CTX_SIZE" --batch-size "$BATCH_SIZE" --gpu-layers "$GPU_LAYERS" --log-disable 2>&1)
    local end_time=$(date +%s.%N)

    # Kill the server after benchmark
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true

    # Calculate tokens per second and latency
    local total_time=$(echo "$end_time - $start_time" | bc)
    local tokens_per_second=$(echo "$N_PREDICT / $total_time" | bc -l)
    local latency_per_second=$(echo "$total_time / $N_PREDICT" | bc -l)

    # Get VRAM usage during the run (we can't easily get peak, so we'll get current usage after a short run)
    # Instead, we'll run a separate command to get VRAM usage while the server is idle?
    # For simplicity, we'll note that we can't get peak without monitoring. We'll use nvidia-smi to get used memory.
    # We'll run a quick nvidia-smi and extract the used memory for the GPU.
    local vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")

    # Output results
    echo "=== $quant_label ==="
    echo "Model: $(basename "$model_path")"
    echo "Time for $N_PREDICT tokens: ${total_time}s"
    echo "Tokens per second: ${tokens_per_second}"
    echo "Average latency per token: ${latency_per_second}s"
    echo "VRAM used: ${vram_used} MB"
    echo ""
}

# Main
stop_llama_server

# Run benchmarks
run_benchmark "$MODEL_IQ3_XXS" "IQ3_XXS (fully resident)"
run_benchmark "$MODEL_IQ4_XS" "IQ4_XS (spilled)"

# Restart server if LLAMA_SERVER_START_CMD is set
if [[ -n "${LLAMA_SERVER_START_CMD:-}" ]]; then
    echo "Restarting server with LLAMA_SERVER_START_CMD..."
    eval "$LLAMA_SERVER_START_CMD" &
    echo "Server restarted."
else
    echo "LLAMA_SERVER_START_CMD not set. Please restart your server manually."
fi

echo "Benchmark complete."