#!/usr/bin/env bash
# Run benchmarks across multiple PyTorch versions using uv.
#
# Usage:
#   chmod +x ./tests/benchmark/main.sh
#   ./tests/benchmark/main.sh
#
# Prerequisites:
#   - uv installed (https://docs.astral.sh/uv/)
#   - CUDA-capable GPU
#   - Run from the nanodrr project root (or set NANODRR_ROOT)

set -euo pipefail

PYTORCH_VERSIONS=(
    "2.5"
    "2.6"
    "2.7"
    "2.8"
    "2.9"
    "2.10"
    "2.11"
)

# Map PyTorch version -> CUDA wheel index
declare -A CUDA_INDEX
CUDA_INDEX["2.5"]="https://download.pytorch.org/whl/cu124"
CUDA_INDEX["2.6"]="https://download.pytorch.org/whl/cu126"
CUDA_INDEX["2.7"]="https://download.pytorch.org/whl/cu128"
CUDA_INDEX["2.8"]="https://download.pytorch.org/whl/cu128"
CUDA_INDEX["2.9"]="https://download.pytorch.org/whl/cu128"
CUDA_INDEX["2.10"]="https://download.pytorch.org/whl/cu128"
CUDA_INDEX["2.11"]="https://download.pytorch.org/whl/cu129"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANODRR_ROOT="${NANODRR_ROOT:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark.py"
RESULTS_CSV="$SCRIPT_DIR/benchmark.csv"

echo "============================================"
echo " nanodrr benchmark — multi PyTorch versions"
echo "============================================"
echo " Project root: $NANODRR_ROOT"
echo " Results:      $RESULTS_CSV"

rm -f "$RESULTS_CSV"

for version in "${PYTORCH_VERSIONS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo " PyTorch ${version}"
    echo "--------------------------------------------"

    # Look up CUDA index, fallback to cu128
    cuda_index="${CUDA_INDEX[$version]:-https://download.pytorch.org/whl/cu128}"

    uv run \
        --python 3.12 \
        --directory "$NANODRR_ROOT" \
        --with "diffdrr" \
        --with "torch>=${version},<${version%.0}.99" \
        --extra-index-url "$cuda_index" \
        "$BENCHMARK_SCRIPT" \
        --output "$RESULTS_CSV" \
    || echo "  ❌   PyTorch ${version} failed"
done

echo ""
echo "============================================"
echo " Generating plots"
echo "============================================"

uv run \
    --python 3.12 \
    --with "pandas" \
    --with "ultraplot" \
    "$SCRIPT_DIR/plot.py"

echo ""
echo "============================================"
echo " Done — results saved to $RESULTS_CSV"
echo "============================================"
