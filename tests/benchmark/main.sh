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
    "2.4"
    "2.5"
    "2.6"
    "2.7"
    "2.8"
    "2.9"
    "2.10.0"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANODRR_ROOT="${NANODRR_ROOT:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark.py"
RESULTS_CSV="$SCRIPT_DIR/benchmark.csv"

echo "============================================"
echo " nanodrr benchmark — multi PyTorch versions"
echo "============================================"
echo " Project root: $NANODRR_ROOT"
echo " Results:      $RESULTS_CSV"

# Start fresh
rm -f "$RESULTS_CSV"

for version in "${PYTORCH_VERSIONS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo " PyTorch ${version}"
    echo "--------------------------------------------"

    # Create an isolated venv per version and run the benchmark.
    # --directory installs the local nanodrr package; --with pins torch.
    # Adjust --index-url if you need a specific CUDA wheel index.
    uv run \
        --python 3.12 \
        --directory "$NANODRR_ROOT" \
        --with "diffdrr" \
        --with "torch>=${version},<${version%.0}.99" \
        "$BENCHMARK_SCRIPT" \
        --output "$RESULTS_CSV" \
    || echo "  ⚠  PyTorch ${version} failed (may not be released yet)"
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