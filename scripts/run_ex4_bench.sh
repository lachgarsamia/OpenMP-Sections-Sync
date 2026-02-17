#!/bin/bash
# =============================================================================
# TP4 - Exercise 4: Benchmark Script for DMVM Barrier Analysis
# =============================================================================
# This script runs the DMVM benchmark with various thread counts and matrix
# sizes, collecting timing data for analysis.
#
# Author: Samia Lachgar
# Date: February 2026
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/src/ex4_dmvm_barriers_nowait"
RESULTS_DIR="$PROJECT_ROOT/results"
FIGURES_DIR="$PROJECT_ROOT/report/figures"

# Matrix sizes to test
MATRIX_SIZES=(1000 2000 3000)

# Thread counts to test
THREAD_COUNTS=(1 2 4 8 16)

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$FIGURES_DIR"

# Build the benchmark
echo "=========================================="
echo "Building DMVM benchmark..."
echo "=========================================="
cd "$BUILD_DIR"
make clean
make

# Output file
OUTPUT_FILE="$RESULTS_DIR/ex4_times.csv"
echo "n,threads,version,time_sec,speedup,efficiency,mflops" > "$OUTPUT_FILE"

echo ""
echo "=========================================="
echo "Running benchmarks..."
echo "=========================================="

for N in "${MATRIX_SIZES[@]}"; do
    echo ""
    echo "--- Matrix size: ${N}x${N} ---"
    
    # Run scaling study for this matrix size
    TEMP_FILE="$RESULTS_DIR/temp_n${N}.csv"
    ./dmvm_benchmark -n "$N" --scaling -o "$TEMP_FILE"
    
    # Append to main results (skip header)
    tail -n +2 "$TEMP_FILE" >> "$OUTPUT_FILE"
    rm -f "$TEMP_FILE"
done

echo ""
echo "=========================================="
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="

# Show summary
echo ""
echo "Summary of results:"
echo "-------------------"
head -20 "$OUTPUT_FILE"

# Generate plots
echo ""
echo "=========================================="
echo "Generating plots..."
echo "=========================================="
cd "$SCRIPT_DIR"
if command -v python3 &> /dev/null; then
    python3 plot_ex4.py
    echo "Plots saved to: $FIGURES_DIR"
else
    echo "Warning: python3 not found, skipping plot generation"
    echo "Run 'python3 scripts/plot_ex4.py' manually to generate plots"
fi

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
