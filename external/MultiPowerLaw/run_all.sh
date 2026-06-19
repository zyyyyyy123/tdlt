#!/bin/bash

# run_all.sh - Script to sequentially run tests and main pipeline for MultiPowerLaw

# Exit on any error
set -e

echo "Starting run_all.sh..."

# 1. Run LR scheduler tests
echo "Running LR scheduler tests..."
python -m tests.test_lrs
echo "LR scheduler tests completed."

# 2. Run data loader tests for 25M, 100M, and 400M
for size in 25 100 400; do
    echo "Running data loader test for ${size}M..."
    python -m tests.test_data_loader -f "$size"
    echo "Data loader test for ${size}M completed."
done

# 3. Run full pipeline (fitting + optimization) for 25M, 100M, and 400M
for size in 25 100 400; do
    echo "Running full pipeline for ${size}M..."
    python -u main.py -f "$size"
    echo "Full pipeline for ${size}M completed. Logs saved to logs/${size}.log"
done

echo "All tasks completed successfully!"