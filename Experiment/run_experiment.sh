#!/bin/bash
# Run CIFAR-10 Federated Learning Experiment

echo "FedEdge-Accel Experiment Runner"
echo "================================"

# Check if datasets are prepared
if [ ! -d "data/cifar10" ]; then
    echo "Preparing CIFAR-10 dataset..."
    python scripts/prepare_datasets.py --dataset cifar10
fi

# Reproducibility: fixed seed for publication
export FL_SEED=42
# Run experiment
echo "Running CIFAR-10 experiment..."
python experiments/cifar10_experiment.py

# Get latest results directory
LATEST_RESULTS=$(ls -td results/cifar10_* | head -1)

if [ -n "$LATEST_RESULTS" ]; then
    echo "Generating plots..."
    python scripts/generate_plots.py --results_dir "$LATEST_RESULTS" --output_dir plots/
    echo "Results saved to: $LATEST_RESULTS"
    echo "Plots saved to: plots/"
else
    echo "No results found!"
fi

echo "Experiment complete!"
