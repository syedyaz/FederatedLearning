# FedEdge-Accel: Experimental Setup

This directory contains the complete experimental setup for simulating federated learning with communication compression and hardware-aware local training acceleration.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare datasets:**
   ```bash
   python scripts/prepare_datasets.py --dataset cifar10
   python scripts/prepare_datasets.py --dataset cifar100
   ```

3. **Run CIFAR-10 experiment:**
   ```bash
   python experiments/cifar10_experiment.py --config configs/cifar10_config.yaml
   ```

4. **Generate results and plots:**
   ```bash
   python scripts/generate_results.py --results_dir results/
   python scripts/generate_plots.py --results_dir results/ --output_dir plots/
   ```

## Directory Structure

```
Experiment/
├── configs/              # Configuration files
├── data/                 # Dataset storage
├── models/               # Model architectures
├── federated/            # Federated learning components
├── experiments/          # Experiment scripts
├── baselines/            # Baseline implementations
├── scripts/              # Utility scripts
├── results/              # Experiment results
├── plots/                # Generated plots
└── utils/                # Utility functions
```

## Experiments

- **CIFAR-10**: `python experiments/cifar10_experiment.py`
- **CIFAR-100**: `python experiments/cifar100_experiment.py`
- **Ablation Studies**: `python experiments/ablation_studies.py`

## Results

Results are saved in `results/` directory with timestamps. Use `scripts/generate_plots.py` to visualize results.
