# FedEdge-Accel: Experimental Setup

This directory contains the complete experimental setup for simulating federated learning with communication compression and hardware-aware local training acceleration.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare datasets:**
   ```bash
   # Prepare standard datasets
   python scripts/prepare_datasets.py --dataset cifar10
   python scripts/prepare_datasets.py --dataset cifar100
   
   # Prepare new datasets for journal submission
   python scripts/prepare_datasets.py --dataset femnist
   python scripts/prepare_datasets.py --dataset har
   ```

3. **Run experiments:**
   ```bash
   # CIFAR-10 experiment
   python experiments/cifar10_experiment.py --config configs/training_config.yaml
   
   # FEMNIST experiment (New)
   python experiments/femnist_experiment.py --config configs/training_config.yaml
   
   # UCI HAR experiment (New)
   python experiments/har_experiment.py --config configs/training_config.yaml
   ```

## Configuration

- `configs/training_config.yaml`: Global federated learning parameters (rounds, LR, compression warmup).
- `configs/device_profiles.yaml`: Heterogeneous hardware profiles (Smartphone, IoT, Tablet, etc.).

### Enabling STC Baseline
To run the Sparse Ternary Compression (STC) baseline mentioned in the paper, ensure your `configs/training_config.yaml` has the following under the `compression` section:
```yaml
compression:
  sparsification:
    method: "stc"
    stc:
      enabled: true
      threshold_ratio: 0.1
```   ```

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
