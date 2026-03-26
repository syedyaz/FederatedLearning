# Quick Start Guide for Federated Learning Simulations

This guide provides the necessary steps for colleagues to set up and run the experiments locally.

## 1. Prerequisites
- **Python 3.8+** installed.
- **Git** installed.
- (Optional) **CUDA**-enabled GPU for faster training.

## 2. Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/syedyaz/FederatedLearning.git
cd FederatedLearning
```

### Install Dependencies
Navigate to the `Experiment` directory and install the required Python packages:
```bash
cd Experiment
pip install -r requirements.txt
```

## 3. Launching Simulations

### One-Click Runner (Windows)
We have provided a convenience script that handles dataset preparation, training, and plot generation:
```cmd
.\run_all_simulations.bat
```

### Manual Steps
If you prefer running steps individually:

1. **Prepare Datasets** (Downloads and formats data):
   ```bash
   python scripts/prepare_datasets.py --dataset femnist
   python scripts/prepare_datasets.py --dataset har
   python scripts/prepare_datasets.py --dataset cifar10
   ```

2. **Run Experiments**:
   ```bash
   python experiments/femnist_experiment.py --config configs/training_config.yaml
   python experiments/har_experiment.py --config configs/training_config.yaml
   ```

3. **Generate Plots**:
   ```bash
   python scripts/generate_plots.py
   ```

## 4. Configuration
You can adjust communication rounds, learning rates, and compression settings (including the **STC baseline**) in:
`configs/training_config.yaml`

> [!NOTE]
> This codebase includes a critical fix for an `AttributeError` that previously occurred at Round 51 when compression was disabled. The system is now stable for long-duration simulations with or without compression.

---
*For journal submission support, refer to the `Paper/` directory for the LaTeX manuscript.*
