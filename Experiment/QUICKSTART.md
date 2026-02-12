# Quick Start Guide

## 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Or run setup script
python setup.py
```

## 2. Prepare Datasets

```bash
# Download and prepare CIFAR-10
python scripts/prepare_datasets.py --dataset cifar10

# Download and prepare CIFAR-100
python scripts/prepare_datasets.py --dataset cifar100
```

## 3. Run Experiments

### CIFAR-10 Experiment

```bash
python experiments/cifar10_experiment.py
```

This will:
- Create IID and non-IID federated splits
- Run FedEdge-Accel and FedAvg baseline
- Save results to `results/cifar10_TIMESTAMP/`

### Custom Configuration

Edit `configs/training_config.yaml` to modify:
- Number of rounds
- Client selection fraction
- Learning rate
- Compression parameters

## 4. Generate Results and Plots

```bash
# Generate plots from results
python scripts/generate_plots.py --results_dir results/cifar10_TIMESTAMP/ --output_dir plots/
```

## 5. View Results

Results are saved as JSON files in `results/` directory:
- `IID_fededge_accel.json`: FedEdge-Accel results on IID data
- `IID_fedavg.json`: FedAvg baseline on IID data
- `NonIID_fededge_accel.json`: FedEdge-Accel results on non-IID data
- `NonIID_fedavg.json`: FedAvg baseline on non-IID data

Plots are saved in `plots/` directory:
- `convergence_curves.png`: Accuracy and loss curves
- `communication_cost.png`: Communication cost comparison
- `comparison_table.txt`: Summary table

## Troubleshooting

### Out of Memory
- Reduce batch size in `configs/device_profiles.yaml`
- Reduce number of clients in `configs/training_config.yaml`
- Use smaller model (MobileNet-V2 instead of ResNet-18)

### Slow Training
- Reduce number of rounds in `configs/training_config.yaml`
- Use fewer local epochs for IoT devices
- Enable GPU if available

### Import Errors
- Make sure you're in the Experiment directory
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+)

## Expected Runtime

- CIFAR-10 with 100 clients, 200 rounds: ~2-4 hours (CPU) or ~30-60 minutes (GPU)
- Results are saved incrementally, so you can stop and resume

## Next Steps

1. Run ablation studies: `python experiments/ablation_studies.py`
2. Experiment with different compression ratios
3. Test on CIFAR-100: `python experiments/cifar100_experiment.py`
