# Execution Guide for FedEdge-Accel Experiments

This guide explains how to execute the federated learning experiments in the Experiment folder.

## Prerequisites

1. **Python 3.8 or higher** installed
2. **Required packages** installed (see Installation section below)

## Installation

### Step 1: Install Dependencies

Navigate to the Experiment directory and install required packages:

```bash
cd Experiment
pip install -r requirements.txt
```

**Note for GPU Users**: The `requirements.txt` installs PyTorch CPU version by default. If you have an NVIDIA GPU and want to use GPU acceleration (recommended for faster training), install PyTorch with CUDA support **before** installing other requirements:

```bash
# Check your CUDA version first: nvidia-smi
# Look for "CUDA Version: X.X" in the output

# For CUDA 11.1 (older drivers like 457.49):
# Try CUDA 11.8 first (may work):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# If that fails, use older PyTorch compatible with CUDA 11.1:
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# For CUDA 11.8 (recommended - requires driver 450.80+):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (requires driver 525.60+):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install remaining dependencies
pip install -r requirements.txt
```

Or use the setup script:

```bash
python setup.py
```

### Step 2: Prepare Datasets

Before running experiments, you need to download and prepare the datasets:

```bash
# For CIFAR-10
python scripts/prepare_datasets.py --dataset cifar10

# For CIFAR-100 (optional)
python scripts/prepare_datasets.py --dataset cifar100
```

## Execution Methods

### Method 1: Using the Batch Script (Windows)

On Windows, you can use the provided batch script:

```bash
run_experiment.bat
```

This script will:
- Check if datasets are prepared
- Run the CIFAR-10 experiment
- Generate plots automatically

### Method 2: Using the Shell Script (Linux/Mac)

On Linux or Mac, use the shell script:

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

### Method 3: Direct Python Execution

You can also run the experiment directly:

```bash
python experiments/cifar10_experiment.py
```

## Understanding the Progress Logging

The experiment now includes comprehensive progress logging that shows:

### 1. **Initialization Phase**
- Configuration loading
- Dataset loading with sample counts
- Federated split creation
- Device profile setup
- Model initialization

### 2. **Training Phase (Per Round)**
- **Round Progress**: Current round number, total rounds, and percentage complete
- **Client Selection**: Which clients are selected for training
- **Local Training**: Progress for each client's local training
  - Number of samples per client
  - Training loss
  - Training time
- **Aggregation**: Time taken to aggregate client updates
- **Evaluation**: When evaluation occurs (every N rounds)
  - Test accuracy
  - Test loss
  - Communication and computation costs (for FedEdge-Accel)
- **Time Estimates**: 
  - Time per round
  - Elapsed time
  - Average time per round
  - Estimated remaining time

### 3. **Completion Summary**
- Total experiment time
- Average time per round
- Results file locations

## Log Files

All progress is logged to:
- **Console**: Real-time progress output
- **Log File**: `results/cifar10_TIMESTAMP/experiment.log` - Complete detailed log

The log file contains:
- Timestamps for all operations
- Detailed debug information
- Performance metrics
- Error messages (if any)

## Output Structure

After execution, results are saved in:

```
results/
└── cifar10_TIMESTAMP/
    ├── experiment.log          # Detailed execution log
    ├── IID_fededge_accel.json  # FedEdge-Accel results (IID)
    ├── IID_fedavg.json         # FedAvg baseline results (IID)
    ├── NonIID_fededge_accel.json # FedEdge-Accel results (Non-IID)
    └── NonIID_fedavg.json      # FedAvg baseline results (Non-IID)
```

## Configuration

You can modify experiment parameters in:

- **`configs/training_config.yaml`**: Training parameters
  - Number of rounds
  - Client selection fraction
  - Learning rate
  - Evaluation frequency
  
- **`configs/device_profiles.yaml`**: Device hardware profiles
  - Batch sizes
  - Local epochs
  - Compression settings

## Monitoring Progress

### Real-time Monitoring

The experiment displays progress in real-time:
- Progress bars for each training round
- Detailed logging output
- Time estimates

### Example Output

```
[Round 5/200] Progress: 2.5%
  Training 10 clients locally...
  Average client training time: 2.34s
  Aggregating updates from 10 clients...
  Evaluating global model...
  Evaluation Results:
    Accuracy: 45.23%
    Loss: 1.2345
    Communication Cost: 123.45
    Computation Cost: 234.56
    Total Cost: 358.01
  Round completed in 25.67s
  Elapsed: 2.1min | Avg/round: 25.67s | Est. remaining: 83.2min
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in `configs/device_profiles.yaml`
   - Reduce number of clients in `configs/training_config.yaml`
   - Use smaller model (MobileNet-V2 instead of ResNet-18)

2. **Slow Training**
   - Reduce number of rounds in `configs/training_config.yaml`
   - Use fewer local epochs for IoT devices
   - Enable GPU if available

3. **Import Errors**
   - Make sure you're in the Experiment directory
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+)

4. **Dataset Not Found**
   - Run `python scripts/prepare_datasets.py --dataset cifar10` first
   - Check that `data/cifar10` directory exists

## GPU Acceleration Setup

The experiments automatically use GPU if available, which can significantly speed up training (4-8x faster). Follow these steps to enable GPU acceleration:

### Step 1: Verify GPU Availability

Check if PyTorch can detect your GPU:

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output if GPU is available:
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 3080
```

### Step 2: Install PyTorch with CUDA Support

If CUDA is not available, you need to install PyTorch with CUDA support. First, check your CUDA version:
```bash
nvidia-smi
```

**Important:** Match the PyTorch CUDA version to your driver's supported CUDA version:

**For CUDA 11.1 (older drivers, e.g., driver 457.49):**
```bash
# Option 1: Try CUDA 11.8 (may work if driver is compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Option 2: If above fails, install older PyTorch version compatible with CUDA 11.1
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

**For CUDA 11.8 (recommended - requires driver 450.80+):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1 (requires driver 525.60+):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only (if no GPU):**
```bash
pip install torch torchvision torchaudio
```

**Note:** If your driver only supports CUDA 11.1 (like driver 457.49), you have two options:
1. **Upgrade your NVIDIA driver** (recommended) to support CUDA 11.8 or 12.1 for better PyTorch compatibility
2. Use the CUDA 11.1 compatible installation above

### Step 3: Verify GPU Usage During Training

When you run the experiment, check the log output. You should see:

```
Initializing FedEdge-Accel server on device: cuda
  Device: cuda
```

If you see `device: cpu` instead, GPU is not being used. Check:
1. PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. GPU drivers are installed: `nvidia-smi` should show your GPU
3. CUDA version compatibility between PyTorch and your drivers

### Step 4: Monitor GPU Usage

While training is running, monitor GPU utilization:

```bash
# On Windows (PowerShell)
nvidia-smi -l 1

# On Linux/Mac
watch -n 1 nvidia-smi
```

You should see GPU memory usage and utilization increase during training.

### GPU Troubleshooting

**Issue: PyTorch reports CUDA unavailable**
- Solution: Install PyTorch with CUDA support matching your CUDA version
- Check: `python -c "import torch; print(torch.version.cuda)"`

**Issue: Out of Memory (OOM) errors**
- Solution: Reduce batch size in `configs/device_profiles.yaml`
- Or use a smaller model (MobileNet-V2 instead of ResNet-18)

**Issue: GPU detected but still using CPU**
- Check: The experiment automatically uses GPU if `torch.cuda.is_available()` returns `True`
- Verify: Look for "Device: cuda" in the log output
- If still using CPU, ensure PyTorch was installed with CUDA support

**Issue: Multiple GPUs**
- The code uses the default GPU (device 0)
- To use a specific GPU, set `CUDA_VISIBLE_DEVICES`:
  ```bash
  # Use GPU 1
  set CUDA_VISIBLE_DEVICES=1
  python experiments/cifar10_experiment.py
  ```

### Performance Tips

1. **Batch Size**: GPU can handle larger batch sizes. Consider increasing `batch_size` in `configs/device_profiles.yaml` for GPU-accelerated training
2. **Mixed Precision**: The code supports mixed precision training automatically when GPU is available
3. **Memory Management**: If you encounter OOM errors, reduce batch sizes or use gradient accumulation

## Expected Runtime




Results are saved incrementally, so you can stop and resume if needed.

## Generating Plots

After the experiment completes, generate visualization plots:

```bash
python scripts/generate_plots.py --results_dir results/cifar10_TIMESTAMP/ --output_dir plots/
```

Replace `TIMESTAMP` with the actual timestamp from your results directory.

## Next Steps

1. Review the log file for detailed execution information
2. Check JSON result files for metrics
3. Generate plots for visualization
4. Experiment with different configurations
5. Run ablation studies: `python experiments/ablation_studies.py`

## Support

For issues or questions:
1. Check the log file: `results/cifar10_TIMESTAMP/experiment.log`
2. Review the QUICKSTART.md guide
3. Check the README.md for additional information
