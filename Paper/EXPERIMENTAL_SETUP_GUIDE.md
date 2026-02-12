# FedEdge-Accel: Detailed Experimental Setup Guide

This document provides a comprehensive guide for setting up and running experiments for the FedEdge-Accel paper.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Federated Data Splits](#federated-data-splits)
4. [Device Profile Configuration](#device-profile-configuration)
5. [Model Architectures](#model-architectures)
6. [Training Configuration](#training-configuration)
7. [Compression Parameters](#compression-parameters)
8. [Hardware Modeling](#hardware-modeling)
9. [Baseline Implementations](#baseline-implementations)
10. [Evaluation Scripts](#evaluation-scripts)
11. [Expected Results](#expected-results)

---

## 1. Environment Setup

### Required Software

```bash
# Python 3.8+
python --version  # Should be 3.8 or higher

# PyTorch 1.12+ with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Flower framework for federated learning
pip install flwr

# Additional dependencies
pip install numpy scipy matplotlib seaborn pandas
pip install scikit-learn
pip install tensorboard
pip install tqdm
```

### Project Structure

```
FedEdge-Accel/
├── data/
│   ├── cifar10/
│   ├── cifar100/
│   └── imagenet/
├── models/
│   ├── resnet.py
│   ├── mobilenet.py
│   └── __init__.py
├── federated/
│   ├── client.py
│   ├── server.py
│   ├── compression.py
│   └── hardware_model.py
├── experiments/
│   ├── cifar10_experiment.py
│   ├── cifar100_experiment.py
│   └── imagenet_experiment.py
├── baselines/
│   ├── fedavg.py
│   ├── fedprox.py
│   ├── qsgd.py
│   └── fedpaq.py
├── utils/
│   ├── data_utils.py
│   ├── model_utils.py
│   └── visualization.py
└── configs/
    ├── device_profiles.yaml
    ├── training_config.yaml
    └── compression_config.yaml
```

---

## 2. Dataset Preparation

### CIFAR-10

```python
# Download and prepare CIFAR-10
from torchvision import datasets, transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(
    root='./data/cifar10',
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root='./data/cifar10',
    train=False,
    download=True,
    transform=transform_test
)
```

### CIFAR-100

```python
train_dataset = datasets.CIFAR100(
    root='./data/cifar100',
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = datasets.CIFAR100(
    root='./data/cifar100',
    train=False,
    download=True,
    transform=transform_test
)
```

### ImageNet (Subset)

```python
# Use ImageNet subset (100K samples)
# Download from ImageNet website and prepare subset
train_dataset = datasets.ImageFolder(
    root='./data/imagenet/train',
    transform=transform_train
)
```

---

## 3. Federated Data Splits

### IID Split

```python
import numpy as np
from torch.utils.data import Subset

def create_iid_split(dataset, num_clients, samples_per_client):
    """Create IID federated split"""
    indices = np.random.permutation(len(dataset))
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets
```

### Non-IID Split (Dirichlet Distribution)

```python
def create_noniid_split(dataset, num_clients, alpha=0.5):
    """Create non-IID federated split using Dirichlet distribution"""
    num_classes = len(dataset.classes)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Sample from Dirichlet distribution
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    client_datasets = []
    for client_id in range(num_clients):
        client_indices = []
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            num_samples = int(len(class_indices) * proportions[class_id, client_id])
            client_indices.extend(np.random.choice(class_indices, num_samples, replace=False))
        
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets
```

### Usage

```python
# CIFAR-10: 100 clients, 500 samples each (IID)
cifar10_iid = create_iid_split(train_dataset, num_clients=100, samples_per_client=500)

# CIFAR-10: 100 clients, non-IID (alpha=0.5)
cifar10_noniid = create_noniid_split(train_dataset, num_clients=100, alpha=0.5)

# CIFAR-10: 100 clients, extreme non-IID (alpha=0.1)
cifar10_extreme_noniid = create_noniid_split(train_dataset, num_clients=100, alpha=0.1)
```

---

## 4. Device Profile Configuration

### Device Profiles (YAML)

```yaml
# configs/device_profiles.yaml

devices:
  high_end_smartphone:
    cpu: "8-core ARM Cortex-A78 (2.84 GHz)"
    gpu: "Mali-G78 MP24"
    memory_gb: 8
    memory_type: "LPDDR5"
    bandwidth_5g_mbps: 100
    bandwidth_wifi_mbps: 1000
    power_peak_w: 5
    power_idle_w: 1
    throughput_fp32_gflops: 500
    throughput_int8_gflops: 2000
    capabilities:
      - full_precision
      - mixed_precision
      - sparse_updates
    
  mid_range_smartphone:
    cpu: "6-core ARM Cortex-A76 (2.2 GHz)"
    gpu: "Adreno 640"
    memory_gb: 4
    memory_type: "LPDDR4X"
    bandwidth_5g_mbps: 50
    bandwidth_wifi_mbps: 500
    power_peak_w: 3
    power_idle_w: 0.5
    throughput_fp32_gflops: 200
    throughput_int8_gflops: 800
    capabilities:
      - mixed_precision
      - sparse_updates
    
  iot_device:
    cpu: "ARM Cortex-A53 (1.4 GHz, 4 cores)"
    gpu: null
    memory_gb: 2
    memory_type: "LPDDR3"
    bandwidth_5g_mbps: 25
    bandwidth_wifi_mbps: 150
    power_peak_w: 1
    power_idle_w: 0.1
    throughput_fp32_gflops: 20
    throughput_int8_gflops: 80
    capabilities:
      - aggressive_quantization
      - sparse_updates
    
  edge_server:
    cpu: "Intel Xeon or ARM Neoverse-N1 (8+ cores)"
    gpu: "Optional NVIDIA Jetson or Edge TPU"
    memory_gb: 16
    memory_type: "DDR4"
    bandwidth_ethernet_mbps: 1000
    bandwidth_wifi_mbps: 1000
    power_peak_w: 20
    power_idle_w: 5
    throughput_fp32_gflops: 2000
    throughput_int8_gflops: 8000
    capabilities:
      - full_precision
      - mixed_precision
      - sparse_updates
      - assist_weaker_devices

device_distribution:
  high_end_smartphone: 0.30
  mid_range_smartphone: 0.40
  iot_device: 0.25
  edge_server: 0.05
```

### Device Profile Python Class

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DeviceProfile:
    device_type: str
    cpu: str
    gpu: Optional[str]
    memory_gb: int
    bandwidth_mbps: float
    power_peak_w: float
    throughput_fp32_gflops: float
    throughput_int8_gflops: float
    capabilities: List[str]
    
    def get_throughput(self, precision: str) -> float:
        """Get throughput for given precision"""
        if precision == 'fp32':
            return self.throughput_fp32_gflops
        elif precision == 'int8':
            return self.throughput_int8_gflops
        else:
            raise ValueError(f"Unknown precision: {precision}")
    
    def can_handle_precision(self, precision: str) -> bool:
        """Check if device can handle given precision"""
        if precision == 'fp32':
            return 'full_precision' in self.capabilities
        elif precision in ['int8', 'int4', 'int2']:
            return 'mixed_precision' in self.capabilities or 'aggressive_quantization' in self.capabilities
        return False
```

---

## 5. Model Architectures

### ResNet-18

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_resnet18(num_classes=10, pretrained=True):
    """Get ResNet-18 model"""
    model = resnet18(pretrained=pretrained)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

### ResNet-50

```python
from torchvision.models import resnet50

def get_resnet50(num_classes=10, pretrained=True):
    """Get ResNet-50 model"""
    model = resnet50(pretrained=pretrained)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

### MobileNet-V2

```python
from torchvision.models import mobilenet_v2

def get_mobilenet_v2(num_classes=10, pretrained=True):
    """Get MobileNet-V2 model"""
    model = mobilenet_v2(pretrained=pretrained)
    if num_classes != 1000:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
```

---

## 6. Training Configuration

### Training Parameters

```python
# configs/training_config.yaml

training:
  total_rounds: 200  # For CIFAR-10/100
  # total_rounds: 100  # For ImageNet
  
  client_selection:
    fraction: 0.1  # 10% of clients per round
    strategy: "random"
  
  local_training:
    epochs:
      high_end: 5
      mid_range: 3
      iot: 1
    
    batch_size:
      high_end: 64
      mid_range: 32
      iot: 16
    
    learning_rate: 0.01
    lr_schedule: "cosine"
    momentum: 0.9
    weight_decay: 1e-4
  
  optimizer: "SGD"
  
  device_dropout_rate: 0.1  # 10% dropout probability
```

### Training Script Template

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from federated.client import FedEdgeAccelClient
from federated.server import FedEdgeAccelServer

def train_federated_round(
    model,
    client_datasets,
    device_profiles,
    round_num,
    config
):
    """Train one federated round"""
    
    # Select clients
    num_selected = int(len(client_datasets) * config['client_selection']['fraction'])
    selected_clients = np.random.choice(len(client_datasets), num_selected, replace=False)
    
    # Create clients
    clients = []
    for idx in selected_clients:
        device_profile = device_profiles[idx]
        client = FedEdgeAccelClient(
            model=model,
            dataset=client_datasets[idx],
            device_profile=device_profile,
            config=config
        )
        clients.append(client)
    
    # Local training
    client_updates = []
    for client in clients:
        update = client.train_local()
        client_updates.append(update)
    
    # Server aggregation
    server = FedEdgeAccelServer(model, config)
    global_model = server.aggregate(client_updates)
    
    return global_model
```

---

## 7. Compression Parameters

### Quantization Configuration

```python
# configs/compression_config.yaml

compression:
  quantization:
    bit_widths: [2, 4, 8, 16]
    scheme: "symmetric_per_tensor"
    learnable_scales: true
    
    layer_wise:
      enabled: true
      sensitivity_method: "gradient_magnitude"  # or "hessian_trace"
    
    device_specific:
      high_end:
        weight_bits: 16
        activation_bits: 8
      mid_range:
        weight_bits: 8
        activation_bits: 4
      iot:
        weight_bits: 4
        activation_bits: 2
  
  sparsification:
    method: "top_k"  # or "threshold"
    top_k_ratio:
      high_end: 1.0  # No sparsification
      mid_range: 0.1  # Keep top 10%
      iot: 0.05  # Keep top 5%
    
    adaptive: true
    early_rounds_sparsity: 0.2
    late_rounds_sparsity: 0.1
  
  adaptive_compression:
    enabled: true
    alpha: 0.3
    beta: 0.1
    progress_metric: "loss_reduction"
```

### Compression Implementation

```python
import torch
import torch.nn as nn

class AdaptiveQuantizer(nn.Module):
    """Adaptive layer-wise quantizer"""
    
    def __init__(self, bit_width, learnable_scale=True):
        super().__init__()
        self.bit_width = bit_width
        self.learnable_scale = learnable_scale
        
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))
    
    def forward(self, x):
        """Quantize input"""
        if self.bit_width == 32:
            return x
        
        # Compute scale
        if self.learnable_scale:
            scale = self.scale
        else:
            scale = x.abs().max() / (2 ** (self.bit_width - 1) - 1)
        
        # Quantize
        quantized = torch.round(x / scale) * scale
        quantized = torch.clamp(quantized, -2**(self.bit_width-1), 2**(self.bit_width-1)-1)
        
        return quantized

class TopKSparsifier:
    """Top-k gradient sparsifier"""
    
    def __init__(self, k_ratio):
        self.k_ratio = k_ratio
    
    def sparsify(self, gradients):
        """Sparsify gradients using top-k"""
        flattened = gradients.flatten()
        k = int(len(flattened) * self.k_ratio)
        
        # Get top-k indices
        _, top_k_indices = torch.topk(flattened.abs(), k)
        
        # Create sparse mask
        mask = torch.zeros_like(flattened)
        mask[top_k_indices] = 1
        
        # Apply mask
        sparse_gradients = gradients * mask.reshape(gradients.shape)
        
        return sparse_gradients
```

---

## 8. Hardware Modeling

### Communication Cost Model

```python
class CommunicationCostModel:
    """Model communication cost for federated learning"""
    
    def __init__(self, device_profile):
        self.device_profile = device_profile
        self.radio_power_w = 0.5  # Radio power consumption
    
    def compute_cost(self, model_size_bits, bandwidth_mbps):
        """Compute communication cost"""
        # Transmission time (seconds)
        transmission_time = model_size_bits / (bandwidth_mbps * 1e6)
        
        # Energy cost (Joules)
        energy_cost = self.radio_power_w * transmission_time
        
        return {
            'time': transmission_time,
            'energy': energy_cost,
            'size_bits': model_size_bits
        }
```

### Computation Cost Model

```python
class ComputationCostModel:
    """Model computation cost for local training"""
    
    def __init__(self, device_profile):
        self.device_profile = device_profile
    
    def compute_cost(self, flops, precision='int8'):
        """Compute computation cost"""
        # Get device throughput
        throughput = self.device_profile.get_throughput(precision)
        
        # Training time (seconds)
        training_time = flops / (throughput * 1e9)  # Convert GFLOPS to FLOPS
        
        # Energy cost (Joules)
        energy_cost = self.device_profile.power_peak_w * training_time
        
        return {
            'time': training_time,
            'energy': energy_cost,
            'flops': flops
        }
```

### Unified Cost Model

```python
class UnifiedCostModel:
    """Unified cost model for communication and computation"""
    
    def __init__(self, device_profile, lambda_comm=0.1, lambda_comp=0.05):
        self.comm_model = CommunicationCostModel(device_profile)
        self.comp_model = ComputationCostModel(device_profile)
        self.lambda_comm = lambda_comm
        self.lambda_comp = lambda_comp
    
    def compute_total_cost(self, model_size_bits, flops, precision='int8', bandwidth_mbps=None):
        """Compute total cost"""
        if bandwidth_mbps is None:
            bandwidth_mbps = self.comm_model.device_profile.bandwidth_mbps
        
        comm_cost = self.comm_model.compute_cost(model_size_bits, bandwidth_mbps)
        comp_cost = self.comp_model.compute_cost(flops, precision)
        
        total_cost = (
            self.lambda_comm * comm_cost['energy'] +
            self.lambda_comp * comp_cost['energy']
        )
        
        return {
            'total_cost': total_cost,
            'comm_cost': comm_cost,
            'comp_cost': comp_cost
        }
```

---

## 9. Baseline Implementations

### FedAvg Baseline

```python
class FedAvgClient:
    """Standard FedAvg client"""
    
    def train_local(self, model, dataset, epochs=5, lr=0.01):
        """Local training without compression"""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return model.state_dict()
```

### QSGD Baseline

```python
class QSGDClient(FedAvgClient):
    """QSGD client with quantization"""
    
    def quantize_gradients(self, gradients, num_levels=8):
        """Quantize gradients using QSGD"""
        norm = gradients.norm()
        s = 2 ** num_levels - 1
        
        # Normalize and quantize
        normalized = gradients / norm
        quantized = torch.round(normalized * s) / s
        
        return quantized * norm
```

---

## 10. Evaluation Scripts

### Main Experiment Script

```python
# experiments/cifar10_experiment.py

import torch
from federated.client import FedEdgeAccelClient
from federated.server import FedEdgeAccelServer
from utils.data_utils import create_iid_split, create_noniid_split
from models import get_resnet18
from configs import load_config

def run_cifar10_experiment():
    """Run CIFAR-10 experiment"""
    
    # Load configuration
    config = load_config('configs/training_config.yaml')
    device_config = load_config('configs/device_profiles.yaml')
    
    # Prepare data
    train_dataset = load_cifar10_train()
    test_dataset = load_cifar10_test()
    
    # Create federated splits
    client_datasets_iid = create_iid_split(train_dataset, num_clients=100, samples_per_client=500)
    client_datasets_noniid = create_noniid_split(train_dataset, num_clients=100, alpha=0.5)
    
    # Initialize model
    model = get_resnet18(num_classes=10, pretrained=True)
    
    # Create device profiles
    device_profiles = create_device_profiles(device_config, num_clients=100)
    
    # Run experiments
    results_iid = train_federated(
        model=model,
        client_datasets=client_datasets_iid,
        device_profiles=device_profiles,
        config=config,
        scenario_name="CIFAR10_IID"
    )
    
    results_noniid = train_federated(
        model=model,
        client_datasets=client_datasets_noniid,
        device_profiles=device_profiles,
        config=config,
        scenario_name="CIFAR10_NonIID"
    )
    
    # Evaluate
    evaluate_model(model, test_dataset, results_iid, results_noniid)
    
    return results_iid, results_noniid

if __name__ == '__main__':
    results = run_cifar10_experiment()
```

### Evaluation Metrics

```python
def evaluate_model(model, test_dataset, results):
    """Evaluate model and compute metrics"""
    
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    
    # Compute communication cost
    comm_cost = sum([r['comm_cost'] for r in results])
    
    # Compute computation cost
    comp_cost = sum([r['comp_cost'] for r in results])
    
    return {
        'accuracy': accuracy,
        'comm_cost': comm_cost,
        'comp_cost': comp_cost,
        'total_cost': comm_cost + comp_cost
    }
```

---

## 11. Expected Results

### Communication Reduction

- **FedEdge-Accel**: 85-95% reduction vs. FedAvg
- **QSGD**: 60-75% reduction
- **FedPAQ**: 70-80% reduction

### Accuracy Preservation

- **FedEdge-Accel**: Within 0.5-1% of FedAvg baseline
- **QSGD**: 1-2% accuracy drop
- **FedPAQ**: 0.5-1.5% accuracy drop

### Convergence Speed

- **FedEdge-Accel**: 3-5× faster convergence (fewer rounds)
- **Baselines**: Standard convergence

### Local Training Efficiency

- **FedEdge-Accel**: 60-80% faster local training
- **Mixed-Precision (Baseline)**: 30-50% faster

---

## Running Experiments

### Step-by-Step Guide

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Datasets**
   ```bash
   python scripts/prepare_datasets.py --dataset cifar10
   python scripts/prepare_datasets.py --dataset cifar100
   ```

3. **Run CIFAR-10 Experiment**
   ```bash
   python experiments/cifar10_experiment.py --config configs/cifar10_config.yaml
   ```

4. **Run CIFAR-100 Experiment**
   ```bash
   python experiments/cifar100_experiment.py --config configs/cifar100_config.yaml
   ```

5. **Run Ablation Studies**
   ```bash
   python experiments/ablation_studies.py --config configs/ablation_config.yaml
   ```

6. **Generate Results and Plots**
   ```bash
   python scripts/generate_results.py --results_dir results/
   python scripts/generate_plots.py --results_dir results/ --output_dir plots/
   ```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Slow Training**: Use mixed-precision training or reduce model size
3. **Convergence Issues**: Adjust learning rate or increase local epochs
4. **Communication Errors**: Check network configuration and bandwidth settings

---

## Contact

For questions or issues, please contact the authors or open an issue on the project repository.
