"""
Model utilities for federated learning experiments.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, mobilenet_v2
from typing import Optional


def get_resnet18(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Get ResNet-18 model."""
    model = resnet18(pretrained=pretrained)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet50(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Get ResNet-50 model."""
    model = resnet50(pretrained=pretrained)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_mobilenet_v2(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Get MobileNet-V2 model."""
    model = mobilenet_v2(pretrained=pretrained)
    if num_classes != 1000:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module, precision_bits: int = 32) -> float:
    """Get model size in MB."""
    num_params = count_parameters(model)
    size_bits = num_params * precision_bits
    size_mb = size_bits / (8 * 1024 * 1024)
    return size_mb


def estimate_flops(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> int:
    """Estimate FLOPs for a forward pass."""
    # Simple estimation based on model type and input size
    # For more accurate FLOPs, use thop or fvcore
    if 'resnet18' in str(type(model)).lower():
        # Approximate FLOPs for ResNet-18: ~1.8 GFLOPs for 32x32 input
        return int(1.8e9)
    elif 'resnet50' in str(type(model)).lower():
        return int(4.1e9)
    elif 'mobilenet' in str(type(model)).lower():
        return int(0.3e9)
    else:
        # Rough estimate: params * 2 (multiply-add operations)
        return count_parameters(model) * 2


def load_model_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def save_model_checkpoint(model: nn.Module, checkpoint_path: str, additional_info: Optional[dict] = None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    if additional_info:
        checkpoint.update(additional_info)
    torch.save(checkpoint, checkpoint_path)
