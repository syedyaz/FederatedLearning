"""
Model utilities for federated learning experiments.
Includes CIFAR-10-appropriate ResNet (32x32 input) for publication-quality results.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, mobilenet_v2
from typing import Optional


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                _conv1x1(in_planes, planes, stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNetCIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10/32 (32x32 input).
    Uses 3x3 conv1 stride 1 and no initial maxpool to preserve spatial resolution.
    Standard in federated learning papers for CIFAR-10.
    """

    def __init__(self, block=_BasicBlock, num_blocks: tuple = (2, 2, 2, 2), num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def get_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 for CIFAR-10 (32x32). Use this for publication-quality FL experiments."""
    model = ResNetCIFAR(block=_BasicBlock, num_blocks=(2, 2, 2, 2), num_classes=num_classes)
    # Initialize weights properly (PyTorch default is Kaiming for conv, but ensure it's done)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    return model


def get_resnet18(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Get ImageNet-style ResNet-18. For CIFAR-10 prefer get_resnet18_cifar()."""
    try:
        model = resnet18(weights=None if not pretrained else 'IMAGENET1K_V1')
    except TypeError:
        model = resnet18(pretrained=pretrained)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet50(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Get ResNet-50 model."""
    try:
        model = resnet50(weights=None if not pretrained else 'IMAGENET1K_V1')
    except TypeError:
        model = resnet50(pretrained=pretrained)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_mobilenet_v2(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Get MobileNet-V2 model."""
    try:
        model = mobilenet_v2(weights=None if not pretrained else 'IMAGENET1K_V1')
    except TypeError:
        model = mobilenet_v2(pretrained=pretrained)
    if num_classes != 1000:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


class SimpleCNN_FEMNIST(nn.Module):
    """Simple 2-layer CNN for FEMNIST (input 1x28x28)."""
    def __init__(self, num_classes=62):
        super(SimpleCNN_FEMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP_HAR(nn.Module):
    """3-layer MLP for UCI HAR dataset (input 561 features)."""
    def __init__(self, input_size=561, hidden_size=256, num_classes=6):
        super(MLP_HAR, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def get_femnist_cnn(num_classes: int = 62) -> nn.Module:
    """Get CNN model for FEMNIST."""
    return SimpleCNN_FEMNIST(num_classes=num_classes)


def get_har_mlp(input_size: int = 561, num_classes: int = 6) -> nn.Module:
    """Get MLP model for HAR."""
    return MLP_HAR(input_size=input_size, num_classes=num_classes)


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
    if 'ResNetCIFAR' in type(model).__name__ or 'resnet18' in str(type(model)).lower():
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
