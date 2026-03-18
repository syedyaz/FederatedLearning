"""
Compression utilities for federated learning: quantization and sparsification.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np


class AdaptiveQuantizer(nn.Module):
    """Adaptive layer-wise quantizer."""
    
    def __init__(self, bit_width: int = 8, learnable_scale: bool = True):
        super().__init__()
        self.bit_width = bit_width
        self.learnable_scale = learnable_scale
        
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor."""
        if self.bit_width == 32:
            return x
        
        # Compute scale
        if self.learnable_scale:
            scale = self.scale
        else:
            max_val = x.abs().max()
            if max_val > 0:
                scale = max_val / (2 ** (self.bit_width - 1) - 1)
            else:
                scale = torch.ones_like(self.scale)
        
        # Quantize
        quantized = torch.round(x / scale) * scale
        quantized = torch.clamp(
            quantized,
            -2 ** (self.bit_width - 1),
            2 ** (self.bit_width - 1) - 1
        )
        
        return quantized
    
    def quantize_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Quantize tensor and return quantized tensor and scale."""
        if self.bit_width == 32:
            return x, 1.0
        
        max_val = x.abs().max().item()
        if max_val == 0:
            return x, 1.0
        
        scale = max_val / (2 ** (self.bit_width - 1) - 1)
        quantized = torch.round(x / scale) * scale
        quantized = torch.clamp(
            quantized,
            -2 ** (self.bit_width - 1),
            2 ** (self.bit_width - 1) - 1
        )
        
        return quantized, scale


class TopKSparsifier:
    """Top-k gradient sparsifier."""
    
    def __init__(self, k_ratio: float = 0.1):
        """
        Args:
            k_ratio: Fraction of gradients to keep (0.1 = keep top 10%)
        """
        self.k_ratio = k_ratio
    
    def sparsify(self, gradients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sparsify gradients using top-k.
        
        Returns:
            sparse_gradients: Sparsified gradients
            mask: Binary mask indicating kept gradients
        """
        if self.k_ratio >= 1.0:
            return gradients, torch.ones_like(gradients, dtype=torch.bool)
        
        flattened = gradients.flatten()
        k = max(1, int(len(flattened) * self.k_ratio))
        
        # Get top-k indices
        _, top_k_indices = torch.topk(flattened.abs(), k)
        
        # Create sparse mask
        mask = torch.zeros_like(flattened, dtype=torch.bool)
        mask[top_k_indices] = True
        
        # Apply mask
        sparse_gradients = gradients * mask.reshape(gradients.shape).float()
        
        return sparse_gradients, mask.reshape(gradients.shape)


class ThresholdSparsifier:
    """Threshold-based gradient sparsifier."""
    
    def __init__(self, threshold_ratio: float = 0.01):
        """
        Args:
            threshold_ratio: Threshold as fraction of max gradient magnitude
        """
        self.threshold_ratio = threshold_ratio
    
    def sparsify(self, gradients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparsify gradients using threshold."""
        threshold = gradients.abs().max() * self.threshold_ratio
        mask = gradients.abs() > threshold
        sparse_gradients = gradients * mask.float()
        return sparse_gradients, mask


class STCCompressor:
    """Sparse Ternary Compression (STC) for federated learning."""
    
    def __init__(self, sparsity_ratio: float = 0.01):
        """
        Args:
            sparsity_ratio: Fraction of elements to keep (0.01 = keep top 1%)
        """
        self.sparsity_ratio = sparsity_ratio
    
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply STC to a tensor: top-K sparsification followed by ternarization.
        Returns the compressed tensor and a mask representing non-zero elements.
        """
        if self.sparsity_ratio >= 1.0:
            return tensor, torch.ones_like(tensor, dtype=torch.bool)
            
        flattened = tensor.flatten()
        k = max(1, int(len(flattened) * self.sparsity_ratio))
        
        # 1. Top-K Sparsification
        abs_tensor = flattened.abs()
        top_k_vals, top_k_indices = torch.topk(abs_tensor, k)
        
        mask = torch.zeros_like(flattened, dtype=torch.bool)
        mask[top_k_indices] = True
        
        # 2. Ternarization of the sparsified elements
        # Find the mean of the absolute values of the top-K elements
        mu = top_k_vals.mean()
        
        # Quantize the non-zero elements to {-mu, +mu}
        signs = flattened.sign()
        ternarized = mu * signs * mask.float()
        
        return ternarized.reshape(tensor.shape), mask.reshape(tensor.shape)


class LayerWiseCompressor:
    """Layer-wise compression with adaptive bit-widths."""
    
    def __init__(self, layer_bit_widths: Dict[str, int], sparsity_ratio: float = 0.1):
        """
        Args:
            layer_bit_widths: Dictionary mapping layer names to bit-widths
            sparsity_ratio: Sparsity ratio for gradients
        """
        self.layer_bit_widths = layer_bit_widths
        self.sparsity_ratio = sparsity_ratio
        self.quantizers = {}
        self.sparsifier = TopKSparsifier(k_ratio=sparsity_ratio)
    
    def compress_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress model state dict."""
        compressed = {}
        for name, tensor in state_dict.items():
            # Get bit-width for this layer (default to 8 if not specified)
            bit_width = self.layer_bit_widths.get(name, 8)
            
            # Quantize
            quantizer = AdaptiveQuantizer(bit_width=bit_width, learnable_scale=False)
            quantized, _ = quantizer.quantize_tensor(tensor)
            
            compressed[name] = quantized
        
        return compressed
    
    def compress_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress gradients with sparsification."""
        compressed = {}
        for name, grad in gradients.items():
            # Sparsify
            sparse_grad, _ = self.sparsifier.sparsify(grad)
            compressed[name] = sparse_grad
        
        return compressed


def compute_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Compute compression ratio."""
    return compressed_size / original_size if original_size > 0 else 1.0


def estimate_compressed_size(state_dict: Dict[str, torch.Tensor], bit_widths: Dict[str, int]) -> int:
    """Estimate compressed size in bits."""
    total_bits = 0
    for name, tensor in state_dict.items():
        bit_width = bit_widths.get(name, 8)
        total_bits += tensor.numel() * bit_width
    return total_bits


def adaptive_compression_ratio(round_num: int, total_rounds: int, alpha: float = 0.3, beta: float = 0.1) -> float:
    """
    Compute adaptive compression ratio based on training progress.
    
    Args:
        round_num: Current round number
        total_rounds: Total number of rounds
        alpha: Maximum compression (early rounds)
        beta: Minimum compression (late rounds)
    
    Returns:
        Compression ratio (lower = more compression)
    """
    progress = round_num / total_rounds
    compression_ratio = alpha * (1 - progress) + beta
    return max(beta, min(alpha, compression_ratio))
