"""
Federated Learning Client Implementation with Compression and Hardware-Aware Training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import copy
import numpy as np

from federated.compression import LayerWiseCompressor, AdaptiveQuantizer, TopKSparsifier
from federated.hardware_model import UnifiedCostModel, estimate_flops_per_sample
from utils.model_utils import estimate_flops


class FedEdgeAccelClient:
    """Federated Learning Client with Compression and Hardware-Aware Training."""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataset,
        device_profile: Dict,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            client_id: Unique client identifier
            model: Neural network model
            dataset: Local dataset
            device_profile: Device hardware profile
            config: Training configuration
            device: PyTorch device (CPU/GPU)
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.dataset = dataset
        self.device_profile = device_profile
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup compression
        self.setup_compression()
        
        # Setup cost model
        # Safely convert config values to float
        lambda_comm = float(config.get('compression', {}).get('cost_weights', {}).get('lambda_comm', 0.1))
        lambda_comp = float(config.get('compression', {}).get('cost_weights', {}).get('lambda_comp', 0.05))
        
        self.cost_model = UnifiedCostModel(
            device_profile,
            lambda_comm=lambda_comm,
            lambda_comp=lambda_comp
        )
        
        # Training statistics
        self.stats = {
            'comm_cost': 0.0,
            'comp_cost': 0.0,
            'total_cost': 0.0,
            'samples_trained': 0
        }
    
    def setup_compression(self):
        """Setup compression parameters based on device profile."""
        compression_config = self.device_profile.get('compression', {})
        
        # Get bit-widths for this device type (ensure int/float types)
        self.weight_bits = int(compression_config.get('weight_bits', 8))
        self.activation_bits = int(compression_config.get('activation_bits', 4))
        self.sparsity_ratio = float(compression_config.get('sparsity_ratio', 0.1))
        
        # Create compressor
        layer_bit_widths = self._get_layer_bit_widths()
        self.compressor = LayerWiseCompressor(layer_bit_widths, self.sparsity_ratio)
        self.sparsifier = TopKSparsifier(k_ratio=self.sparsity_ratio)
    
    def _get_layer_bit_widths(self) -> Dict[str, int]:
        """Get bit-width for each layer."""
        layer_bit_widths = {}
        for name, _ in self.model.named_parameters():
            layer_bit_widths[name] = self.weight_bits
        return layer_bit_widths
    
    def train_local(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Perform local training with hardware-aware optimization.
        
        Returns:
            Dictionary with model update and statistics
        """
        if num_epochs is None:
            num_epochs = int(self.device_profile.get('local_epochs', 3))
        
        batch_size = int(self.device_profile.get('batch_size', 32))
        learning_rate = float(self.device_profile.get('learning_rate', 0.01))
        # Ensure batch_size >= 2 so BatchNorm gets >1 sample per channel (avoids RuntimeError)
        batch_size = max(2, min(batch_size, len(self.dataset)))
        
        # Create data loader; drop_last=True avoids batch size 1 which breaks BatchNorm in training
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Setup optimizer
        # Safely convert config values to float
        momentum = float(self.config.get('training', {}).get('local_training', {}).get('momentum', 0.9))
        weight_decay = float(self.config.get('training', {}).get('local_training', {}).get('weight_decay', 1e-4))
        
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Store initial model state
        initial_state = copy.deepcopy(self.model.state_dict())
        
        # Local training
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_samples = len(self.dataset)
        batches_per_epoch = len(dataloader)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Apply sparsification to gradients
                if self.sparsity_ratio < 1.0:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            sparse_grad, _ = self.sparsifier.sparsify(param.grad)
                            param.grad = sparse_grad
                
                optimizer.step()
                
                batch_loss = loss.item()
                total_loss += batch_loss
                epoch_loss += batch_loss
                num_batches += 1
                epoch_batches += 1
                
                # Log progress every 10% of batches in epoch
                if (batch_idx + 1) % max(1, batches_per_epoch // 10) == 0:
                    progress = ((batch_idx + 1) / batches_per_epoch) * 100
                    avg_loss = epoch_loss / epoch_batches
                    # Only log if logging is configured (avoid errors if not)
                    import logging
                    logger = logging.getLogger(__name__)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"      Client {self.client_id} Epoch {epoch+1}/{num_epochs}: "
                                   f"{progress:.0f}% | Batch {batch_idx+1}/{batches_per_epoch} | "
                                   f"Loss: {avg_loss:.4f}")
        
        # Compute model update
        model_update = {}
        for name, param in self.model.named_parameters():
            model_update[name] = param.data - initial_state[name]
        
        # If no batches were run (e.g. too few samples after drop_last), treat as zero contribution
        effective_samples = len(self.dataset) if num_batches > 0 else 0
        
        # Compute statistics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        num_samples = effective_samples * num_epochs if num_batches > 0 else 0
        
        # Estimate computation cost
        model_name = type(self.model).__name__.lower()
        flops_per_sample = estimate_flops_per_sample(model_name)
        total_flops = flops_per_sample * num_samples
        
        precision = 'int8' if self.weight_bits == 8 else ('int4' if self.weight_bits == 4 else 'fp32')
        comp_cost = self.cost_model.comp_model.compute_cost(total_flops, precision, num_epochs) if num_batches > 0 else {'energy': 0.0}
        
        self.stats['comp_cost'] += comp_cost['energy']
        self.stats['samples_trained'] += num_samples
        
        return {
            'model_update': model_update,
            'num_samples': effective_samples,
            'loss': avg_loss,
            'comp_cost': comp_cost
        }
    
    def compress_update(self, model_update: Dict, round_num: int, total_rounds: int) -> Tuple[Dict, Dict]:
        """
        Compress model update for communication.
        
        Returns:
            compressed_update: Compressed model update
            compression_stats: Statistics about compression
        """
        # Adaptive compression ratio
        if self.config.get('compression', {}).get('adaptive', {}).get('enabled', False):
            compression_ratio = self._get_adaptive_compression_ratio(round_num, total_rounds)
            # Adjust sparsity based on compression ratio
            effective_sparsity = self.sparsity_ratio * compression_ratio
        else:
            effective_sparsity = self.sparsity_ratio
        
        # Compress update
        compressed_update = self.compressor.compress_state_dict(model_update)
        
        # Apply sparsification
        if effective_sparsity < 1.0:
            sparsifier = TopKSparsifier(k_ratio=effective_sparsity)
            for name in compressed_update:
                sparse_grad, _ = sparsifier.sparsify(compressed_update[name])
                compressed_update[name] = sparse_grad
        
        # Compute compression statistics
        original_size = sum(p.numel() * 32 for p in model_update.values())  # Assume FP32
        compressed_size = sum(p.numel() * self.weight_bits for p in compressed_update.values())
        compression_ratio_actual = compressed_size / original_size if original_size > 0 else 1.0
        
        # Estimate communication cost
        comm_cost = self.cost_model.comm_model.compute_cost(compressed_size * 8)  # Convert bytes to bits
        
        self.stats['comm_cost'] += comm_cost['energy']
        self.stats['total_cost'] = self.stats['comm_cost'] + self.stats['comp_cost']
        
        compression_stats = {
            'original_size_bits': original_size * 32,
            'compressed_size_bits': compressed_size * 8,
            'compression_ratio': compression_ratio_actual,
            'comm_cost': comm_cost
        }
        
        return compressed_update, compression_stats
    
    def _get_adaptive_compression_ratio(self, round_num: int, total_rounds: int) -> float:
        """Get adaptive compression ratio based on training progress."""
        # Safely convert config values to float
        alpha = float(self.config.get('compression', {}).get('adaptive', {}).get('alpha', 0.3))
        beta = float(self.config.get('compression', {}).get('adaptive', {}).get('beta', 0.1))
        progress = round_num / total_rounds
        compression_ratio = alpha * (1 - progress) + beta
        return max(beta, min(alpha, compression_ratio))
    
    def get_statistics(self) -> Dict:
        """Get client statistics."""
        return self.stats.copy()
