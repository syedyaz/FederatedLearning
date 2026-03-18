"""
Federated Learning Server Implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import copy
import numpy as np


class FedEdgeAccelServer:
    """Federated Learning Server with Aggregation."""
    
    def __init__(self, model: nn.Module, config: Dict):
        """
        Args:
            model: Global model
            config: Server configuration
        """
        self.model = model
        self.config = config
        self.round_num = 0
        
        # Statistics
        self.stats = {
            'total_rounds': 0,
            'total_clients': 0,
            'aggregation_history': []
        }
    
    def aggregate(
        self,
        client_updates: List[Dict],
        client_weights: Optional[List[float]] = None
    ) -> Dict:
        """
        Aggregate client updates using weighted averaging.
        
        Args:
            client_updates: List of client model updates
            client_weights: Weights for each client (if None, uses uniform)
        
        Returns:
            Aggregated model update
        """
        if len(client_updates) == 0:
            return {}
        
        # Use uniform weights if not provided
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Aggregate updates
        aggregated_update = {}
        for name in client_updates[0].keys():
            aggregated_update[name] = torch.zeros_like(client_updates[0][name])
            for update, weight in zip(client_updates, client_weights):
                aggregated_update[name] += weight * update[name]
        
        # Debug: Check update magnitude and model change
        total_update_norm = sum(u.norm().item() for u in aggregated_update.values())
        max_update_abs = max(u.abs().max().item() for u in aggregated_update.values())
        
        # Store model state before update
        import logging
        logger = logging.getLogger(__name__)
        if self.round_num < 3:  # Log first 3 rounds
            model_norm_before = sum(p.norm().item() for p in self.model.parameters())
            logger.info(f"  [Round {self.round_num}] Before update: model_norm={model_norm_before:.6f}, "
                       f"update_norm={total_update_norm:.6f}, max_update_abs={max_update_abs:.6f}")
        
        # Update global model (only trainable parameters, not BatchNorm running stats)
        self.model.train()  # Set to train mode before updating
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_update:
                    # Skip BatchNorm running_mean/running_var - these are buffers, not parameters
                    # They should be recomputed during evaluation
                    if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                        continue
                    # Ensure update is on same device as param
                    update_tensor = aggregated_update[name].to(param.device)
                    param.data += update_tensor
        
        # Check model changed
        if self.round_num < 3:
            model_norm_after = sum(p.norm().item() for p in self.model.parameters())
            logger.info(f"  [Round {self.round_num}] After update: model_norm={model_norm_after:.6f}, "
                       f"change={abs(model_norm_after - model_norm_before):.6f}")
        
        self.round_num += 1
        self.stats['total_rounds'] = self.round_num
        self.stats['total_clients'] += len(client_updates)
        
        return aggregated_update
    
    def get_model_state(self) -> Dict:
        """Get current global model state."""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_state(self, state_dict: Dict):
        """Set global model state."""
        self.model.load_state_dict(state_dict)
    
    def evaluate(self, test_loader, device: Optional[torch.device] = None) -> Dict:
        """
        Evaluate global model on test set.
        
        Returns:
            Dictionary with accuracy and loss
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(device)
        self.model.eval()
        
        # Reset BatchNorm running stats and recompute from test data
        # This ensures consistent evaluation regardless of client-specific BatchNorm stats
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None  # Use current batch statistics
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                total_loss += loss.item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def get_statistics(self) -> Dict:
        """Get server statistics."""
        return self.stats.copy()
