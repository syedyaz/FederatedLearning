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
        
        # Update global model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_update:
                    param.data += aggregated_update[name]
        
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
