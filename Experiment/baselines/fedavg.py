"""
FedAvg Baseline Implementation.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import copy

from federated.server import FedEdgeAccelServer


class FedAvgClient:
    """Standard FedAvg client without compression."""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataset,
        device_profile: Dict,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.dataset = dataset
        self.device_profile = device_profile
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.stats = {
            'samples_trained': 0
        }
    
    def train_local(
        self,
        num_epochs: Optional[int] = None,
        round_num: Optional[int] = None,
        total_rounds: Optional[int] = None
    ) -> Dict:
        """Perform local training (standard FedAvg)."""
        if num_epochs is None:
            num_epochs = int(self.device_profile.get('local_epochs', 3))
        
        batch_size = int(self.device_profile.get('batch_size', 32))
        base_lr = float(
            self.config.get('training', {}).get('local_training', {}).get('learning_rate')
            or self.device_profile.get('learning_rate', 0.004)
        )
        # LR schedule: warmup (constant) for first N rounds, then cosine decay
        warmup = int(self.config.get('training', {}).get('local_training', {}).get('lr_warmup_rounds', 50))
        if round_num is not None and total_rounds is not None and total_rounds > 0:
            if round_num < warmup:
                learning_rate = base_lr
            else:
                lr_min = base_lr * 0.01
                progress = (round_num - warmup) / max(1, total_rounds - warmup)
                progress = min(1.0, progress)
                learning_rate = lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * progress))
        else:
            learning_rate = base_lr
        # Ensure batch_size >= 2 so BatchNorm gets >1 sample per channel (avoids RuntimeError)
        batch_size = max(2, min(batch_size, len(self.dataset)))
        
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        
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
        
        initial_state = copy.deepcopy(self.model.state_dict())
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute model update
        model_update = {}
        for name, param in self.model.named_parameters():
            model_update[name] = param.data - initial_state[name]
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        effective_samples = len(self.dataset) if num_batches > 0 else 0
        self.stats['samples_trained'] += effective_samples * num_epochs
        
        return {
            'model_update': model_update,
            'num_samples': effective_samples,
            'loss': avg_loss
        }
    
    def get_statistics(self) -> Dict:
        """Get client statistics."""
        return self.stats.copy()


class FedAvgServer(FedEdgeAccelServer):
    """FedAvg server (inherits from FedEdgeAccelServer)."""
    pass
