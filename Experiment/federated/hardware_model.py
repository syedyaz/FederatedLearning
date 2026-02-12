"""
Hardware modeling for communication and computation costs.
"""

from typing import Dict, Optional
import numpy as np


class CommunicationCostModel:
    """Model communication cost for federated learning."""
    
    def __init__(self, device_profile: Dict):
        """
        Args:
            device_profile: Device profile dictionary with bandwidth and power info
        """
        self.device_profile = device_profile
        self.radio_power_w = device_profile.get('power_peak_w', 0.5) * 0.1  # 10% of peak for radio
    
    def compute_cost(self, model_size_bits: int, bandwidth_mbps: Optional[float] = None) -> Dict:
        """
        Compute communication cost.
        
        Args:
            model_size_bits: Size of model update in bits
            bandwidth_mbps: Bandwidth in Mbps (if None, uses device default)
        
        Returns:
            Dictionary with time (seconds) and energy (Joules)
        """
        if bandwidth_mbps is None:
            # Use WiFi bandwidth as default
            bandwidth_mbps = self.device_profile.get('bandwidth_wifi_mbps', 100)
        
        # Transmission time (seconds)
        transmission_time = model_size_bits / (bandwidth_mbps * 1e6)
        
        # Energy cost (Joules)
        energy_cost = self.radio_power_w * transmission_time
        
        return {
            'time': transmission_time,
            'energy': energy_cost,
            'size_bits': model_size_bits,
            'bandwidth_mbps': bandwidth_mbps
        }


class ComputationCostModel:
    """Model computation cost for local training."""
    
    def __init__(self, device_profile: Dict):
        """
        Args:
            device_profile: Device profile dictionary with throughput and power info
        """
        self.device_profile = device_profile
    
    def get_throughput(self, precision: str = 'int8') -> float:
        """Get device throughput for given precision."""
        if precision == 'fp32':
            return self.device_profile.get('throughput_fp32_gflops', 100)
        elif precision == 'int8':
            return self.device_profile.get('throughput_int8_gflops', 400)
        elif precision == 'int4':
            return self.device_profile.get('throughput_int4_gflops', 800)
        else:
            return self.device_profile.get('throughput_int8_gflops', 400)
    
    def compute_cost(self, flops: int, precision: str = 'int8', num_epochs: int = 1) -> Dict:
        """
        Compute computation cost.
        
        Args:
            flops: Number of floating-point operations
            precision: Precision used (fp32, int8, int4)
            num_epochs: Number of local training epochs
        
        Returns:
            Dictionary with time (seconds) and energy (Joules)
        """
        # Get device throughput
        throughput_gflops = self.get_throughput(precision)
        
        # Training time (seconds) - multiply by epochs
        training_time = (flops / (throughput_gflops * 1e9)) * num_epochs
        
        # Energy cost (Joules)
        power_w = self.device_profile.get('power_peak_w', 5.0)
        energy_cost = power_w * training_time
        
        return {
            'time': training_time,
            'energy': energy_cost,
            'flops': flops,
            'throughput_gflops': throughput_gflops,
            'precision': precision
        }


class UnifiedCostModel:
    """Unified cost model for communication and computation."""
    
    def __init__(self, device_profile: Dict, lambda_comm: float = 0.1, lambda_comp: float = 0.05):
        """
        Args:
            device_profile: Device profile dictionary
            lambda_comm: Weight for communication cost
            lambda_comp: Weight for computation cost
        """
        self.comm_model = CommunicationCostModel(device_profile)
        self.comp_model = ComputationCostModel(device_profile)
        self.lambda_comm = lambda_comm
        self.lambda_comp = lambda_comp
    
    def compute_total_cost(
        self,
        model_size_bits: int,
        flops: int,
        precision: str = 'int8',
        num_epochs: int = 1,
        bandwidth_mbps: Optional[float] = None
    ) -> Dict:
        """
        Compute total cost (communication + computation).
        
        Returns:
            Dictionary with total cost and breakdown
        """
        comm_cost = self.comm_model.compute_cost(model_size_bits, bandwidth_mbps)
        comp_cost = self.comp_model.compute_cost(flops, precision, num_epochs)
        
        total_cost = (
            self.lambda_comm * comm_cost['energy'] +
            self.lambda_comp * comp_cost['energy']
        )
        
        return {
            'total_cost': total_cost,
            'comm_cost': comm_cost,
            'comp_cost': comp_cost,
            'total_time': comm_cost['time'] + comp_cost['time'],
            'total_energy': comm_cost['energy'] + comp_cost['energy']
        }


def estimate_model_size_bits(state_dict: Dict, bit_widths: Dict[str, int]) -> int:
    """Estimate model size in bits given quantization bit-widths."""
    total_bits = 0
    for name, tensor in state_dict.items():
        bit_width = bit_widths.get(name, 8)
        total_bits += tensor.numel() * bit_width
    return total_bits


def estimate_flops_per_sample(model_name: str, input_size: tuple = (3, 32, 32)) -> int:
    """Estimate FLOPs per sample for common models."""
    # Rough estimates for CIFAR-10 (32x32 images)
    flops_estimates = {
        'resnet18': int(1.8e9),
        'resnet50': int(4.1e9),
        'mobilenet_v2': int(0.3e9),
    }
    
    model_key = model_name.lower()
    for key, flops in flops_estimates.items():
        if key in model_key:
            return flops
    
    # Default estimate
    return int(1e9)
