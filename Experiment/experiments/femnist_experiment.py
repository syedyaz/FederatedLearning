"""
FEMNIST Federated Learning Experiment.
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import logging
import time

from utils.data_utils import load_femnist, create_iid_split, create_noniid_split, get_data_statistics, create_device_profiles
from utils.model_utils import get_femnist_cnn, get_model_size_mb, estimate_flops
from federated.client import FedEdgeAccelClient
from federated.server import FedEdgeAccelServer
from baselines.fedavg import FedAvgClient, FedAvgServer
from typing import Optional


def setup_logging(results_dir: str):
    """Setup logging configuration."""
    log_file = os.path.join(results_dir, 'experiment.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility (publication requirement)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True


def run_federated_learning(
    model,
    train_datasets,
    test_dataset,
    device_profiles,
    config,
    baseline: bool = False,
    logger: Optional[logging.Logger] = None
):
    """Run federated learning simulation."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    method_name = "FedAvg" if baseline else "FedEdge-Accel"
    logger.info(f"Initializing {method_name} server for FEMNIST Experiment on device: {device}")
    
    # Create server
    if baseline:
        server = FedAvgServer(model, config)
    else:
        server = FedEdgeAccelServer(model, config)
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Training configuration
    total_rounds = config['training']['total_rounds']
    fraction = config['training']['client_selection']['fraction']
    num_selected = max(1, int(len(train_datasets) * fraction))
    eval_frequency = config['training'].get('eval_frequency', 5)
    
    # Results storage
    results = {
        'rounds': [],
        'accuracies': [],
        'losses': [],
        'comm_costs': [],
        'comp_costs': [],
        'total_costs': []
    }
    
    logger.info(f"Starting federated learning simulation")
    logger.info(f"  Method: {method_name}")
    logger.info(f"  Total rounds: {total_rounds}")
    logger.info(f"  Total clients: {len(train_datasets)}")
    logger.info(f"  Clients per round: {num_selected} ({fraction*100:.1f}%)")
    logger.info(f"  Evaluation frequency: Every {eval_frequency} rounds")
    logger.info(f"  Device: {device}")
    
    start_time = time.time()
    
    for round_num in tqdm(range(total_rounds), desc=f"{method_name} Training"):
        round_start_time = time.time()
        progress_pct = ((round_num + 1) / total_rounds) * 100
        
        logger.info(f"[Round {round_num + 1}/{total_rounds}] Progress: {progress_pct:.1f}%")
        
        # Select clients
        selected_indices = np.random.choice(len(train_datasets), num_selected, replace=False)
        logger.debug(f"  Selected clients: {sorted(selected_indices)}")
        
        # Create clients and train (don't keep client refs to free GPU memory between clients)
        client_updates = []
        client_weights = []
        comm_costs_round = []  # For FedEdge-Accel cost logging (avoids keeping client refs)
        comp_costs_round = []
        
        logger.info(f"  Training {num_selected} clients locally...")
        client_training_times = []
        
        for client_idx, idx in enumerate(selected_indices):
            client_start_time = time.time()
            
            if baseline:
                client = FedAvgClient(
                    client_id=idx,
                    model=model,
                    dataset=train_datasets[idx],
                    device_profile=device_profiles[idx],
                    config=config,
                    device=device
                )
            else:
                client = FedEdgeAccelClient(
                    client_id=idx,
                    model=model,
                    dataset=train_datasets[idx],
                    device_profile=device_profiles[idx],
                    config=config,
                    device=device
                )
            
            # Local training (pass round for cosine LR decay)
            train_result = client.train_local(
                round_num=round_num,
                total_rounds=total_rounds
            )
            client_training_time = time.time() - client_start_time
            client_training_times.append(client_training_time)
            
            logger.debug(f"    Client {idx}: {len(train_datasets[idx])} samples, "
                        f"loss={train_result['loss']:.4f}, "
                        f"time={client_training_time:.2f}s")
            
            # Compress update (only for FedEdge-Accel)
            if not baseline:
                compressed_update, comp_stats = client.compress_update(
                    train_result['model_update'],
                    round_num,
                    total_rounds
                )
                client_updates.append(compressed_update)
                comm_costs_round.append(client.get_statistics()['comm_cost'])
                comp_costs_round.append(client.get_statistics()['comp_cost'])
                logger.debug(f"    Client {idx} compression: "
                           f"ratio={comp_stats['compression_ratio']:.3f}, "
                           f"comm_cost={comp_stats['comm_cost']['energy']:.2f}")
            else:
                client_updates.append(train_result['model_update'])
            
            client_weights.append(train_result['num_samples'])
            
            # Free client GPU memory before next client (critical for 4GB GPUs)
            del client
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        avg_client_time = np.mean(client_training_times)
        logger.info(f"  Average client training time: {avg_client_time:.2f}s")
        
        # Aggregate updates
        logger.info(f"  Aggregating updates from {len(client_updates)} clients...")
        agg_start_time = time.time()
        server.aggregate(client_updates, client_weights)
        agg_time = time.time() - agg_start_time
        logger.debug(f"  Aggregation completed in {agg_time:.2f}s")
        
        # CRITICAL: Ensure model reference is synced (server.model and model should be same object)
        # But to be safe, explicitly sync model state from server after aggregation
        model.load_state_dict(server.model.state_dict())
        
        # Evaluate periodically
        if (round_num + 1) % eval_frequency == 0:
            logger.info(f"  Evaluating global model...")
            eval_start_time = time.time()
            eval_result = server.evaluate(test_loader, device)
            eval_time = time.time() - eval_start_time
            
            results['rounds'].append(round_num + 1)
            results['accuracies'].append(eval_result['accuracy'])
            results['losses'].append(eval_result['loss'])
            
            # Compute costs (for FedEdge-Accel)
            if not baseline:
                total_comm_cost = sum(comm_costs_round)
                total_comp_cost = sum(comp_costs_round)
                results['comm_costs'].append(total_comm_cost)
                results['comp_costs'].append(total_comp_cost)
                results['total_costs'].append(total_comm_cost + total_comp_cost)
                
                logger.info(f"  Evaluation Results:")
                logger.info(f"    Accuracy: {eval_result['accuracy']:.2f}%")
                logger.info(f"    Loss: {eval_result['loss']:.4f}")
                logger.info(f"    Communication Cost: {total_comm_cost:.2f}")
                logger.info(f"    Computation Cost: {total_comp_cost:.2f}")
                logger.info(f"    Total Cost: {total_comm_cost + total_comp_cost:.2f}")
            else:
                logger.info(f"  Evaluation Results:")
                logger.info(f"    Accuracy: {eval_result['accuracy']:.2f}%")
                logger.info(f"    Loss: {eval_result['loss']:.4f}")
            
            logger.debug(f"  Evaluation time: {eval_time:.2f}s")
        
        round_time = time.time() - round_start_time
        elapsed_time = time.time() - start_time
        avg_round_time = elapsed_time / (round_num + 1)
        estimated_remaining = avg_round_time * (total_rounds - round_num - 1)
        
        logger.info(f"  Round completed in {round_time:.2f}s")
        logger.info(f"  Elapsed: {elapsed_time/60:.1f}min | "
                   f"Avg/round: {avg_round_time:.2f}s | "
                   f"Est. remaining: {estimated_remaining/60:.1f}min")
    
    total_time = time.time() - start_time
    logger.info(f"Federated learning completed!")
    logger.info(f"  Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    logger.info(f"  Average time per round: {total_time/total_rounds:.2f}s")
    
    return results, server


def main():
    """Main experiment function."""
    # Reproducibility: set seeds before any randomness (required for publication)
    seed = int(os.environ.get('FL_SEED', 42))
    set_seed(seed)
    
    # Create results directory first for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/femnist_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(results_dir)
    logger.info("="*70)
    logger.info("FEMNIST Federated Learning Experiment (Publication-Ready)")
    logger.info("="*70)
    logger.info(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Random seed: {seed}")
    
    # Load configurations
    logger.info("Loading configuration files...")
    config = load_config('configs/training_config.yaml')
    device_config = load_config('configs/device_profiles.yaml')
    logger.info("Configuration loaded successfully")
    
    print("\n--- Phase 2: Loading FEMNIST Dataset ---")
    data_dir = config.get('data', {}).get('data_dir', './data/femnist')
    
    print(f"Loading FEMNIST from {data_dir}...")
    load_start = time.time()
    train_dataset = load_femnist(data_dir=data_dir, train=True)
    test_dataset = load_femnist(data_dir=data_dir, train=False)
    load_time = time.time() - load_start
    logger.info(f"Dataset loaded in {load_time:.2f}s")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Create federated splits
    num_clients = config['datasets']['femnist']['num_clients']
    
    logger.info(f"Creating federated splits for {num_clients} clients...")
    split_start = time.time()
    
    logger.info(f"  Creating IID split...")
    train_datasets_iid = create_iid_split(
        train_dataset,
        num_clients,
        config['datasets']['femnist']['samples_per_client_iid']
    )
    
    logger.info(f"  Creating non-IID split (alpha={config['datasets']['femnist']['noniid_alpha']})...")
    train_datasets_noniid = create_noniid_split(
        train_dataset,
        num_clients,
        alpha=config['datasets']['femnist']['noniid_alpha']
    )
    
    split_time = time.time() - split_start
    logger.info(f"Federated splits created in {split_time:.2f}s")
    
    # Create device profiles
    logger.info("Creating device profiles...")
    device_profiles = create_device_profiles(num_clients, device_config)
    logger.info(f"Created {len(device_profiles)} device profiles")
    
    # CIFAR-10-appropriate ResNet (32x32) for publication-quality accuracy
    logger.info("Initializing baseline model FedAvg (CNN_FEMNIST)...")
    baseline_model = get_femnist_cnn(num_classes=62)
    logger.info("Model initialized")
    
    # Run experiments
    scenarios = [
        ('IID', train_datasets_iid),
        ('NonIID', train_datasets_noniid)
    ]
    
    experiment_start_time = time.time()
    
    for scenario_name, train_datasets in scenarios:
        logger.info("")
        logger.info("="*70)
        logger.info(f"Running {scenario_name} scenario")
        logger.info("="*70)
        
        scenario_start_time = time.time()
        
        # FedEdge-Accel
        logger.info("")
        logger.info("-"*70)
        logger.info("Method: FedEdge-Accel")
        logger.info("-"*70)
        model_fededge = get_femnist_cnn(num_classes=62)
        results_fededge, server_fededge = run_federated_learning(
            model_fededge,
            train_datasets,
            test_dataset,
            device_profiles,
            config,
            baseline=False,
            logger=logger
        )
        
        # Save results
        results_file = os.path.join(results_dir, f'{scenario_name}_fededge_accel.json')
        with open(results_file, 'w') as f:
            json.dump(results_fededge, f, indent=2)
        logger.info(f"FedEdge-Accel results saved to: {results_file}")
        
        # Baseline: FedAvg
        logger.info("")
        logger.info("-"*70)
        logger.info("Method: FedAvg Baseline")
        logger.info("-"*70)
        model_fedavg = get_femnist_cnn(num_classes=62)
        results_fedavg, server_fedavg = run_federated_learning(
            model_fedavg,
            train_datasets,
            test_dataset,
            device_profiles,
            config,
            baseline=True,
            logger=logger
        )
        
        # Save results
        results_file = os.path.join(results_dir, f'{scenario_name}_fedavg.json')
        with open(results_file, 'w') as f:
            json.dump(results_fedavg, f, indent=2)
        logger.info(f"FedAvg results saved to: {results_file}")
        
        scenario_time = time.time() - scenario_start_time
        logger.info(f"{scenario_name} scenario completed in {scenario_time/60:.1f} minutes")
    
    total_experiment_time = time.time() - experiment_start_time
    logger.info("")
    logger.info("="*70)
    logger.info("Experiment Summary")
    logger.info("="*70)
    logger.info(f"Total experiment time: {total_experiment_time/60:.1f} minutes ({total_experiment_time:.1f} seconds)")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    print(f"\n{'='*70}")
    print(f"Experiment completed!")
    print(f"Results saved to: {results_dir}")
    print(f"Total time: {total_experiment_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
