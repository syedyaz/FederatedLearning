"""
Data utilities for federated learning experiments.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional
import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_cifar10_transforms(train=True):
    """Get CIFAR-10 data transforms."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


def get_cifar100_transforms(train=True):
    """Get CIFAR-100 data transforms."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])


def load_cifar10(data_dir='./data/cifar10', train=True):
    """Load CIFAR-10 dataset."""
    transform = get_cifar10_transforms(train=train)
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    return dataset


def load_cifar100(data_dir='./data/cifar100', train=True):
    """Load CIFAR-100 dataset."""
    transform = get_cifar100_transforms(train=train)
    dataset = datasets.CIFAR100(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    return dataset


def load_femnist(data_dir='./data/femnist', train=True):
    """
    Load FEMNIST dataset (Federated Extended MNIST).
    Uses torchvision EMNIST ByClass split which contains 62 classes (digits + letters).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.EMNIST(
        root=data_dir,
        split='byclass',
        train=train,
        download=True,
        transform=transform
    )
    return dataset


def load_har(data_dir='./data/har', train=True):
    """
    Load UCI Human Activity Recognition dataset.
    Downloads and formats the standard UCI HAR dataset for PyTorch.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'UCI_HAR_Dataset.zip')
    extracted_dir = os.path.join(data_dir, 'UCI HAR Dataset')
    
    if not os.path.exists(extracted_dir):
        if not os.path.exists(zip_path):
            print("Downloading UCI HAR Dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
            urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting UCI HAR Dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
    prefix = 'train' if train else 'test'
    
    # Load data
    x_path = os.path.join(extracted_dir, prefix, f'X_{prefix}.txt')
    y_path = os.path.join(extracted_dir, prefix, f'y_{prefix}.txt')
    
    # Read using pandas (space separated)
    X = pd.read_csv(x_path, sep='\s+', engine='python', header=None).values
    # y is 1-indexed in UCI HAR (1-6), convert to 0-indexed (0-5)
    y = pd.read_csv(y_path, sep='\s+', engine='python', header=None).values.flatten() - 1
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Attach labels attribute for compatibility with split functions
    dataset.targets = y.tolist()
    
    return dataset


def create_iid_split(dataset: Dataset, num_clients: int, samples_per_client: Optional[int] = None) -> List[Subset]:
    """
    Create IID federated split.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        samples_per_client: Samples per client (if None, divides evenly)
    
    Returns:
        List of Subset datasets, one per client
    """
    total_samples = len(dataset)
    if samples_per_client is None:
        samples_per_client = total_samples // num_clients
    
    # Shuffle indices
    indices = np.random.permutation(total_samples)
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = min((i + 1) * samples_per_client, total_samples)
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets


def create_noniid_split(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    samples_per_client: Optional[int] = None
) -> List[Subset]:
    """
    Create non-IID federated split using Dirichlet distribution.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet distribution parameter (smaller = more non-IID)
        samples_per_client: Target samples per client
    
    Returns:
        List of Subset datasets, one per client
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Extract labels from dataset
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    total_samples = len(dataset)
    
    if samples_per_client is None:
        samples_per_client = total_samples // num_clients
    
    # Sample from Dirichlet distribution
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # Assign samples to clients
    client_indices_list = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        
        # Distribute samples of this class to clients
        for client_id in range(num_clients):
            num_samples = int(len(class_indices) * proportions[class_id, client_id])
            if num_samples > 0 and len(class_indices) > 0:
                selected = class_indices[:num_samples]
                client_indices_list[client_id].extend(selected.tolist())
                class_indices = class_indices[num_samples:]
    
    # Create Subset datasets
    client_datasets = []
    for client_id in range(num_clients):
        if len(client_indices_list[client_id]) > 0:
            client_datasets.append(Subset(dataset, client_indices_list[client_id]))
        else:
            # If client has no samples, assign random samples
            random_indices = np.random.choice(total_samples, samples_per_client, replace=False)
            client_datasets.append(Subset(dataset, random_indices))
    
    return client_datasets


def create_device_profiles(num_clients: int, device_config: dict) -> List[dict]:
    """
    Create device profiles for clients.
    
    Args:
        num_clients: Number of clients
        device_config: Device configuration dictionary
    
    Returns:
        List of device profiles
    """
    device_types = list(device_config['device_distribution'].keys())
    probabilities = list(device_config['device_distribution'].values())
    
    # Sample device types for each client
    device_types_assigned = np.random.choice(
        device_types,
        size=num_clients,
        p=probabilities
    )
    
    device_profiles = []
    compression_by_device = device_config.get('compression_by_device', {})
    for device_type in device_types_assigned:
        profile = device_config['devices'][device_type].copy()
        profile['device_type'] = device_type
        # Merge compression params for this device type
        if device_type in compression_by_device:
            profile['compression'] = compression_by_device[device_type].copy()
        device_profiles.append(profile)
    
    return device_profiles


def get_data_statistics(client_datasets: List[Subset]) -> dict:
    """Get statistics about federated data distribution."""
    total_samples = sum(len(ds) for ds in client_datasets)
    
    # Count samples per client
    samples_per_client = [len(ds) for ds in client_datasets]
    
    # Count classes per client (if possible)
    classes_per_client = []
    for ds in client_datasets:
        if hasattr(ds.dataset, 'targets'):
            labels = [ds.dataset.targets[i] for i in ds.indices]
            classes_per_client.append(len(np.unique(labels)))
        else:
            classes_per_client.append(None)
    
    return {
        'num_clients': len(client_datasets),
        'total_samples': total_samples,
        'avg_samples_per_client': np.mean(samples_per_client),
        'std_samples_per_client': np.std(samples_per_client),
        'classes_per_client': classes_per_client
    }
