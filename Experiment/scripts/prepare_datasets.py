"""
Prepare datasets for federated learning experiments.
"""

import argparse
import os
from utils.data_utils import load_cifar10, load_cifar100


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], required=True)
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    
    args = parser.parse_args()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    print(f"Preparing {args.dataset} dataset...")
    
    if args.dataset == 'cifar10':
        print("Loading CIFAR-10 train set...")
        train_dataset = load_cifar10(data_dir=os.path.join(args.data_dir, 'cifar10'), train=True)
        print(f"Train set: {len(train_dataset)} samples")
        
        print("Loading CIFAR-10 test set...")
        test_dataset = load_cifar10(data_dir=os.path.join(args.data_dir, 'cifar10'), train=False)
        print(f"Test set: {len(test_dataset)} samples")
        
    elif args.dataset == 'cifar100':
        print("Loading CIFAR-100 train set...")
        train_dataset = load_cifar100(data_dir=os.path.join(args.data_dir, 'cifar100'), train=True)
        print(f"Train set: {len(train_dataset)} samples")
        
        print("Loading CIFAR-100 test set...")
        test_dataset = load_cifar100(data_dir=os.path.join(args.data_dir, 'cifar100'), train=False)
        print(f"Test set: {len(test_dataset)} samples")
    
    print("Dataset preparation complete!")


if __name__ == '__main__':
    main()
