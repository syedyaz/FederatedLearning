"""
Setup script for FedEdge-Accel experiments.
"""

import os
import subprocess
import sys


def create_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'results',
        'plots',
        'logs',
        'checkpoints'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies."""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    print("Dependencies installed successfully!")


def main():
    """Main setup function."""
    print("Setting up FedEdge-Accel experiment environment...")
    
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Installing dependencies...")
    try:
        install_dependencies()
    except subprocess.CalledProcessError:
        print("Warning: Failed to install some dependencies. Please install manually:")
        print("  pip install -r requirements.txt")
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("  1. Prepare datasets: python scripts/prepare_datasets.py --dataset cifar10")
    print("  2. Run experiment: python experiments/cifar10_experiment.py")
    print("  3. Generate plots: python scripts/generate_plots.py --results_dir results/...")


if __name__ == '__main__':
    main()
