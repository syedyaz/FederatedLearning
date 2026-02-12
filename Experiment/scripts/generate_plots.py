"""
Generate plots from experiment results.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import argparse


def load_results(results_dir: str) -> Dict:
    """Load all results from results directory."""
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                key = filename.replace('.json', '')
                results[key] = json.load(f)
    return results


def plot_convergence(results: Dict, output_dir: str):
    """Plot convergence curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy curves
    ax1 = axes[0]
    for key, data in results.items():
        if 'rounds' in data and 'accuracies' in data:
            ax1.plot(data['rounds'], data['accuracies'], label=key.replace('_', ' '))
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Convergence: Accuracy vs Rounds')
    ax1.legend()
    ax1.grid(True)
    
    # Loss curves
    ax2 = axes[1]
    for key, data in results.items():
        if 'rounds' in data and 'losses' in data:
            ax2.plot(data['rounds'], data['losses'], label=key.replace('_', ' '))
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Convergence: Loss vs Rounds')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_curves.png'), dpi=300)
    plt.close()


def plot_communication_cost(results: Dict, output_dir: str):
    """Plot communication cost comparison."""
    if 'comm_costs' not in list(results.values())[0]:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for key, data in results.items():
        if 'comm_costs' in data and len(data['comm_costs']) > 0:
            rounds = data.get('rounds', list(range(len(data['comm_costs']))))
            cumulative_cost = np.cumsum(data['comm_costs'])
            ax.plot(rounds, cumulative_cost, label=key.replace('_', ' '))
    
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Cumulative Communication Cost (Joules)')
    ax.set_title('Communication Cost Comparison')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'communication_cost.png'), dpi=300)
    plt.close()


def plot_comparison_table(results: Dict, output_dir: str):
    """Generate comparison table."""
    comparison_data = []
    
    for key, data in results.items():
        if 'accuracies' in data and len(data['accuracies']) > 0:
            final_accuracy = data['accuracies'][-1]
            total_comm_cost = sum(data.get('comm_costs', [0]))
            total_comp_cost = sum(data.get('comp_costs', [0]))
            
            comparison_data.append({
                'Method': key.replace('_', ' '),
                'Final Accuracy (%)': f"{final_accuracy:.2f}",
                'Comm Cost (J)': f"{total_comm_cost:.4f}",
                'Comp Cost (J)': f"{total_comp_cost:.4f}"
            })
    
    # Save as text file
    with open(os.path.join(output_dir, 'comparison_table.txt'), 'w') as f:
        f.write("Comparison Table\n")
        f.write("=" * 60 + "\n")
        for row in comparison_data:
            f.write(f"{row['Method']:30s} | Acc: {row['Final Accuracy (%)']:6s} | "
                   f"Comm: {row['Comm Cost (J)']:10s} | Comp: {row['Comp Cost (J)']:10s}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from results')
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    parser.add_argument('--output_dir', type=str, default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Generate plots
    print("Generating convergence plots...")
    plot_convergence(results, args.output_dir)
    
    print("Generating communication cost plots...")
    plot_communication_cost(results, args.output_dir)
    
    print("Generating comparison table...")
    plot_comparison_table(results, args.output_dir)
    
    print(f"Plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
